# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deceptive maze navigation task (Lehman & Stanley 2011 style).

The agent is a 2D point in a continuous 100x100 world bounded by walls and
broken up by interior walls that form a deceptive U-shape: heading straight
toward the goal pulls the agent into a dead-end. Sensors are K rangefinder
rays plus the goal direction in egocentric coordinates; the action is
``(turn_rate, forward_speed)`` in ``[-1, 1]``.

Reward is shaped as ``-distance_to_goal / WORLD_SIZE`` per step, plus a
large terminal bonus for reaching the goal. With objective-based search this
shaping is what makes the task deceptive — the dead-end is closer in L2 than
the actual route. Novelty search ignores the shaped reward and explores
behavior space instead, which is what makes the canonical NEAT-wins demo
work.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


WORLD_SIZE = 100.0

# Agent kinematics
MAX_TURN_RATE = 0.35  # radians per step
MAX_SPEED = 2.0       # world units per step
AGENT_RADIUS = 1.5    # for collision rejection

GOAL_RADIUS = 5.0
GOAL_BONUS = 100.0

# Sensors
N_RAYS = 8
RAY_FOV = jnp.pi  # rays span ±pi/2 from heading (180° FOV)
MAX_RAY_DIST = 30.0

# Episode
DEFAULT_MAX_STEPS = 200

# How much to perturb the start position on reset (uniform in a square).
# Small enough that the deception still bites; large enough that test rollouts
# aren't identical clones.
START_NOISE = 2.0


_OUTER = [
    (0.0, 0.0, WORLD_SIZE, 0.0),
    (WORLD_SIZE, 0.0, WORLD_SIZE, WORLD_SIZE),
    (WORLD_SIZE, WORLD_SIZE, 0.0, WORLD_SIZE),
    (0.0, WORLD_SIZE, 0.0, 0.0),
]


# ----------------------------------------------------------------------------
# Layout: "easy" (2 walls, ~200-unit path)
# Two parallel horizontal walls with offset gaps. Wall B at y=30 has its
# gap on the far left, wall A at y=70 has its gap on the far right. Heading
# straight up bonks A and the agent must travel rightward (away from goal in
# L2) to find A's gap, then back left to the goal — the canonical Lehman/
# Stanley deception in its smallest form.
# ----------------------------------------------------------------------------
_EASY_TRAP = [
    (0.0, 70.0, 85.0, 70.0),    # Wall A (upper) — gap at x in [85, 100]
    (15.0, 30.0, 100.0, 30.0),  # Wall B (lower) — gap at x in [0, 15]
]

# ----------------------------------------------------------------------------
# Layout: "hard" (4-wall snake + a deceptive dead-end pocket)
# Four horizontal walls with alternating LEFT / RIGHT gaps force the agent
# through a Z-shaped path of ~350 units against a 500-unit budget — too long
# for SGA random walks to stumble through and tight enough that even
# novelty methods need to explore systematically. The small interior box
# straddles the L2 line near the start and acts as a sticky pocket: agents
# that head straight for the goal get caught inside and waste their step
# budget rattling around.
# ----------------------------------------------------------------------------
_HARD_TRAP = [
    # 4 horizontal walls, alternating gaps (left / right / left / right)
    (12.0, 20.0, 100.0, 20.0),  # gap x in [0, 12]
    (0.0, 40.0, 88.0, 40.0),    # gap x in [88, 100]
    (12.0, 60.0, 100.0, 60.0),  # gap x in [0, 12]
    (0.0, 80.0, 88.0, 80.0),    # gap x in [88, 100]
    # Dead-end pocket sitting on the L2-greedy descent line near the start.
    # Closed on three sides; opens downward so the agent can drift in but
    # has to back out without making L2 progress.
    (40.0, 8.0, 40.0, 14.0),
    (60.0, 8.0, 60.0, 14.0),
    (40.0, 14.0, 60.0, 14.0),
]


_LAYOUTS = {
    "easy": {
        "walls": _OUTER + _EASY_TRAP,
        "start_pos": (50.0, 10.0),
        "start_heading": jnp.pi / 2.0,
        "goal_pos": (5.0, 90.0),
        "default_max_steps": 200,
    },
    "hard": {
        "walls": _OUTER + _HARD_TRAP,
        "start_pos": (50.0, 4.0),
        "start_heading": jnp.pi / 2.0,
        "goal_pos": (50.0, 96.0),
        "default_max_steps": 250,
    },
}


# Default layout (preserves backwards-compatible WALLS / GOAL_POS / START_POS
# module constants for any callers that import them directly).
_DEFAULT_LAYOUT = "easy"
WALLS = jnp.array(_LAYOUTS[_DEFAULT_LAYOUT]["walls"], dtype=jnp.float32)
GOAL_POS = jnp.array(_LAYOUTS[_DEFAULT_LAYOUT]["goal_pos"], dtype=jnp.float32)
START_POS = jnp.array(_LAYOUTS[_DEFAULT_LAYOUT]["start_pos"], dtype=jnp.float32)
START_HEADING = jnp.array(_LAYOUTS[_DEFAULT_LAYOUT]["start_heading"],
                           dtype=jnp.float32)


@dataclass
class MazeState(TaskState):
    obs: jnp.ndarray
    pos: jnp.ndarray         # (2,) agent position
    heading: jnp.ndarray     # () agent heading in radians
    steps: jnp.int32
    reached: jnp.int32       # 1 once goal reached, sticky
    key: jnp.ndarray


def _segments_intersect(p1, p2, q1, q2):
    """Whether segment p1-p2 intersects segment q1-q2.

    Standard 2D segment-segment intersection via parametric line solve. Returns
    a JAX bool. Degenerate (parallel) cases return False, which is the safe
    behavior for collision rejection.
    """
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]
    denom = rx * sy - ry * sx
    safe_denom = jnp.where(jnp.abs(denom) < 1e-9, 1.0, denom)
    qpx = q1[0] - p1[0]
    qpy = q1[1] - p1[1]
    t = (qpx * sy - qpy * sx) / safe_denom
    u = (qpx * ry - qpy * rx) / safe_denom
    hit = (jnp.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return hit


def _ray_segment_dist(origin, direction, w1, w2, max_dist):
    """Distance from ``origin`` along unit ``direction`` to wall segment.

    Returns ``max_dist`` if no valid hit (parallel, behind ray, or outside
    segment span). The ray is treated as a half-line ``origin + t*direction``,
    ``t >= 0``.
    """
    rx, ry = direction[0], direction[1]
    sx, sy = w2[0] - w1[0], w2[1] - w1[1]
    denom = rx * sy - ry * sx
    safe_denom = jnp.where(jnp.abs(denom) < 1e-9, 1.0, denom)
    qpx = w1[0] - origin[0]
    qpy = w1[1] - origin[1]
    t = (qpx * sy - qpy * sx) / safe_denom
    u = (qpx * ry - qpy * rx) / safe_denom
    valid = (jnp.abs(denom) > 1e-9) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return jnp.where(valid, jnp.minimum(t, max_dist), max_dist)


def _cast_one_ray(origin, direction, walls):
    """Distance from origin along direction to nearest wall (or MAX_RAY_DIST)."""
    w1 = walls[:, :2]
    w2 = walls[:, 2:]
    dists = jax.vmap(
        lambda a, b: _ray_segment_dist(origin, direction, a, b, MAX_RAY_DIST)
    )(w1, w2)
    return jnp.min(dists)


def _all_rays(pos, heading, walls):
    """Cast N_RAYS rays evenly spaced over RAY_FOV centered on ``heading``."""
    angles = heading + jnp.linspace(-RAY_FOV / 2, RAY_FOV / 2, N_RAYS)
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    dists = jax.vmap(lambda d: _cast_one_ray(pos, d, walls))(dirs)
    return dists


def _check_collision(old_pos, new_pos, walls):
    """True if the line from old_pos to new_pos crosses any wall segment."""
    w1 = walls[:, :2]
    w2 = walls[:, 2:]
    hits = jax.vmap(lambda a, b: _segments_intersect(old_pos, new_pos, a, b))(w1, w2)
    return jnp.any(hits)


def _build_obs(pos, heading, walls, goal_pos):
    """Observation = N_RAYS rangefinder distances + 4 pie-slice goal sensors.

    The pie-slice sensors are 4 binary-ish values indicating which quadrant of
    the agent's egocentric frame the goal lies in. Matches the Lehman &
    Stanley 2011 setup — the agent knows the goal *direction* coarsely but
    not the *path*, which is what makes the geometry deceptive.
    """
    ray_dists = _all_rays(pos, heading, walls) / MAX_RAY_DIST
    to_goal = goal_pos - pos
    cos_h, sin_h = jnp.cos(heading), jnp.sin(heading)
    ego_x = cos_h * to_goal[0] + sin_h * to_goal[1]
    ego_y = -sin_h * to_goal[0] + cos_h * to_goal[1]
    pie = jnp.array([
        ((ego_x > 0) & (ego_y > 0)).astype(jnp.float32),
        ((ego_x > 0) & (ego_y <= 0)).astype(jnp.float32),
        ((ego_x <= 0) & (ego_y > 0)).astype(jnp.float32),
        ((ego_x <= 0) & (ego_y <= 0)).astype(jnp.float32),
    ])
    return jnp.concatenate([ray_dists, pie], axis=0)


def _make_reset_fn(walls, goal_pos, start_pos, start_heading):

    def _reset_single(key):
        pos_key, heading_key, next_key = random.split(key, 3)
        pos = start_pos + random.uniform(
            pos_key, shape=(2,), minval=-START_NOISE, maxval=START_NOISE)
        heading = start_heading + random.uniform(
            heading_key, shape=(), minval=-0.1, maxval=0.1)
        obs = _build_obs(pos, heading, walls, goal_pos)
        return MazeState(
            obs=obs, pos=pos, heading=heading,
            steps=jnp.zeros((), dtype=jnp.int32),
            reached=jnp.zeros((), dtype=jnp.int32),
            key=next_key,
        )

    return _reset_single


def _make_step_fn(walls, goal_pos, max_steps, sparse_reward):
    """Returns a per-agent step function bound to a layout's walls/goal."""

    def _step_single(state, action):
        action = jnp.clip(action, -1.0, 1.0)
        turn = action[0] * MAX_TURN_RATE
        speed = (action[1] + 1.0) * 0.5 * MAX_SPEED  # remap [-1,1] -> [0, MAX]

        new_heading = state.heading + turn
        delta = jnp.array([jnp.cos(new_heading), jnp.sin(new_heading)]) * speed
        proposed = state.pos + delta

        collided = _check_collision(state.pos, proposed, walls)
        new_pos = jnp.where(collided, state.pos, proposed)

        dist = jnp.linalg.norm(goal_pos - new_pos)
        reached_now = (dist < GOAL_RADIUS).astype(jnp.int32)
        reached = jnp.maximum(state.reached, reached_now)

        bonus = GOAL_BONUS * (reached_now * (1 - state.reached))
        if sparse_reward:
            reward = bonus
        else:
            reward = -dist / WORLD_SIZE + bonus

        new_steps = state.steps + 1
        done = (reached > 0) | (new_steps >= max_steps)
        obs = _build_obs(new_pos, new_heading, walls, goal_pos)

        new_state = MazeState(
            obs=obs, pos=new_pos, heading=new_heading,
            steps=new_steps, reached=reached, key=state.key,
        )
        return new_state, reward, done.astype(jnp.int32)

    return _step_single


def get_layout(name: str):
    """Return ``(walls, goal_pos, start_pos, start_heading, default_max_steps)``."""
    if name not in _LAYOUTS:
        raise ValueError(
            f"unknown maze layout {name!r}; choices: {list(_LAYOUTS)}")
    cfg = _LAYOUTS[name]
    return (
        jnp.asarray(cfg["walls"], dtype=jnp.float32),
        jnp.asarray(cfg["goal_pos"], dtype=jnp.float32),
        jnp.asarray(cfg["start_pos"], dtype=jnp.float32),
        jnp.asarray(cfg["start_heading"], dtype=jnp.float32),
        int(cfg["default_max_steps"]),
    )


class DeceptiveMaze(VectorizedTask):
    """2D maze with deceptive geometry. Pick ``layout`` to swap the wall set."""

    def __init__(self,
                 max_steps: int = None,
                 test: bool = False,
                 sparse_reward: bool = True,
                 layout: str = "easy"):
        walls, goal_pos, start_pos, start_heading, default_max_steps = (
            get_layout(layout))
        self.layout = layout
        self.walls = walls
        self.goal_pos = goal_pos
        self.start_pos = start_pos
        self.start_heading = start_heading
        self.max_steps = int(max_steps) if max_steps is not None else default_max_steps
        self.test = test
        self.sparse_reward = sparse_reward
        self.obs_shape = tuple([N_RAYS + 4, ])
        self.act_shape = tuple([2, ])
        self.multi_agent_training = False

        self._reset_fn = jax.jit(jax.vmap(
            _make_reset_fn(walls, goal_pos, start_pos, start_heading)))
        self._step_fn = jax.jit(jax.vmap(
            _make_step_fn(walls, goal_pos, self.max_steps, sparse_reward)))

    def reset(self, key: jnp.ndarray) -> MazeState:
        return self._reset_fn(key)

    def step(self,
             state: MazeState,
             action: jnp.ndarray) -> Tuple[MazeState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
