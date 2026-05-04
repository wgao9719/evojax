"""Render the deceptive maze with best-policy trajectories overlaid.

Loads the ``best.npz`` saved by ``train_deceptive_maze.py`` for one or more
runs and replays the policies in fresh environments, recording per-step
positions. Plots:
  - Walls (black line segments)
  - Start (green square) and goal region (gold disk)
  - One trajectory per (run, seed), colored by run

Usage:
    python scripts/render_maze.py \
        log/maze_neat_novelty/best.npz:NEAT+novelty:neat \
        log/maze_simplega_objective/best.npz:SimpleGA+objective:simplega \
        -o log/maze_trajectories.png
"""

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from evojax.task.deceptive_maze import (
    DeceptiveMaze, GOAL_RADIUS, WORLD_SIZE, DEFAULT_MAX_STEPS,
)
from evojax.policy import MLPPolicy
from evojax.policy.neat_policy import NEATPolicy
from evojax.policy.hyperneat_policy import HyperNEATPolicy
from evojax.policy._substrate import make_grid_substrate


def build_policy(kind: str, n_inputs: int, n_outputs: int):
    if kind == "neat":
        return NEATPolicy(
            n_inputs=n_inputs, n_outputs=n_outputs,
            max_hidden=12, max_connections=120,
            n_forward_iters=4, output_act_fn="tanh")
    if kind == "hyperneat":
        substrate = make_grid_substrate(
            n_inputs=n_inputs, hidden_layers=(8, 8), n_outputs=n_outputs)
        return HyperNEATPolicy(
            substrate=substrate, max_hidden=8, max_connections=64,
            n_forward_iters=4, substrate_act_fn="tanh", output_act_fn="tanh",
            weight_scale=3.0, weight_threshold=0.2)
    if kind == "simplega" or kind == "mlp":
        return MLPPolicy(input_dim=n_inputs, hidden_dims=[32, 32],
                         output_dim=n_outputs)
    raise ValueError(f"unknown policy kind: {kind}")


def rollout_one(task, policy, params, key):
    """Run one episode, return (positions [T,2], reached: bool)."""
    state = task.reset(key[None])  # batch of 1
    p_state = policy.reset(state)
    params_b = params[None, :]
    positions = [np.asarray(state.pos[0])]
    for _ in range(task.max_steps):
        actions, p_state = policy.get_actions(state, params_b, p_state)
        state, _, done = task.step(state, actions)
        positions.append(np.asarray(state.pos[0]))
        if bool(done[0]):
            break
    return np.array(positions), bool(state.reached[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+",
                        help="best.npz:LABEL:POLICY_KIND triples")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--n-trajectories", type=int, default=8,
                        help="Rollouts per spec (different reset noise seeds).")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--layout", choices=["easy", "hard"], default="easy")
    parser.add_argument("--title", default="Deceptive maze — best-policy trajectories")
    args = parser.parse_args()

    task = DeceptiveMaze(max_steps=args.max_steps, sparse_reward=True,
                         layout=args.layout)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # --- Walls ---
    walls_np = np.asarray(task.walls)
    for x1, y1, x2, y2 in walls_np:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.5)

    # --- Start + goal ---
    sp = np.asarray(task.start_pos)
    gp = np.asarray(task.goal_pos)
    ax.add_patch(Rectangle((sp[0] - 1.5, sp[1] - 1.5), 3.0, 3.0,
                           facecolor="#2ca02c", edgecolor="black",
                           label="Start", zorder=5))
    ax.add_patch(Circle((gp[0], gp[1]), GOAL_RADIUS,
                        facecolor="gold", edgecolor="black",
                        alpha=0.7, label="Goal region", zorder=4))

    # --- Trajectories ---
    colors = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e"]
    for spec_idx, spec in enumerate(args.specs):
        parts = spec.split(":")
        if len(parts) < 3:
            raise SystemExit(f"spec must be PATH:LABEL:KIND, got: {spec}")
        path, label, kind = parts[0], parts[1], parts[2]
        data = np.load(path)
        params = jnp.asarray(data["params"])
        n_inputs = task.obs_shape[0]
        n_outputs = task.act_shape[0]
        policy = build_policy(kind, n_inputs, n_outputs)
        color = colors[spec_idx % len(colors)]

        n_reached = 0
        for j in range(args.n_trajectories):
            key = jax.random.PRNGKey(1000 + j)
            traj, reached = rollout_one(task, policy, params, key)
            n_reached += int(reached)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.55,
                    linewidth=1.4,
                    label=(f"{label} ({n_reached}/{args.n_trajectories} reach goal)"
                           if j == args.n_trajectories - 1 else None),
                    zorder=3)
            # Mark endpoint
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color,
                       marker="x" if not reached else "*",
                       s=60, zorder=6, linewidths=2)

        print(f"{label}: {n_reached}/{args.n_trajectories} trajectories reached goal")

    ax.set_xlim(-2, WORLD_SIZE + 2)
    ax.set_ylim(-2, WORLD_SIZE + 2)
    ax.set_aspect("equal")
    ax.set_title(args.title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
