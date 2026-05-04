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

"""Deceptive maze: NEAT/SimpleGA, with optional novelty search.

Bypasses :class:`evojax.Trainer` because novelty search needs to substitute
the rollout fitness with a per-individual novelty score (computed from final
positions) before calling ``solver.tell``. Doing that cleanly inside the
stock Trainer would require either a hook or a custom callback path; running
a small loop here is simpler and keeps the change isolated.

Examples:
    # NEAT + novelty search (the canonical "should beat objective" config)
    python examples/train_deceptive_maze.py --algo=neat --use-novelty
    # SimpleGA on raw shaped reward (the deception trap)
    python examples/train_deceptive_maze.py --algo=simplega
"""

import argparse
import os
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from evojax.task.deceptive_maze import DeceptiveMaze, GOAL_POS, GOAL_RADIUS
from evojax.policy import MLPPolicy
from evojax.policy.neat_policy import NEATPolicy
from evojax.policy.hyperneat_policy import HyperNEATPolicy
from evojax.policy._substrate import make_grid_substrate
from evojax.algo import NEAT, SimpleGA, NoveltyArchive
from evojax.sim_mgr import SimManager
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["neat", "simplega", "hyperneat"],
                        default="neat")
    parser.add_argument("--use-novelty", action="store_true",
                        help="Score individuals by behavior novelty instead of raw reward.")
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--num-tests", type=int, default=32)
    parser.add_argument("--test-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override default max_steps for the chosen layout.")
    parser.add_argument("--layout", choices=["easy", "hard"], default="easy")
    parser.add_argument("--dense-reward", action="store_true",
                        help="Use L2-shaped reward instead of sparse (success-only).")
    # NEAT-specific
    parser.add_argument("--max-hidden", type=int, default=12)
    parser.add_argument("--max-connections", type=int, default=120)
    parser.add_argument("--n-forward-iters", type=int, default=4)
    parser.add_argument("--add-node-rate", type=float, default=0.06)
    parser.add_argument("--add-connection-rate", type=float, default=0.12)
    parser.add_argument("--compat-threshold", type=float, default=2.0)
    # HyperNEAT-specific
    parser.add_argument("--hyperneat-hidden", type=int, nargs="*", default=[8, 8],
                        help="Substrate hidden-layer sizes (e.g. 8 8).")
    parser.add_argument("--hyperneat-cppn-hidden", type=int, default=8)
    parser.add_argument("--hyperneat-cppn-connections", type=int, default=64)
    parser.add_argument("--hyperneat-weight-scale", type=float, default=3.0)
    parser.add_argument("--hyperneat-weight-threshold", type=float, default=0.2)
    # Novelty-specific
    parser.add_argument("--novelty-k", type=int, default=15)
    parser.add_argument("--novelty-threshold", type=float, default=4.0)
    parser.add_argument("--novelty-archive-size", type=int, default=2500)
    parser.add_argument("--novelty-blend", type=float, default=0.0,
                        help="Weight on raw reward when use-novelty (0 = pure novelty).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_known_args()[0]


def build_policy_and_solver(cfg, train_task):
    n_inputs = train_task.obs_shape[0]
    n_outputs = train_task.act_shape[0]
    if cfg.algo == "neat":
        policy = NEATPolicy(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            max_hidden=cfg.max_hidden,
            max_connections=cfg.max_connections,
            n_forward_iters=cfg.n_forward_iters,
            output_act_fn="tanh",
        )
        solver = NEAT(
            pop_size=cfg.pop_size,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            max_hidden=cfg.max_hidden,
            max_connections=cfg.max_connections,
            compat_threshold=cfg.compat_threshold,
            parsimony_weight=0.0,
            add_connection_rate=cfg.add_connection_rate,
            add_node_rate=cfg.add_node_rate,
            seed=cfg.seed,
        )
        assert policy.num_params == solver.param_size
    elif cfg.algo == "hyperneat":
        substrate = make_grid_substrate(
            n_inputs=n_inputs,
            hidden_layers=tuple(cfg.hyperneat_hidden),
            n_outputs=n_outputs,
        )
        policy = HyperNEATPolicy(
            substrate=substrate,
            max_hidden=cfg.hyperneat_cppn_hidden,
            max_connections=cfg.hyperneat_cppn_connections,
            n_forward_iters=cfg.n_forward_iters,
            substrate_act_fn="tanh",
            output_act_fn="tanh",
            weight_scale=cfg.hyperneat_weight_scale,
            weight_threshold=cfg.hyperneat_weight_threshold,
        )
        solver = NEAT(
            pop_size=cfg.pop_size,
            n_inputs=substrate.cppn_input_dim,
            n_outputs=1,
            max_hidden=cfg.hyperneat_cppn_hidden,
            max_connections=cfg.hyperneat_cppn_connections,
            compat_threshold=cfg.compat_threshold,
            parsimony_weight=0.0,
            add_connection_rate=cfg.add_connection_rate,
            add_node_rate=cfg.add_node_rate,
            activation_set=["tanh", "sin", "gauss", "abs", "sigmoid", "identity"],
            mutate_activation_rate=0.05,
            seed=cfg.seed,
        )
        assert policy.num_params == solver.param_size, (
            f"HyperNEAT policy num_params ({policy.num_params}) != "
            f"NEAT solver param_size ({solver.param_size})")
    else:
        policy = MLPPolicy(input_dim=n_inputs,
                           hidden_dims=[32, 32],
                           output_dim=n_outputs)
        solver = SimpleGA(pop_size=cfg.pop_size,
                          param_size=policy.num_params,
                          sigma=0.05,
                          seed=cfg.seed)
    return policy, solver


def extract_bds(final_states) -> np.ndarray:
    """Mean final position across n_repeats — one BD per individual."""
    pos = np.asarray(final_states.pos)  # (pop_size, n_repeats, 2)
    return pos.mean(axis=1)


def reached_rate(final_states) -> Tuple[float, float]:
    """Fraction of test rollouts that reached the goal."""
    reached = np.asarray(final_states.reached)
    return float(reached.mean())


def main(cfg):
    if cfg.log_dir is None:
        suffix = "novelty" if cfg.use_novelty else "objective"
        cfg.log_dir = f"./log/maze_{cfg.algo}_{suffix}"
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = util.create_logger(name="DeceptiveMaze", log_dir=cfg.log_dir,
                                debug=cfg.debug)
    logger.info("=" * 50)
    logger.info("Deceptive Maze — algo=%s, use_novelty=%s", cfg.algo, cfg.use_novelty)
    logger.info("pop=%d, iters=%d, n_repeats=%d, max_steps=%d",
                cfg.pop_size, cfg.max_iter, cfg.n_repeats, cfg.max_steps)

    sparse = not cfg.dense_reward
    train_task = DeceptiveMaze(max_steps=cfg.max_steps, test=False,
                               sparse_reward=sparse, layout=cfg.layout)
    test_task = DeceptiveMaze(max_steps=cfg.max_steps, test=True,
                              sparse_reward=sparse, layout=cfg.layout)
    policy, solver = build_policy_and_solver(cfg, train_task)
    logger.info("Policy num_params=%d", policy.num_params)

    sim_mgr = SimManager(
        n_repeats=cfg.n_repeats,
        test_n_repeats=1,
        pop_size=cfg.pop_size,
        n_evaluations=cfg.num_tests,
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=cfg.seed,
        logger=logger,
    )

    archive = (NoveltyArchive(k=cfg.novelty_k,
                              threshold=cfg.novelty_threshold,
                              max_size=cfg.novelty_archive_size)
               if cfg.use_novelty else None)

    best_raw_score = -float("inf")
    best_params = None

    logger.info("Start to train for %d iterations.", cfg.max_iter)
    for it in range(1, cfg.max_iter + 1):
        params = solver.ask()
        scores, final_states = sim_mgr.eval_params(params, test=False)
        raw_scores = np.asarray(scores)

        if cfg.use_novelty:
            bds = extract_bds(final_states)
            novelty = archive.score(bds)
            archive.update(bds)
            blend = cfg.novelty_blend
            tell_scores = (1 - blend) * novelty + blend * raw_scores
        else:
            tell_scores = raw_scores

        solver.tell(jnp.asarray(tell_scores))

        # Track best by RAW reward (the actual task objective), not novelty.
        gen_best_idx = int(np.argmax(raw_scores))
        gen_best = float(raw_scores[gen_best_idx])
        if gen_best > best_raw_score:
            best_raw_score = gen_best
            best_params = np.asarray(params)[gen_best_idx].copy()

        if it % cfg.log_interval == 0:
            extra = (f", arch={archive.size}, novelty_avg={float(novelty.mean()):.2f}"
                     if cfg.use_novelty else "")
            logger.info("Iter=%d, raw_max=%.3f, raw_avg=%.3f, raw_min=%.3f%s",
                        it, float(raw_scores.max()), float(raw_scores.mean()),
                        float(raw_scores.min()), extra)

        if it % cfg.test_interval == 0 and best_params is not None:
            # Test the best-by-raw-reward genome we've seen, not solver.best_params
            # (which under novelty search is the most-novel genome, not the most
            # goal-reaching one).
            test_scores, test_final = sim_mgr.eval_params(
                jnp.asarray(best_params), test=True)
            test_raw = np.asarray(test_scores)
            success_rate = reached_rate(test_final)
            logger.info(
                "[TEST] Iter=%d, #tests=%d, raw_max=%.3f, raw_avg=%.3f, "
                "success_rate=%.3f",
                it, cfg.num_tests,
                float(test_raw.max()), float(test_raw.mean()), success_rate)

    # Final
    if best_params is not None:
        np.savez(os.path.join(cfg.log_dir, "best.npz"),
                 params=best_params, best_raw_score=best_raw_score)
    logger.info("Training done. best_raw_score=%.3f", best_raw_score)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
