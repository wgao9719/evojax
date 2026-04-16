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

"""Train NEAT on the classic CartPole swing up task.

Example:
    python train_cartpole_neat.py --easy --max-iter=500 --pop-size=128
"""

import argparse
import os

from evojax import Trainer
from evojax.task.cartpole import CartPoleSwingUp
from evojax.algo import NEAT
from evojax.policy import NEATPolicy
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--max-hidden", type=int, default=10)
    parser.add_argument("--max-connections", type=int, default=80)
    parser.add_argument("--n-forward-iters", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--test-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compat-threshold", type=float, default=3.0)
    parser.add_argument("--parsimony-weight", type=float, default=0.001)
    parser.add_argument("--add-node-rate", type=float, default=0.03)
    parser.add_argument("--add-connection-rate", type=float, default=0.05)
    parser.add_argument("--gpu-id", type=str)
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_known_args()[0]


def main(config):
    hard = not config.easy
    log_dir = "./log/cartpole_neat_{}".format("hard" if hard else "easy")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name="CartPoleNEAT", log_dir=log_dir,
                                 debug=config.debug)

    logger.info("EvoJAX CartPole NEAT Demo (%s)", "hard" if hard else "easy")
    logger.info("=" * 40)

    train_task = CartPoleSwingUp(test=False, harder=hard)
    test_task = CartPoleSwingUp(test=True, harder=hard)

    policy = NEATPolicy(
        n_inputs=train_task.obs_shape[0],
        n_outputs=train_task.act_shape[0],
        max_hidden=config.max_hidden,
        max_connections=config.max_connections,
        n_forward_iters=config.n_forward_iters,
        output_act_fn="tanh",
    )

    solver = NEAT(
        pop_size=config.pop_size,
        n_inputs=train_task.obs_shape[0],
        n_outputs=train_task.act_shape[0],
        max_hidden=config.max_hidden,
        max_connections=config.max_connections,
        compat_threshold=config.compat_threshold,
        parsimony_weight=config.parsimony_weight,
        add_connection_rate=config.add_connection_rate,
        add_node_rate=config.add_node_rate,
        seed=config.seed,
        logger=logger,
    )
    assert policy.num_params == solver.param_size, (
        "Policy num_params ({}) and NEAT param_size ({}) must match".format(
            policy.num_params, solver.param_size))

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    main(cfg)
