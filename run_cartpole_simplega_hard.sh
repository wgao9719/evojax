#!/bin/bash
set -euo pipefail

PY="${PY:-$(which python)}"

XLA_FLAGS='--xla_force_host_platform_device_count=8' \
JAX_PLATFORMS=cpu PYTHONPATH=. "$PY" -c "
from evojax import Trainer
from evojax.task.cartpole import CartPoleSwingUp
from evojax.policy import MLPPolicy
from evojax.algo import SimpleGA
tr = CartPoleSwingUp(test=False, harder=True)
te = CartPoleSwingUp(test=True,  harder=True)
p = MLPPolicy(input_dim=tr.obs_shape[0], hidden_dims=[32, 32], output_dim=tr.act_shape[0])
s = SimpleGA(pop_size=128, param_size=p.num_params, sigma=0.05, seed=0)
Trainer(policy=p, solver=s, train_task=tr, test_task=te,
        max_iter=1000, log_interval=10, test_interval=50,
        n_repeats=16, n_evaluations=32,
        seed=0, log_dir='./log/cartpole_simplega_hard').run()
"
