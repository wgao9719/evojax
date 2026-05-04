#!/bin/bash
set -euo pipefail

PY="${PY:-$(which python)}"

XLA_FLAGS='--xla_force_host_platform_device_count=8' \
JAX_PLATFORMS=cpu PYTHONPATH=. "$PY" examples/train_cartpole_neat.py \
    --max-iter=1000 --pop-size=128 \
    --num-tests=32 --n-repeats=16 \
    --max-hidden=16 --max-connections=160 \
    --compat-threshold=2.0 --add-node-rate=0.02 --add-connection-rate=0.03
