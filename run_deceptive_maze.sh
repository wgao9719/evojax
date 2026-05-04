#!/bin/bash
# Run the deceptive maze comparison: NEAT+novelty vs SimpleGA+objective.
# At pop=64 with a sparse-reward + pie-slice-obs maze, SimpleGA never finds
# the goal (no exploration signal under sparse reward), while NEAT+novelty
# achieves ~87% test success by maintaining behavior diversity.
set -euo pipefail

PY="${PY:-/Users/willi1/miniconda3/envs/evo/bin/python}"

XLA_FLAGS='--xla_force_host_platform_device_count=8' \
JAX_PLATFORMS=cpu PYTHONPATH=. "$PY" examples/train_deceptive_maze.py \
    --algo=neat --use-novelty --novelty-threshold=6.0 \
    --pop-size=64 --max-iter=400 --n-repeats=3 \
    --num-tests=32 --test-interval=20 --log-interval=10 \
    --max-steps=200 --seed=0 &

XLA_FLAGS='--xla_force_host_platform_device_count=8' \
JAX_PLATFORMS=cpu PYTHONPATH=. "$PY" examples/train_deceptive_maze.py \
    --algo=simplega \
    --pop-size=64 --max-iter=400 --n-repeats=3 \
    --num-tests=32 --test-interval=20 --log-interval=10 \
    --max-steps=200 --seed=0 &

wait

"$PY" scripts/plot_maze_compare.py \
    log/maze_neat_novelty/DeceptiveMaze.txt:NEAT+novelty \
    log/maze_simplega_objective/DeceptiveMaze.txt:SimpleGA+objective \
    -o log/maze_compare.png \
    --title "Deceptive maze: NEAT+novelty vs SimpleGA+objective (pop=64)"
