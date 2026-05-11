"""Three-panel cross-task comparison of NEAT protection modes.

Each panel shows one task: hard maze (success rate), cartpole-easy
(test reward), cartpole-no-velocity (test reward). Within each panel,
three lines (fitness baseline, improvement, hybrid) with mean ± std
across seeds.

Usage:
    python scripts/plot_cross_task.py -o log/phase2_cross_task.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Maze logs: '[TEST] Iter=N, #tests=K, raw_max=X, raw_avg=Y, ..., success_rate=Z'
MAZE_TEST_RE = re.compile(
    r"\[TEST\] Iter=(\d+),\s*#tests=\d+,\s*raw_max=(-?\d+\.\d+),\s*"
    r"raw_avg=(-?\d+\.\d+),\s*success_rate=(-?\d+\.\d+)")

# Trainer logs (cartpole): '[TEST] Iter=N, #tests=K, max=X, avg=Y, min=Z, std=W'
TRAINER_TEST_RE = re.compile(
    r"\[TEST\] Iter=(\d+),\s*#tests=\d+,\s*max=(-?\d+\.\d+),\s*avg=(-?\d+\.\d+)")


def parse_maze(path):
    its, vals = [], []
    for line in Path(path).read_text().splitlines():
        m = MAZE_TEST_RE.search(line)
        if m:
            its.append(int(m.group(1)))
            vals.append(float(m.group(4)))  # success_rate
    return np.array(its), np.array(vals)


def parse_trainer(path):
    its, vals = [], []
    for line in Path(path).read_text().splitlines():
        m = TRAINER_TEST_RE.search(line)
        if m:
            its.append(int(m.group(1)))
            vals.append(float(m.group(3)))  # avg
    return np.array(its), np.array(vals)


def gather(prefix, mode, seeds, parser, log_name):
    per_seed = []
    for s in seeds:
        log = Path(f"log/{prefix}_seed{s}_{mode}/{log_name}")
        if not log.exists():
            continue
        its, vals = parser(log)
        per_seed.append((its, vals))
    if not per_seed:
        return None
    min_len = min(len(s[1]) for s in per_seed)
    aligned = np.stack([s[1][:min_len] for s in per_seed], axis=0)
    return per_seed[0][0][:min_len], aligned.mean(0), aligned.std(0)


def plot_panel(ax, prefix, parser, log_name, seeds, title, ylabel, ylim=None):
    colors = {"fitness": "#1f77b4", "improvement": "#2ca02c", "hybrid": "#d62728"}
    for mode in ["fitness", "improvement", "hybrid"]:
        result = gather(prefix, mode, seeds, parser, log_name)
        if result is None:
            continue
        its, mean, std = result
        c = colors[mode]
        ax.plot(its, mean, color=c, marker="o", linewidth=1.8, markersize=3,
                label=f"{mode}  ({mean[-1]:.2f} ± {std[-1]:.2f})")
        lo = mean - std
        hi = mean + std
        if ylim is not None:
            lo = np.clip(lo, ylim[0], ylim[1])
            hi = np.clip(hi, ylim[0], ylim[1])
        ax.fill_between(its, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_panel(axes[0], "phase2", parse_maze, "DeceptiveMaze.txt",
               args.seeds, "Hard maze (success rate, sparse reward)",
               "Success rate", ylim=(-0.02, 1.05))
    plot_panel(axes[1], "cp_easy", parse_trainer, "CartPoleNEAT.txt",
               args.seeds, "Cartpole-easy (test reward, dense)",
               "Test reward (avg)")
    plot_panel(axes[2], "cp_nv", parse_trainer, "CartPoleNEAT.txt",
               args.seeds, "Cartpole no-velocity (test reward, dense, POMDP)",
               "Test reward (avg)")

    fig.suptitle("Selective species protection across three tasks "
                 "(NEAT, mean ± std across 3 seeds)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
