"""Plot success_rate over training for two deceptive-maze runs.

Parses ``[TEST] Iter=N, ..., success_rate=X`` lines from each log and overlays
them. Intended for comparing NEAT+novelty vs SimpleGA+objective.

Usage:
    python scripts/plot_maze_compare.py \
        log/maze_neat_novelty/DeceptiveMaze.txt:NEAT+novelty \
        log/maze_simplega_objective/DeceptiveMaze.txt:SimpleGA+objective \
        -o log/maze_compare.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TEST_RE = re.compile(
    r"\[TEST\] Iter=(\d+),\s*#tests=\d+,\s*raw_max=(-?\d+\.\d+),\s*"
    r"raw_avg=(-?\d+\.\d+),\s*success_rate=(-?\d+\.\d+)")


def parse(path):
    its, succ, raw_avg = [], [], []
    for line in Path(path).read_text().splitlines():
        m = TEST_RE.search(line)
        if m:
            its.append(int(m.group(1)))
            raw_avg.append(float(m.group(3)))
            succ.append(float(m.group(4)))
    return its, succ, raw_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="PATH:LABEL specs")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--title", default="Deceptive maze: success rate")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    for spec, color in zip(args.logs, [f"C{i}" for i in range(len(args.logs))]):
        path_str, _, label = spec.partition(":")
        its, succ, raw = parse(path_str)
        axes[0].plot(its, succ, marker="o", linestyle="-", color=color,
                     label=label, linewidth=1.8, markersize=5)
        axes[1].plot(its, raw, marker="o", linestyle="-", color=color,
                     label=label, linewidth=1.8, markersize=5)
        print(f"{label}: {len(its)} test windows, "
              f"mean success={sum(succ)/max(1,len(succ)):.3f}")

    axes[0].set_title("Test success rate (fraction of rollouts reaching goal)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_title("Test mean raw reward")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
