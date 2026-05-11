"""Phase 2 combined plot: NEAT vs HyperNEAT × 3 protection modes.

Two-panel layout: left = NEAT modes, right = HyperNEAT modes. Each panel
shows mean ± std across seeds. Same y-axis for direct comparison.

Usage:
    python scripts/plot_phase2_neat_vs_hyperneat.py \
        --seeds 0 1 2 -o log/phase2_neat_vs_hyperneat.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TEST_RE = re.compile(
    r"\[TEST\] Iter=(\d+),\s*#tests=\d+,\s*raw_max=(-?\d+\.\d+),\s*"
    r"raw_avg=(-?\d+\.\d+),\s*success_rate=(-?\d+\.\d+)")


def parse(path):
    its, succ = [], []
    for line in Path(path).read_text().splitlines():
        m = TEST_RE.search(line)
        if m:
            its.append(int(m.group(1)))
            succ.append(float(m.group(4)))
    return np.array(its), np.array(succ)


def gather(prefix, mode, seeds):
    per_seed = []
    for s in seeds:
        log = Path(f"log/{prefix}_seed{s}_{mode}/DeceptiveMaze.txt")
        if not log.exists():
            continue
        its, succ = parse(log)
        per_seed.append((its, succ))
    if not per_seed:
        return None
    min_len = min(len(s[1]) for s in per_seed)
    aligned = np.stack([s[1][:min_len] for s in per_seed], axis=0)
    return per_seed[0][0][:min_len], aligned.mean(0), aligned.std(0)


def plot_panel(ax, prefix, modes, seeds, title):
    colors = {"fitness": "#1f77b4", "improvement": "#2ca02c", "hybrid": "#d62728"}
    for mode in modes:
        result = gather(prefix, mode, seeds)
        if result is None:
            continue
        its, mean, std = result
        c = colors.get(mode)
        ax.plot(its, mean, color=c, marker="o", linewidth=2.0, markersize=4,
                label=f"{mode}  (final {mean[-1]:.2f} ± {std[-1]:.2f})")
        ax.fill_between(its, np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1), color=c, alpha=0.18)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test success rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    plot_panel(axes[0], "phase2", ["fitness", "improvement", "hybrid"],
               args.seeds, "NEAT — protection modes")
    plot_panel(axes[1], "phase2_hn", ["fitness", "improvement", "hybrid"],
               args.seeds, "HyperNEAT — protection modes")
    fig.suptitle(
        "Selective species protection on hard maze — NEAT vs HyperNEAT "
        f"(mean ± std across {len(args.seeds)} seeds)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
