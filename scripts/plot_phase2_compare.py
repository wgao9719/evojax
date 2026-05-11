"""Phase 2 headline plot: success-rate-over-time, mean ± std across seeds.

Reads multiple seed × mode runs of train_deceptive_maze and aggregates to
one curve per mode (mean across seeds, shaded ±std band).

Usage:
    python scripts/plot_phase2_compare.py \
        --log-root log --modes fitness improvement hybrid --seeds 0 1 2 \
        --prefix phase2 -o log/phase2_headline.png
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="log")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--prefix", default="phase2",
                        help="Log dirs are {prefix}_seed{S}_{mode}/")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--title", default="Phase 2 — species protection modes (mean ± std across seeds)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"fitness": "#1f77b4", "improvement": "#2ca02c", "hybrid": "#d62728"}

    for mode in args.modes:
        per_seed = []
        common_iters = None
        for seed in args.seeds:
            d = Path(args.log_root) / f"{args.prefix}_seed{seed}_{mode}"
            log = d / "DeceptiveMaze.txt"
            if not log.exists():
                print(f"missing {log}, skipping")
                continue
            its, succ = parse(log)
            per_seed.append((its, succ))
            if common_iters is None:
                common_iters = its
        if not per_seed:
            continue

        # Align to the shortest run
        min_len = min(len(s[1]) for s in per_seed)
        aligned = np.stack([s[1][:min_len] for s in per_seed], axis=0)
        iters = per_seed[0][0][:min_len]
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)
        c = colors.get(mode, None)
        ax.plot(iters, mean, color=c, marker="o", linewidth=2.0, markersize=4,
                label=f"{mode}  (final {mean[-1]:.2f} ± {std[-1]:.2f})")
        ax.fill_between(iters, np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1), color=c, alpha=0.18)
        print(f"{mode}: {len(per_seed)} seeds, final mean={mean[-1]:.3f} std={std[-1]:.3f}")

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test success rate")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
