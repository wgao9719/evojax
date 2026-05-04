# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Overlay fitness curves from multiple evojax training logs.

Parses the same ``Iter=... max=... avg=... min=... std=...`` lines as
``plot_neat_log.py``. Each log gets its own color; train is a solid line,
test points are markers. Three panels: avg (with std band), max, min.

Usage:
    python scripts/plot_compare_logs.py \
        log/slimevolley_neat/SlimeVolleyNEAT.txt:NEAT \
        log/slimevolley_simplega/Trainer.txt:SimpleGA \
        -o log/slimevolley_compare.png --title "SlimeVolley: NEAT vs SimpleGA"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LINE_RE = re.compile(
    r"(\[TEST\] )?Iter=(\d+),\s*(?:size=\d+|#tests=\d+),\s*"
    r"max=(-?\d+\.\d+),\s*avg=(-?\d+\.\d+),\s*"
    r"min=(-?\d+\.\d+),\s*std=(-?\d+\.\d+)")

RUN_START_RE = re.compile(r"Start to train for (\d+) iterations")


def parse_log(path: Path):
    last_start = -1
    with path.open() as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if RUN_START_RE.search(line):
            last_start = idx
    tail = lines[last_start + 1:] if last_start >= 0 else lines

    train = {"iter": [], "max": [], "avg": [], "min": [], "std": []}
    test = {"iter": [], "max": [], "avg": [], "min": [], "std": []}
    for line in tail:
        m = LINE_RE.search(line)
        if not m:
            continue
        is_test, it, mx, avg, mn, std = m.groups()
        bucket = test if is_test else train
        bucket["iter"].append(int(it))
        bucket["max"].append(float(mx))
        bucket["avg"].append(float(avg))
        bucket["min"].append(float(mn))
        bucket["std"].append(float(std))
    return train, test


def _smooth(arr, window):
    """Centered rolling mean with shrinking window at the edges.

    Avoids the zero-padding artifact of ``np.convolve(..., mode='same')``,
    which drags boundary points toward zero. At each position we average over
    only the values that actually exist, so the curve faithfully tracks the
    raw data at the start and end.
    """
    if window <= 1 or len(arr) < 2:
        return np.asarray(arr, dtype=float)
    a = np.asarray(arr, dtype=float)
    n = len(a)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = a[lo:hi].mean()
    return out


def plot(runs, title, out_path, test_only=False, smooth=1, ymax=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    colors = [f"C{i}" for i in range(len(runs))]

    # Panel 1: avg (+ ±std band when train is shown)
    ax = axes[0]
    for (label, train, test), color in zip(runs, colors):
        if not test_only and train["iter"]:
            it = np.array(train["iter"])
            avg = _smooth(train["avg"], smooth)
            std = _smooth(train["std"], smooth)
            ax.plot(it, avg, color=color, label=f"{label} train avg",
                    linewidth=1.5)
            ax.fill_between(it, avg - std, avg + std, color=color, alpha=0.15)
        if test["iter"]:
            ax.plot(test["iter"], _smooth(test["avg"], smooth),
                    color=color, marker="o", linestyle="--",
                    label=f"{label} test avg",
                    linewidth=1.5, markersize=6)
    suffix = f" (smooth={smooth})" if smooth > 1 else ""
    ax.set_title(("Average fitness" if test_only
                  else "Average fitness (band = ±std)") + suffix)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    # Panels 2 & 3: max / min
    for panel_idx, (field, panel_title) in enumerate(
            [("max", "Best fitness"), ("min", "Worst fitness")], start=1):
        ax = axes[panel_idx]
        for (label, train, test), color in zip(runs, colors):
            if not test_only and train["iter"]:
                ax.plot(train["iter"], _smooth(train[field], smooth),
                        color=color, label=f"{label} train", linewidth=1.5)
            if test["iter"]:
                ax.plot(test["iter"], _smooth(test[field], smooth),
                        color=color, marker="o", linestyle="--",
                        label=f"{label} test",
                        linewidth=1.5, markersize=6)
        ax.set_title(panel_title + suffix)
        ax.set_xlabel("Iteration")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        if ymax is not None:
            ax.set_ylim(top=ymax)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved plot -> {out_path}")


def _parse_spec(spec: str):
    if ":" in spec:
        path_str, label = spec.rsplit(":", 1)
    else:
        path_str, label = spec, Path(spec).stem
    return Path(path_str), label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=str,
                        help="One or more log specs: PATH or PATH:LABEL.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Training comparison")
    parser.add_argument("--test-only", action="store_true",
                        help="Only plot [TEST] entries.")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Clip data to iterations <= this value.")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling-mean window (1 = no smoothing).")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Cap y-axis at this value on all panels.")
    args = parser.parse_args()

    runs = []
    for spec in args.logs:
        path, label = _parse_spec(spec)
        train, test = parse_log(path)
        if args.max_iter is not None:
            for bucket in (train, test):
                keep = [i for i, it in enumerate(bucket["iter"])
                        if it <= args.max_iter]
                for key in bucket:
                    bucket[key] = [bucket[key][i] for i in keep]
        print(f"{label}: {len(train['iter'])} train, {len(test['iter'])} test")
        if not train["iter"] and not test["iter"]:
            raise SystemExit(f"No parseable Iter lines in {path}")
        runs.append((label, train, test))

    plot(runs, args.title, Path(args.output), test_only=args.test_only,
         smooth=args.smooth, ymax=args.ymax)


if __name__ == "__main__":
    main()
