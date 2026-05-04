# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Plot fitness curves from an evojax training log.

Parses ``Iter=... max=... avg=... min=... std=...`` lines (train) and
``[TEST] Iter=... max=... avg=... min=... std=...`` lines (test). Writes a
PNG with three panels: mean +/- std band, max, min.

Usage:
    python scripts/plot_neat_log.py log/slimevolley_neat/SlimeVolleyNEAT.txt
    python scripts/plot_neat_log.py log/.../SlimeVolleyNEAT.txt -o curves.png
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
    """Return the metrics from the *last* training run in the log file."""
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


def plot(train, test, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    def _panel(ax, field, label):
        if train["iter"]:
            ax.plot(train["iter"], train[field], label="train " + label,
                    color="C0", linewidth=1.5)
        if test["iter"]:
            ax.plot(test["iter"], test[field], label="test " + label,
                    color="C1", marker="o", linewidth=1.5, markersize=5)

    # Panel 1: avg ± std band
    ax = axes[0]
    if train["iter"]:
        it = np.array(train["iter"])
        avg = np.array(train["avg"])
        std = np.array(train["std"])
        ax.plot(it, avg, color="C0", label="train avg")
        ax.fill_between(it, avg - std, avg + std, color="C0", alpha=0.2,
                        label="train ±std")
    if test["iter"]:
        ax.plot(test["iter"], test["avg"], color="C1", marker="o",
                label="test avg", linewidth=1.5, markersize=5)
    ax.set_title("Average fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: max
    _panel(axes[1], "max", "max")
    axes[1].set_title("Best fitness")
    axes[1].set_xlabel("Iteration")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: min
    _panel(axes[2], "min", "min")
    axes[2].set_title("Worst fitness")
    axes[2].set_xlabel("Iteration")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str,
                        help="Path to *.txt log produced by evojax Trainer.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PNG path. Defaults to <log>.png")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title. Defaults to the log filename.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    out_path = Path(args.output) if args.output else log_path.with_suffix(".png")
    title = args.title or log_path.stem

    train, test = parse_log(log_path)
    print(f"Parsed {len(train['iter'])} train entries, "
          f"{len(test['iter'])} test entries from last run.")
    if not train["iter"] and not test["iter"]:
        raise SystemExit("No parseable Iter=... lines found in log.")
    plot(train, test, title, out_path)


if __name__ == "__main__":
    main()
