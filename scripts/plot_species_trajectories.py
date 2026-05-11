"""Phase 1 diagnostic plots for NEAT species protection research.

Reads a ``species_history.csv`` produced by NEAT's telemetry and emits
three panels:

  1. Per-species max-fitness trajectory (one colored line per species;
     retired species end at their retirement generation).
  2. Per-species improvement-rate trajectory (rolling Δmax_fit over the
     algorithm's ``improvement_window``).
  3. Histogram of improvement rate measured at retirement — answers the
     motivating question "how often does NEAT kill species that were
     still actively improving?".

A species is "still improving at retirement" if its improvement_rate at
its final logged generation is strictly positive.

Usage:
    python scripts/plot_species_trajectories.py \
        log/maze_hard_neat_baseline_seed0/species_history.csv \
        -o log/phase1_diagnostics.png
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "generation": int(r["generation"]),
                "species_id": int(r["species_id"]),
                "max_fit": float(r["max_fit"]) if r["max_fit"] != "nan" else math.nan,
                "improvement_rate": (float(r["improvement_rate"])
                                     if r["improvement_rate"] != "nan" else math.nan),
                "n_offspring": int(r["n_offspring"]),
                "retired": int(r["retired"]),
                "n_members": int(r["n_members"]),
            })
    return rows


def by_species(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["species_id"]].append(r)
    for sid in out:
        out[sid].sort(key=lambda x: x["generation"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="species_history.csv path")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--title-prefix", default="NEAT species diagnostics")
    args = parser.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    species = by_species(rows)
    print(f"{args.csv}: {len(rows)} rows, {len(species)} unique species")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel 1: per-species max fitness ---
    ax = axes[0]
    cmap = plt.get_cmap("tab20")
    for i, (sid, srows) in enumerate(sorted(species.items())):
        gens = [r["generation"] for r in srows if r["n_members"] > 0]
        fits = [r["max_fit"] for r in srows if r["n_members"] > 0]
        if not gens:
            continue
        color = cmap(i % 20)
        ax.plot(gens, fits, color=color, alpha=0.7, linewidth=1.2)
        # Mark retirement with an X
        retired_rows = [r for r in srows if r["retired"] == 1]
        for rr in retired_rows:
            ax.scatter(rr["generation"], rr["max_fit"], color=color,
                       marker="x", s=40, linewidths=2, zorder=5)
    ax.set_title("Per-species max fitness over generations\n(× = retired)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Max raw fitness")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: per-species improvement rate ---
    ax = axes[1]
    for i, (sid, srows) in enumerate(sorted(species.items())):
        gens = [r["generation"] for r in srows
                if r["n_members"] > 0 and not math.isnan(r["improvement_rate"])]
        rates = [r["improvement_rate"] for r in srows
                 if r["n_members"] > 0 and not math.isnan(r["improvement_rate"])]
        if not gens:
            continue
        color = cmap(i % 20)
        ax.plot(gens, rates, color=color, alpha=0.7, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Per-species improvement rate (Δmax_fit over window)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Improvement rate")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: improvement rate at retirement (histogram) ---
    ax = axes[2]
    retirement_rates = []
    for sid, srows in species.items():
        retired_rows = [r for r in srows if r["retired"] == 1]
        if not retired_rows:
            continue
        rate = retired_rows[-1]["improvement_rate"]
        if not math.isnan(rate):
            retirement_rates.append(rate)
    if retirement_rates:
        ax.hist(retirement_rates, bins=20, color="#d62728",
                edgecolor="black", alpha=0.85)
        n_active = sum(1 for r in retirement_rates if r > 0)
        ax.set_title(
            f"Improvement rate at retirement\n"
            f"{n_active}/{len(retirement_rates)} retired species "
            f"({100*n_active/len(retirement_rates):.0f}%) were still improving")
    else:
        ax.text(0.5, 0.5, "no retirements recorded",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Improvement rate at retirement")
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Improvement rate at retirement gen")
    ax.set_ylabel("# species")
    ax.grid(True, alpha=0.3)

    fig.suptitle(args.title_prefix + f" ({Path(args.csv).parent.name})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
