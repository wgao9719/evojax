"""Decode a saved NEAT or HyperNEAT best-genome flat vector and draw it.

The flat encoding lives in ``evojax.algo._neat_genome.neat_param_size`` and is
shared between the algorithm and the policy. This script reverses that
encoding into a graph (nodes by role, edges with weight + enabled flag) and
plots it with a layered layout: inputs on the left, outputs on the right,
hidden nodes spread between by a topological-order heuristic.

For HyperNEAT runs the flat vector is the *CPPN* genome (input dim = 4 for
2D substrate coordinates), and each hidden node carries an activation ID
which we display as a colored label.

Usage:
    python scripts/visualize_topology.py log/maze_neat_novelty/best.npz \
        --kind neat --n-inputs 12 --n-outputs 2 \
        --max-hidden 12 --max-connections 120 \
        -o log/topology_neat.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CPPN_ACT_NAMES = ["tanh", "sin", "gauss", "abs", "sigmoid", "identity", "relu"]
CPPN_ACT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2"]


def decode(flat, n_inputs, n_outputs, max_hidden, max_connections,
           include_activations):
    """Return (nodes_active, conns, biases, act_ids).

    nodes_active: bool array of length n_nodes (input/output always True).
    conns: list of (src, dst, weight, enabled).
    biases: float array of length n_nodes.
    act_ids: int array of length n_nodes (zeros for non-CPPN runs).
    """
    N = n_inputs + n_outputs + max_hidden
    C = max_connections
    H = max_hidden
    weight = flat[:C]
    enabled_raw = flat[C:2 * C]
    src = np.clip(flat[2 * C:3 * C], 0, N - 1).astype(int)
    dst = np.clip(flat[3 * C:4 * C], 0, N - 1).astype(int)
    bias = flat[4 * C:4 * C + N]
    hidden_active = flat[4 * C + N:4 * C + N + H] > 0.5

    nodes_active = np.zeros(N, dtype=bool)
    nodes_active[:n_inputs + n_outputs] = True
    nodes_active[n_inputs + n_outputs:] = hidden_active

    conns = []
    for i in range(C):
        en = enabled_raw[i] > 0.5
        # Skip dummy slots (zero-init: src=0,dst=0,enabled=0)
        if not en and weight[i] == 0.0 and src[i] == dst[i]:
            continue
        # Skip connections that touch deactivated hidden nodes
        if not nodes_active[src[i]] or not nodes_active[dst[i]]:
            continue
        conns.append((int(src[i]), int(dst[i]), float(weight[i]), bool(en)))

    if include_activations:
        act_ids = np.clip(
            flat[4 * C + N + H:4 * C + 2 * N + H], 0, len(CPPN_ACT_NAMES) - 1
        ).astype(int)
    else:
        act_ids = np.zeros(N, dtype=int)
    return nodes_active, conns, bias, act_ids


def layered_positions(n_inputs, n_outputs, max_hidden, nodes_active, conns):
    """Place nodes in a left-to-right layered layout.

    Inputs at x=0; outputs at x=1; hidden nodes get an x in (0, 1) by a
    cheap topological heuristic (one BFS pass from inputs through enabled
    edges, with cycles broken arbitrarily).
    """
    N = n_inputs + n_outputs + max_hidden
    pos = {}
    # Inputs evenly spaced top-to-bottom on the left
    for i in range(n_inputs):
        pos[i] = (0.0, 1.0 - (i / max(1, n_inputs - 1)))
    # Outputs on the right
    for i in range(n_outputs):
        pos[n_inputs + i] = (1.0, 1.0 - (i / max(1, n_outputs - 1)))

    # Compute hidden node depths via BFS from inputs (enabled edges only)
    hidden_start = n_inputs + n_outputs
    depth = {i: 0 for i in range(n_inputs)}
    enabled_out = {i: [] for i in range(N)}
    for s, d, _, en in conns:
        if en:
            enabled_out[s].append(d)
    frontier = list(range(n_inputs))
    visited = set(frontier)
    while frontier:
        nxt = []
        for u in frontier:
            for v in enabled_out[u]:
                if v in visited:
                    continue
                if v >= hidden_start and nodes_active[v]:
                    depth[v] = depth.get(u, 0) + 1
                    nxt.append(v)
                    visited.add(v)
        frontier = nxt

    active_hidden = [i for i in range(hidden_start, N) if nodes_active[i]]
    if active_hidden:
        max_d = max((depth.get(h, 1) for h in active_hidden), default=1)
        # Bucket by depth, lay each bucket out vertically
        buckets = {}
        for h in active_hidden:
            d = depth.get(h, max(1, max_d // 2))
            buckets.setdefault(d, []).append(h)
        for d, nodes in buckets.items():
            x = 0.15 + 0.7 * (d / max(1, max_d))
            for j, n in enumerate(sorted(nodes)):
                y = 1.0 - (j / max(1, len(nodes) - 1)) if len(nodes) > 1 else 0.5
                pos[n] = (x, y)
    return pos


def draw(pos, nodes_active, conns, biases, act_ids,
         n_inputs, n_outputs, title, out_path, include_activations):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Edges
    max_abs_w = max((abs(w) for _, _, w, en in conns if en), default=1.0)
    for s, d, w, en in conns:
        if s not in pos or d not in pos:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        color = "#1f77b4" if w > 0 else "#d62728"
        lw = 0.4 + 2.5 * (abs(w) / max_abs_w) if en else 0.5
        alpha = 0.75 if en else 0.18
        ls = "-" if en else "--"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                alpha=alpha, linestyle=ls, zorder=1)

    # Nodes
    n_nodes = len(biases)
    hidden_start = n_inputs + n_outputs
    for i in range(n_nodes):
        if not nodes_active[i] or i not in pos:
            continue
        x, y = pos[i]
        if i < n_inputs:
            color, marker, label = "#2ca02c", "s", f"in{i}"
        elif i < hidden_start:
            color, marker, label = "gold", "D", f"out{i - n_inputs}"
        else:
            if include_activations:
                color = CPPN_ACT_COLORS[act_ids[i] % len(CPPN_ACT_COLORS)]
                label = CPPN_ACT_NAMES[act_ids[i] % len(CPPN_ACT_NAMES)]
            else:
                color, label = "#7f7f7f", ""
            marker = "o"
        ax.scatter(x, y, s=380, c=color, marker=marker,
                   edgecolors="black", linewidths=1.2, zorder=3)
        if label:
            ax.text(x, y - 0.04, label, ha="center", va="top",
                    fontsize=8, zorder=4)

    # Counts
    n_h = int(nodes_active[hidden_start:].sum())
    n_e = sum(1 for _, _, _, en in conns if en)
    ax.set_title(f"{title}\n{n_h} active hidden nodes, "
                 f"{n_e} enabled connections")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")

    # Legend for activations on HyperNEAT
    if include_activations:
        used_acts = sorted({int(act_ids[i]) for i in range(hidden_start, n_nodes)
                            if nodes_active[i]})
        if used_acts:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=CPPN_ACT_COLORS[a],
                              markeredgecolor="black", markersize=10,
                              label=CPPN_ACT_NAMES[a])
                       for a in used_acts]
            ax.legend(handles=handles, loc="lower left", title="CPPN activations",
                      fontsize=9)
    else:
        # Legend for edge color
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="#1f77b4", lw=2, label="positive weight"),
            Line2D([0], [0], color="#d62728", lw=2, label="negative weight"),
            Line2D([0], [0], color="gray", lw=1.2, ls="--", label="disabled"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="best.npz path with 'params' field")
    parser.add_argument("--kind", choices=["neat", "hyperneat"], default="neat")
    parser.add_argument("--n-inputs", type=int, required=True)
    parser.add_argument("--n-outputs", type=int, required=True)
    parser.add_argument("--max-hidden", type=int, required=True)
    parser.add_argument("--max-connections", type=int, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    data = np.load(args.npz)
    flat = data["params"]
    include_activations = (args.kind == "hyperneat")

    # For HyperNEAT, the flat vector encodes the CPPN, NOT the substrate.
    # CPPN's n_inputs = 2 * substrate.coord_dim (4 for our 2D substrate),
    # n_outputs = 1. The substrate itself isn't stored in the genome.
    if include_activations:
        cppn_n_in, cppn_n_out = 4, 1
        nodes_active, conns, biases, act_ids = decode(
            flat, cppn_n_in, cppn_n_out, args.max_hidden,
            args.max_connections, include_activations=True)
        pos = layered_positions(cppn_n_in, cppn_n_out, args.max_hidden,
                                nodes_active, conns)
        title = args.title or f"HyperNEAT CPPN ({Path(args.npz).parent.name})"
        draw(pos, nodes_active, conns, biases, act_ids,
             cppn_n_in, cppn_n_out, title, args.output, True)
    else:
        nodes_active, conns, biases, act_ids = decode(
            flat, args.n_inputs, args.n_outputs, args.max_hidden,
            args.max_connections, include_activations=False)
        pos = layered_positions(args.n_inputs, args.n_outputs, args.max_hidden,
                                nodes_active, conns)
        title = args.title or f"NEAT topology ({Path(args.npz).parent.name})"
        draw(pos, nodes_active, conns, biases, act_ids,
             args.n_inputs, args.n_outputs, title, args.output, False)


if __name__ == "__main__":
    main()
