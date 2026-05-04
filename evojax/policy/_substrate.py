# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Substrate geometry for HyperNEAT.

A substrate is a fixed-topology feedforward network whose weights are not
evolved directly — instead, an evolved CPPN is queried at the coordinate
pair (src_pos, dst_pos) for every connection to produce that connection's
weight. This module owns just the geometry (per-layer coordinate grids and
the precomputed src/dst coordinate-pair tensors); the actual CPPN-query +
substrate-forward is in ``hyperneat_policy.py``.
"""

from dataclasses import dataclass
from typing import List
from typing import Tuple

import numpy as np


@dataclass
class Substrate:
    """Per-layer geometry for the substrate network.

    Attributes:
        layer_sizes: number of nodes in each layer (input, hidden..., output).
        coords: list of length ``len(layer_sizes)``; ``coords[i]`` has shape
            ``(layer_sizes[i], coord_dim)`` giving each node's position in
            the substrate's coordinate space.
        pair_src: list of length ``len(layer_sizes) - 1``; ``pair_src[l]``
            has shape ``(layer_sizes[l] * layer_sizes[l+1], coord_dim)`` —
            the source coordinate for every (src, dst) edge in layer ``l``.
            Edges are enumerated in row-major (dst-major) order so that
            ``weights[l].reshape(layer_sizes[l+1], layer_sizes[l])`` lines
            up with a standard ``W @ x`` matmul.
        pair_dst: same shape as ``pair_src``, holding the dst coordinate.
        coord_dim: dimensionality of substrate coordinates (e.g. 2 for a
            2D plane). The CPPN's input dim is ``2 * coord_dim``.
    """

    layer_sizes: Tuple[int, ...]
    coords: List[np.ndarray]
    pair_src: List[np.ndarray]
    pair_dst: List[np.ndarray]
    coord_dim: int

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes)

    @property
    def cppn_input_dim(self) -> int:
        return 2 * self.coord_dim


def _line_coords(n: int) -> np.ndarray:
    """``n`` evenly spaced points on [-1, 1]. Single point goes to 0."""
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    return np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(-1, 1)


def _build_pairs(src_coords: np.ndarray,
                 dst_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """All (src, dst) coordinate pairs in dst-major order.

    Returns arrays of shape ``(n_dst * n_src, coord_dim)`` so the caller can
    reshape CPPN outputs into ``(n_dst, n_src)`` matrices that act as
    ``W`` in ``out = W @ in``.
    """
    n_src = src_coords.shape[0]
    n_dst = dst_coords.shape[0]
    src_rep = np.broadcast_to(src_coords[None, :, :],
                              (n_dst, n_src, src_coords.shape[1]))
    dst_rep = np.broadcast_to(dst_coords[:, None, :],
                              (n_dst, n_src, dst_coords.shape[1]))
    return (src_rep.reshape(-1, src_coords.shape[1]).copy(),
            dst_rep.reshape(-1, dst_coords.shape[1]).copy())


def make_grid_substrate(n_inputs: int,
                        hidden_layers: Tuple[int, ...],
                        n_outputs: int,
                        y_levels: Tuple[float, ...] = None,
                        ) -> Substrate:
    """Build a 2D substrate with one row of nodes per layer.

    Each layer's nodes are placed on a horizontal line at a distinct y-level
    (input row at y=-1, output row at y=+1, hidden rows evenly between),
    with x-coordinates evenly spaced on [-1, 1]. This is the standard
    "flat substrate" most HyperNEAT papers use for low-dim control tasks.

    Args:
        n_inputs: input layer size; should match the task's obs dim.
        hidden_layers: tuple of hidden-layer sizes (e.g. (8, 8)).
        n_outputs: output layer size; should match the task's act dim.
        y_levels: optional override for per-layer y. Length must equal
            ``1 + len(hidden_layers) + 1``. Defaults to evenly-spaced
            values from -1 to +1.

    Returns:
        ``Substrate`` ready to hand to ``HyperNEATPolicy``.
    """
    layer_sizes = (n_inputs,) + tuple(hidden_layers) + (n_outputs,)
    L = len(layer_sizes)
    if y_levels is None:
        if L == 1:
            ys: Tuple[float, ...] = (0.0,)
        else:
            ys = tuple(np.linspace(-1.0, 1.0, L, dtype=np.float32).tolist())
    else:
        if len(y_levels) != L:
            raise ValueError(
                f"y_levels length {len(y_levels)} != n_layers {L}")
        ys = tuple(y_levels)

    coords: List[np.ndarray] = []
    for size, y in zip(layer_sizes, ys):
        xs = _line_coords(size)  # (size, 1)
        layer = np.concatenate(
            [xs, np.full_like(xs, fill_value=y)], axis=1)  # (size, 2)
        coords.append(layer.astype(np.float32))

    pair_src: List[np.ndarray] = []
    pair_dst: List[np.ndarray] = []
    for l in range(L - 1):
        s, d = _build_pairs(coords[l], coords[l + 1])
        pair_src.append(s)
        pair_dst.append(d)

    return Substrate(
        layer_sizes=layer_sizes,
        coords=coords,
        pair_src=pair_src,
        pair_dst=pair_dst,
        coord_dim=2,
    )
