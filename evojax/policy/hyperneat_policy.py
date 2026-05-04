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

"""HyperNEAT policy: indirect encoding via a CPPN over a fixed substrate.

The evolved object is a CPPN — a NEAT-evolved network with diverse
activation functions (sin, gauss, abs, ...) that maps coordinate tuples
``(x_src, y_src, x_dst, y_dst)`` to a connection weight. At forward-pass
time we query the CPPN for every (src, dst) pair in each substrate layer
to materialize that layer's weight matrix, then run the substrate (a
fixed-topology MLP) on the observation.

Pair with :class:`evojax.algo.NEAT` configured with
``activation_set=[...]`` and ``mutate_activation_rate>0`` — the param
sizes will line up automatically.
"""

import logging
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from evojax.algo._neat_genome import CPPN_ACTIVATIONS
from evojax.algo._neat_genome import neat_param_size
from evojax.policy._substrate import Substrate
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


# CPPN activation functions, indexed identically to ``CPPN_ACTIVATIONS``.
# Each takes a single jnp array and returns the same shape.
def _act_tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x)


def _act_sin(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x)


def _act_gauss(x: jnp.ndarray) -> jnp.ndarray:
    # Bell curve centered at 0; classic CPPN building block for symmetry.
    return jnp.exp(-jnp.square(x))


def _act_abs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(x)


def _act_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)


def _act_identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


def _act_relu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(x)


_CPPN_ACT_FNS = (
    _act_tanh, _act_sin, _act_gauss, _act_abs,
    _act_sigmoid, _act_identity, _act_relu,
)
assert len(_CPPN_ACT_FNS) == len(CPPN_ACTIVATIONS), (
    "CPPN activation function table is out of sync with CPPN_ACTIVATIONS")


def _apply_per_node_activation(pre: jnp.ndarray,
                               act_ids: jnp.ndarray) -> jnp.ndarray:
    """Apply per-node activation: out[i] = ACT_FNS[act_ids[i]](pre[i]).

    Computes every activation function on every node and gathers the
    matching one — vectorized and JIT-friendly. Costs O(F*N) per call
    where F is the number of CPPN activations (small constant).
    """
    stacked = jnp.stack([fn(pre) for fn in _CPPN_ACT_FNS], axis=0)  # (F, N)
    n = pre.shape[0]
    return stacked[act_ids, jnp.arange(n)]


class HyperNEATPolicy(PolicyNetwork):
    """Indirect-encoding policy: a CPPN paints weights onto a fixed substrate.

    Args:
        substrate: substrate geometry (see ``_substrate.Substrate``). The
            CPPN's input dim is set from ``substrate.cppn_input_dim``.
        max_hidden: max hidden nodes in the CPPN genome (NEAT cap).
        max_connections: max connections in the CPPN genome (NEAT cap).
        n_forward_iters: how many propagation iterations to run on the CPPN
            for each (src, dst) query. CPPNs are conceptually feedforward;
            3-5 iters lets activations settle through evolved hidden layers.
        substrate_act_fn: nonlinearity applied between substrate layers.
        output_act_fn: final-layer nonlinearity ("tanh", "softmax", "linear").
        weight_scale: multiplier on CPPN-emitted weights before substrate
            matmul. Helps when the CPPN output is bounded (e.g. tanh) but
            substrate needs larger effective weights.
        weight_threshold: zero out queried weights with abs value below this
            (the standard HyperNEAT "expression threshold"; 0 disables).
    """

    def __init__(self,
                 substrate: Substrate,
                 max_hidden: int = 8,
                 max_connections: int = 64,
                 n_forward_iters: int = 4,
                 substrate_act_fn: str = "tanh",
                 output_act_fn: str = "tanh",
                 weight_scale: float = 3.0,
                 weight_threshold: float = 0.2,
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = create_logger(name="HyperNEATPolicy")
        else:
            self._logger = logger

        self.substrate = substrate
        self.cppn_n_inputs = substrate.cppn_input_dim
        self.cppn_n_outputs = 1
        self.max_hidden = int(max_hidden)
        self.max_connections = int(max_connections)
        self.n_forward_iters = int(n_forward_iters)
        self.substrate_act_fn = substrate_act_fn
        self.output_act_fn = output_act_fn
        self.weight_scale = float(weight_scale)
        self.weight_threshold = float(weight_threshold)

        # The CPPN genome carries activation IDs, so include_activations=True.
        self.num_params = neat_param_size(
            self.cppn_n_inputs, self.cppn_n_outputs,
            self.max_hidden, self.max_connections,
            include_activations=True)

        # Stack each layer's (src, dst) coordinate pairs into a single
        # (K_l, cppn_in) array; held as constants in the JIT graph.
        self._pair_inputs: List[jnp.ndarray] = []
        for s, d in zip(substrate.pair_src, substrate.pair_dst):
            pairs = np.concatenate([s, d], axis=-1).astype(np.float32)
            self._pair_inputs.append(jnp.asarray(pairs))
        self._layer_shapes: List[Tuple[int, int]] = [
            (substrate.layer_sizes[l + 1], substrate.layer_sizes[l])
            for l in range(substrate.n_layers - 1)
        ]

        single_fn = partial(
            _hyperneat_forward_single,
            cppn_n_inputs=self.cppn_n_inputs,
            cppn_n_outputs=self.cppn_n_outputs,
            max_hidden=self.max_hidden,
            max_connections=self.max_connections,
            n_iters=self.n_forward_iters,
            substrate_act_fn=self.substrate_act_fn,
            output_act_fn=self.output_act_fn,
            weight_scale=self.weight_scale,
            weight_threshold=self.weight_threshold,
            pair_inputs=tuple(self._pair_inputs),
            layer_shapes=tuple(self._layer_shapes),
        )
        self._forward_fn = jax.vmap(single_fn, in_axes=(0, 0))

        self._logger.info(
            "HyperNEATPolicy: substrate=%s, cppn_in=%d, "
            "max_hidden=%d, max_connections=%d, num_params=%d",
            substrate.layer_sizes, self.cppn_n_inputs,
            self.max_hidden, self.max_connections, self.num_params)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState,
                    ) -> Tuple[jnp.ndarray, PolicyState]:
        actions = self._forward_fn(params, t_states.obs)
        return actions, p_states


def _decode_cppn(flat: jnp.ndarray,
                 cppn_n_inputs: int,
                 cppn_n_outputs: int,
                 max_hidden: int,
                 max_connections: int,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                            jnp.ndarray, jnp.ndarray, jnp.ndarray,
                            jnp.ndarray]:
    """Slice a flat CPPN genome into its component arrays.

    Returns: (W, bias, node_mask, act_ids, n_inputs, n_outputs, n_nodes).
    Layout matches ``NEAT._encode_genome`` with include_activations=True.
    """
    N = cppn_n_inputs + cppn_n_outputs + max_hidden
    C = max_connections
    H = max_hidden

    weight = flat[:C]
    enabled = (flat[C:2 * C] > 0.5).astype(jnp.float32)
    src = jnp.clip(flat[2 * C:3 * C], 0, N - 1).astype(jnp.int32)
    dst = jnp.clip(flat[3 * C:4 * C], 0, N - 1).astype(jnp.int32)
    bias = flat[4 * C:4 * C + N]
    hidden_active = (flat[4 * C + N:4 * C + N + H] > 0.5).astype(jnp.float32)
    act_ids = jnp.clip(flat[4 * C + N + H:4 * C + 2 * N + H],
                       0, len(CPPN_ACTIVATIONS) - 1).astype(jnp.int32)

    io_mask = jnp.ones(cppn_n_inputs + cppn_n_outputs, dtype=jnp.float32)
    node_mask = jnp.concatenate([io_mask, hidden_active], axis=0)

    W = jnp.zeros((N, N), dtype=jnp.float32)
    W = W.at[dst, src].add(weight * enabled)
    return W, bias, node_mask, act_ids, cppn_n_inputs, cppn_n_outputs, N


def _cppn_eval_single(W: jnp.ndarray,
                      bias: jnp.ndarray,
                      node_mask: jnp.ndarray,
                      act_ids: jnp.ndarray,
                      cppn_input: jnp.ndarray,
                      n_inputs: int,
                      n_outputs: int,
                      n_iters: int) -> jnp.ndarray:
    """Run the CPPN on a single coordinate-pair input. Returns output vector.

    Uses the same iterative-propagation scheme as ``NEATPolicy`` but with
    per-node activations and no recurrent state carry — each query starts
    from a zero internal state. Inputs are clamped each iter so recurrent
    edges into input slots can't overwrite the query coordinates.
    """
    N = bias.shape[0]
    a = jnp.zeros(N, dtype=jnp.float32)
    a = a.at[:n_inputs].set(cppn_input)
    for _ in range(n_iters):
        pre = W @ a + bias
        a_new = _apply_per_node_activation(pre, act_ids) * node_mask
        a = a_new.at[:n_inputs].set(cppn_input)
    return a[n_inputs:n_inputs + n_outputs]


def _hyperneat_forward_single(flat: jnp.ndarray,
                              obs: jnp.ndarray,
                              cppn_n_inputs: int,
                              cppn_n_outputs: int,
                              max_hidden: int,
                              max_connections: int,
                              n_iters: int,
                              substrate_act_fn: str,
                              output_act_fn: str,
                              weight_scale: float,
                              weight_threshold: float,
                              pair_inputs: Tuple[jnp.ndarray, ...],
                              layer_shapes: Tuple[Tuple[int, int], ...],
                              ) -> jnp.ndarray:
    """End-to-end per-genome forward pass.

    1. Decode the CPPN genome from the flat vector.
    2. For each substrate layer, vmap the CPPN over that layer's coordinate
       pairs to produce a (n_dst, n_src) weight matrix.
    3. Run the substrate MLP on the observation.
    """
    W, bias, node_mask, act_ids, n_in, n_out, _ = _decode_cppn(
        flat, cppn_n_inputs, cppn_n_outputs, max_hidden, max_connections)

    cppn_query = partial(
        _cppn_eval_single,
        W, bias, node_mask, act_ids,
        n_inputs=n_in, n_outputs=n_out, n_iters=n_iters,
    )
    batched_cppn = jax.vmap(cppn_query)  # vmap over coord-pair dim

    # Build substrate weight matrices by querying the CPPN.
    substrate_weights = []
    for pairs, (n_dst, n_src) in zip(pair_inputs, layer_shapes):
        out = batched_cppn(pairs)  # (K, 1)
        w = out[:, 0].reshape(n_dst, n_src) * weight_scale
        if weight_threshold > 0.0:
            w = jnp.where(jnp.abs(w) < weight_threshold,
                          jnp.zeros_like(w), w)
        substrate_weights.append(w)

    # Substrate forward.
    a = obs
    n_layers = len(substrate_weights)
    for i, w in enumerate(substrate_weights):
        z = w @ a
        if i == n_layers - 1:
            if output_act_fn == "tanh":
                a = jnp.tanh(z)
            elif output_act_fn == "sigmoid":
                a = jax.nn.sigmoid(z)
            elif output_act_fn == "softmax":
                a = jax.nn.softmax(z, axis=-1)
            elif output_act_fn == "linear":
                a = z
            else:
                raise ValueError(
                    f"Unsupported output_act_fn: {output_act_fn}")
        else:
            if substrate_act_fn == "tanh":
                a = jnp.tanh(z)
            elif substrate_act_fn == "relu":
                a = jax.nn.relu(z)
            elif substrate_act_fn == "sigmoid":
                a = jax.nn.sigmoid(z)
            else:
                raise ValueError(
                    f"Unsupported substrate_act_fn: {substrate_act_fn}")
    return a
