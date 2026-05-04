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

"""NEATPolicy: JAX forward pass over the flat genome encoding.

Pairs with :class:`evojax.algo.NEAT`. The flat parameter vector encodes a
graph of at most ``max_hidden`` hidden nodes and ``max_connections`` edges;
this policy builds a dense ``(N, N)`` weight matrix from the enabled
connections and propagates activations for ``n_forward_iters`` steps within
a single env step, then carries the final node activations across env steps
as persistent hidden state so recurrent edges can encode temporal memory.
"""

import logging
from functools import partial
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from evojax.algo._neat_genome import neat_param_size
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


@dataclass
class NEATPolicyState(PolicyState):
    """NEAT policy state.

    ``hidden`` holds per-env node activations (shape ``(batch, n_nodes)``)
    carried from one env step to the next. This is what makes evolved
    recurrent edges functional — without it, a feedforward-only pass can't
    integrate information across time.
    """

    hidden: jnp.ndarray


class NEATPolicy(PolicyNetwork):
    """Decodes a NEAT flat genome and runs K iterations of forward propagation.

    ``num_params`` matches :func:`evojax.algo._neat_genome.neat_param_size`
    exactly so this policy plugs into :class:`evojax.algo.NEAT` without drift.
    """

    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 max_hidden: int = 16,
                 max_connections: int = 128,
                 n_forward_iters: int = 3,
                 output_act_fn: str = "tanh",
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = create_logger(name="NEATPolicy")
        else:
            self._logger = logger

        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.max_hidden = int(max_hidden)
        self.max_connections = int(max_connections)
        self.n_forward_iters = int(n_forward_iters)
        self.output_act_fn = output_act_fn

        self.n_nodes = self.n_inputs + self.n_outputs + self.max_hidden
        self.num_params = neat_param_size(self.n_inputs, self.n_outputs,
                                          self.max_hidden,
                                          self.max_connections)
        self._logger.info(
            "NEATPolicy.num_params = %d (n_inputs=%d, n_outputs=%d, "
            "max_hidden=%d, max_connections=%d)",
            self.num_params, self.n_inputs, self.n_outputs, self.max_hidden,
            self.max_connections)

        single_fn = partial(
            _neat_forward_single,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            max_hidden=self.max_hidden,
            max_connections=self.max_connections,
            n_iters=self.n_forward_iters,
            output_act_fn=self.output_act_fn,
        )
        self._forward_fn = jax.vmap(single_fn, in_axes=(0, 0, 0))

    def reset(self, states: TaskState) -> NEATPolicyState:
        batch = states.obs.shape[0]
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        hidden = jnp.zeros((batch, self.n_nodes), dtype=jnp.float32)
        return NEATPolicyState(keys=keys, hidden=hidden)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: NEATPolicyState,
                    ) -> Tuple[jnp.ndarray, NEATPolicyState]:
        actions, new_hidden = self._forward_fn(
            params, t_states.obs, p_states.hidden)
        return actions, p_states.replace(hidden=new_hidden)


def _neat_forward_single(flat: jnp.ndarray,
                         obs: jnp.ndarray,
                         prev_hidden: jnp.ndarray,
                         n_inputs: int,
                         n_outputs: int,
                         max_hidden: int,
                         max_connections: int,
                         n_iters: int,
                         output_act_fn: str,
                         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass for a single genome's flat vector on a single observation.

    ``jax.vmap`` lifts this over (pop_size * n_repeats, ...). Returns both the
    action and the final node activations so the caller can carry them into
    the next env step — this is what makes evolved recurrent edges
    functional as temporal memory rather than just within-step settling.
    """
    N = n_inputs + n_outputs + max_hidden
    C = max_connections

    weight = flat[:C]
    enabled = (flat[C:2 * C] > 0.5).astype(jnp.float32)
    src = jnp.clip(flat[2 * C:3 * C], 0, N - 1).astype(jnp.int32)
    dst = jnp.clip(flat[3 * C:4 * C], 0, N - 1).astype(jnp.int32)
    bias = flat[4 * C:4 * C + N]
    hidden_active = (flat[4 * C + N:] > 0.5).astype(jnp.float32)

    # Node activation mask: inputs and outputs are always live; hidden nodes
    # gated by the flat-vector ``hidden_active`` flag.
    io_mask = jnp.ones(n_inputs + n_outputs, dtype=jnp.float32)
    node_mask = jnp.concatenate([io_mask, hidden_active], axis=0)

    # Scatter-add weights into W[dst, src] so a = W @ activations.
    W = jnp.zeros((N, N), dtype=jnp.float32)
    W = W.at[dst, src].add(weight * enabled)

    # Start from previous hidden state (zeroed on reset). Clamp inputs to the
    # current obs each inner iter so recurrent edges feeding back into input
    # indices don't overwrite the observation.
    a = prev_hidden
    a = a.at[:n_inputs].set(obs)
    for _ in range(n_iters):
        a_new = jnp.tanh(W @ a + bias) * node_mask
        a = a_new.at[:n_inputs].set(obs)

    out = a[n_inputs:n_inputs + n_outputs]
    if output_act_fn == "tanh":
        # Hidden layers already used tanh; re-apply so a NEAT genome with no
        # hidden layer still emits a bounded action.
        action = jnp.tanh(out)
    elif output_act_fn == "softmax":
        action = jax.nn.softmax(out, axis=-1)
    elif output_act_fn == "linear":
        action = out
    else:
        raise ValueError(f"Unsupported output_act_fn: {output_act_fn}")

    return action, a
