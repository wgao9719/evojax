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


class TestHyperNEAT:
    def test_substrate_geometry(self):
        from evojax.policy import make_grid_substrate

        sub = make_grid_substrate(n_inputs=4, hidden_layers=(8,), n_outputs=2)
        assert sub.layer_sizes == (4, 8, 2)
        assert sub.cppn_input_dim == 4
        # Two layer transitions: 4->8 and 8->2 → pair counts 32 and 16.
        assert sub.pair_src[0].shape == (32, 2)
        assert sub.pair_dst[0].shape == (32, 2)
        assert sub.pair_src[1].shape == (16, 2)

    def test_param_size_matches_neat_cppn(self):
        from evojax.algo import NEAT
        from evojax.policy import HyperNEATPolicy, make_grid_substrate

        sub = make_grid_substrate(4, (8, 8), 2)
        policy = HyperNEATPolicy(substrate=sub,
                                 max_hidden=6, max_connections=32)
        # CPPN evolved by NEAT: 4 coord inputs → 1 weight output.
        algo = NEAT(pop_size=8,
                    n_inputs=sub.cppn_input_dim, n_outputs=1,
                    max_hidden=6, max_connections=32,
                    activation_set=["tanh", "sin", "gauss", "abs"],
                    mutate_activation_rate=0.05,
                    seed=0)
        assert algo.param_size == policy.num_params

    def test_forward_runs_and_changes_with_genome(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evojax.algo import NEAT
        from evojax.policy import HyperNEATPolicy, make_grid_substrate
        from evojax.policy.base import PolicyState

        sub = make_grid_substrate(n_inputs=4, hidden_layers=(6,), n_outputs=2)
        policy = HyperNEATPolicy(substrate=sub,
                                 max_hidden=5, max_connections=24,
                                 weight_threshold=0.0)
        algo = NEAT(pop_size=4,
                    n_inputs=sub.cppn_input_dim, n_outputs=1,
                    max_hidden=5, max_connections=24,
                    activation_set=["tanh", "sin", "gauss"],
                    mutate_activation_rate=0.1, seed=1)
        params = algo.ask()
        assert params.shape == (4, policy.num_params)

        class _Obs:
            def __init__(self, obs):
                self.obs = obs

        obs = jnp.ones((4, 4), dtype=jnp.float32) * 0.3
        t = _Obs(obs)
        ps = PolicyState(keys=jax.random.split(jax.random.PRNGKey(0), 4))
        actions, _ = policy.get_actions(t, params, ps)
        assert actions.shape == (4, 2)
        assert jnp.all(jnp.isfinite(actions))
        # Output is tanh by default → bounded.
        assert jnp.all(actions >= -1.0) and jnp.all(actions <= 1.0)

        # Two genomes with genuinely different flat vectors should produce
        # different actions (sanity that the CPPN-paint path actually depends
        # on params, not just on obs).
        diff = float(jnp.abs(actions[0] - actions[1]).sum())
        assert diff > 0.0 or float(jnp.abs(
            jnp.asarray(params[0] - params[1])).sum()) == 0.0

    def test_activation_encoding_roundtrip(self):
        # The activations slot in the flat vector should hold valid IDs
        # drawn from activation_set, and changing them should influence
        # forward output (verified by forcing a hidden node active first).
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evojax.algo import NEAT
        from evojax.algo._neat_genome import CPPN_ACTIVATIONS
        from evojax.policy import HyperNEATPolicy, make_grid_substrate
        from evojax.policy.base import PolicyState

        sub = make_grid_substrate(2, (3,), 1)
        n_in_cppn = sub.cppn_input_dim  # 4
        n_out_cppn = 1
        max_hidden = 4
        max_conn = 16
        algo = NEAT(pop_size=4,
                    n_inputs=n_in_cppn, n_outputs=n_out_cppn,
                    max_hidden=max_hidden, max_connections=max_conn,
                    activation_set=["tanh", "sin"],
                    mutate_activation_rate=0.0, seed=3)
        params = algo.ask()
        N = n_in_cppn + n_out_cppn + max_hidden
        H = max_hidden
        C = max_conn
        act_start = 4 * C + N + H
        act_slice = np.asarray(params[:, act_start:act_start + N])
        sin_id = CPPN_ACTIVATIONS.index("sin")  # = 1
        tanh_id = CPPN_ACTIVATIONS.index("tanh")  # = 0
        # Inputs/outputs always 0 (tanh); hidden nodes from {0, 1}.
        assert np.all(act_slice[:, :n_in_cppn + n_out_cppn] == tanh_id)
        hidden_acts = act_slice[:, n_in_cppn + n_out_cppn:]
        assert set(np.unique(hidden_acts).tolist()).issubset(
            {float(tanh_id), float(sin_id)})

        # Force hidden node 0 active and add a connection through it so the
        # activation actually fires, then flip its ID and verify forward
        # changes. We mutate a deep copy of the flat vector.
        policy = HyperNEATPolicy(substrate=sub,
                                 max_hidden=max_hidden,
                                 max_connections=max_conn,
                                 weight_threshold=0.0)
        flat_a = np.asarray(params[0]).copy()
        # First connection: input 0 -> hidden 0 (= node index n_in+n_out).
        hidden0 = n_in_cppn + n_out_cppn
        # Second connection: hidden 0 -> output 0.
        out0 = n_in_cppn
        # Overwrite the first two connection slots.
        flat_a[0] = 1.5  # weight 1
        flat_a[C + 0] = 1.0  # enabled
        flat_a[2 * C + 0] = 0  # src = input 0
        flat_a[3 * C + 0] = hidden0  # dst = hidden 0
        flat_a[1] = 1.5  # weight 2
        flat_a[C + 1] = 1.0  # enabled
        flat_a[2 * C + 1] = hidden0  # src = hidden 0
        flat_a[3 * C + 1] = out0  # dst = output 0
        # Activate hidden node 0.
        flat_a[4 * C + N + 0] = 1.0  # hidden_active[0]
        # Set hidden 0's activation to tanh.
        flat_a[act_start + hidden0] = float(tanh_id)

        flat_b = flat_a.copy()
        flat_b[act_start + hidden0] = float(sin_id)

        batch = jnp.asarray(np.stack([flat_a, flat_b]))

        class _Obs:
            def __init__(self, obs):
                self.obs = obs

        obs = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        ps = PolicyState(keys=jax.random.split(jax.random.PRNGKey(0), 2))
        actions, _ = policy.get_actions(_Obs(obs), batch, ps)
        # tanh vs sin on the same pre-activation must produce different
        # downstream substrate weights -> different actions.
        assert float(jnp.abs(actions[0] - actions[1]).sum()) > 1e-6
