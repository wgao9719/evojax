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


class TestNEAT:
    def test_param_size_matches_policy(self):
        from evojax.algo import NEAT
        from evojax.policy import NEATPolicy

        algo = NEAT(pop_size=8, n_inputs=4, n_outputs=2,
                    max_hidden=6, max_connections=32, seed=0)
        policy = NEATPolicy(n_inputs=4, n_outputs=2,
                            max_hidden=6, max_connections=32)
        assert algo.param_size == policy.num_params

    def test_ask_shape(self):
        from evojax.algo import NEAT

        algo = NEAT(pop_size=12, n_inputs=3, n_outputs=2,
                    max_hidden=4, max_connections=16, seed=0)
        params = algo.ask()
        assert params.shape == (12, algo.param_size)
        assert params.dtype.name == "float32"

    def test_tell_advances_population(self):
        import numpy as np
        from evojax.algo import NEAT

        algo = NEAT(pop_size=16, n_inputs=4, n_outputs=2,
                    max_hidden=4, max_connections=24, seed=7)
        before = algo.ask()
        # Random fitness so selection + mutation actually shuffle the pop.
        rng = np.random.default_rng(0)
        fitness = rng.normal(size=algo.pop_size).astype("float32")
        algo.tell(fitness)
        after = algo.ask()
        # Something must have changed (high probability).
        assert not (before == after).all()

    def test_best_params_shape_and_setter(self):
        import numpy as np
        from evojax.algo import NEAT

        algo = NEAT(pop_size=8, n_inputs=2, n_outputs=1,
                    max_hidden=3, max_connections=12, seed=1)
        _ = algo.ask()
        algo.tell(np.arange(8, dtype="float32"))
        best = algo.best_params
        assert best.shape == (algo.param_size,)

        # Inject a synthetic flat vector and read it back.
        injected = np.zeros(algo.param_size, dtype="float32")
        injected[0] = 1.5
        algo.best_params = injected
        assert float(algo.best_params[0]) == 1.5

    def test_save_load_state(self):
        import numpy as np
        from evojax.algo import NEAT

        algo = NEAT(pop_size=8, n_inputs=3, n_outputs=2,
                    max_hidden=4, max_connections=16, seed=42)
        for _ in range(3):
            _ = algo.ask()
            algo.tell(np.random.default_rng(0).normal(size=8).astype("float32"))

        state = algo.save_state()
        flat_before = algo.best_params

        # Run more generations to perturb internal state.
        for _ in range(2):
            _ = algo.ask()
            algo.tell(np.random.default_rng(1).normal(size=8).astype("float32"))

        algo.load_state(state)
        flat_after = algo.best_params
        assert (np.asarray(flat_before) == np.asarray(flat_after)).all()

    def test_policy_forward_runs(self):
        import jax
        import jax.numpy as jnp
        from evojax.algo import NEAT
        from evojax.policy import NEATPolicy
        from evojax.policy.base import PolicyState

        algo = NEAT(pop_size=4, n_inputs=4, n_outputs=2,
                    max_hidden=3, max_connections=16, seed=0)
        policy = NEATPolicy(n_inputs=4, n_outputs=2,
                            max_hidden=3, max_connections=16,
                            n_forward_iters=3)
        params = algo.ask()

        class _FakeTaskState:
            def __init__(self, obs):
                self.obs = obs

        obs = jnp.ones((4, 4), dtype=jnp.float32) * 0.1
        t_states = _FakeTaskState(obs)
        p_states = PolicyState(keys=jax.random.split(jax.random.PRNGKey(0), 4))

        actions, _ = policy.get_actions(t_states, params, p_states)
        assert actions.shape == (4, 2)
        # tanh output, finite
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(actions >= -1.0) and jnp.all(actions <= 1.0)
