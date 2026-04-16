# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

EvoJAX is a hardware-accelerated neuroevolution toolkit built on JAX. The framework implements evolution algorithms, neural networks, and tasks end-to-end in JAX so the whole loop JIT-compiles and runs on GPUs/TPUs. Paper: https://arxiv.org/abs/2202.05008.

## Common commands

```bash
# Editable install with all optional extras (matches CI)
pip install -e .[extra]

# Run the full test suite (what CI runs)
pytest -W ignore::DeprecationWarning

# Run a single test class / test
pytest tests/test_algo.py::TestAlgo
pytest tests/test_algo.py::TestAlgo::test_cma_es_jax_save_and_load_state

# Lint (same two invocations CI uses — the first fails the build, the second is advisory)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=100 --max-line-length=127 --statistics

# Run an example (each example's header comment shows its canonical invocation)
python examples/train_mnist.py --gpu-id=0

# Algorithm benchmarks (see scripts/benchmarks/Readme.md)
python scripts/benchmarks/train.py --config=scripts/benchmarks/configs/<config>.yaml
```

Supported Python versions: 3.8, 3.9, 3.10 (CI matrix).

## Architecture

EvoJAX is organized around three pluggable abstractions plus two orchestrators. To add a new algorithm, policy, or task, implement the corresponding base class — the orchestrators are agnostic to which concrete implementation you wire in.

### Three pluggable interfaces

1. **`NEAlgorithm`** (`evojax/algo/base.py`) — ask/tell ES loop. Exposes `pop_size`, `ask() -> (pop_size, param_size)`, `tell(fitness)`, and `best_params`. `QualityDiversityMethod` extends this with `observe_bd(bd)` and `params_lattice`/`fitness_lattice`/`occupancy_lattice` (see `MAPElites`). Concrete algos live in `evojax/algo/` and are registered in `evojax/algo/__init__.py`'s `Strategies` dict.
2. **`PolicyNetwork`** (`evojax/policy/base.py`) — holds `num_params` and implements `get_actions(t_states, params, p_states)` where `params` has shape `(num_envs, num_params)`. Examples: `MLPPolicy`, `ConvNetPolicy`, `Seq2seqPolicy`, `PermutationInvariantPolicy` (`mlp_pi.py`).
3. **`VectorizedTask`** (`evojax/task/base.py`) — vectorized env with `reset(key) -> TaskState`, `step(state, action) -> (state, reward, done)`, plus `obs_shape`, `act_shape`, `max_steps`, and optional `multi_agent_training`. `TaskState` subclasses are typically `flax.struct.dataclass` PyTrees so they `jit`/`vmap` cleanly. For QD methods, `BDExtractor` augments a task state with behavior-descriptor fields.

### Orchestrators

- **`SimManager`** (`evojax/sim_mgr.py`) — runs rollouts. Jits/pmaps the policy-env inner loop, handles population × n_repeats expansion, multi-agent mode, observation normalization, and distributes across `jax.local_device_count()` devices. `get_task_reset_keys`, `split_params_for_pmap`, and `duplicate_params` are the static-argnum-compiled hot path — touching their signatures re-triggers compilation.
- **`Trainer`** (`evojax/trainer.py`) — owns the ask→eval→tell loop, periodic testing, `ObsNormalizer` plumbing, model checkpointing (`util.save_model`), and QD lattice persistence (`util.save_lattices`). When `solver` is a `QualityDiversityMethod`, the trainer calls `observe_bd(bds)` before `tell`.

Key invariant: the components are usable independently. ES-CLIP-style projects use `evojax.algo` alone with a custom training loop. Keep that separation — don't have algos reach into `SimManager` or `Trainer`.

### Observation normalization

`ObsNormalizer` (`evojax/obs_norm.py`) is always constructed, but acts as a no-op when `normalize_obs=False` (`dummy=True`). Its running stats (`obs_params`) are checkpointed alongside policy params so eval reproduces training normalization.

## Release workflow (from README_DEVLEOPMENT.md)

Versions follow PEP 440 in `evojax/version.py`. To cut a release: bump `__version__`, commit, `git tag vX.Y.Z`, `git push origin --tags`. Pushing a tag triggers the TestPyPI upload workflow; creating a GitHub Release for that tag triggers the PyPI upload workflow (see `.github/workflows/`). CI lints and tests on every push to any branch.
