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

"""NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

Wraps Python/NumPy genome state around the evojax ``NEAlgorithm`` contract.
Pair with :class:`evojax.policy.NEATPolicy`, which decodes the flat-vector
genome and runs a JAX forward pass.

Since genomes have variable effective topology, the flat vector is sized for
a fixed maximum (``max_hidden``, ``max_connections``) and carries an enable
mask per connection. Exceeding the max causes structural mutations to no-op.
"""

import copy
import logging
import pickle
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from evojax.algo.base import NEAlgorithm
from evojax.algo._neat_genome import (
    Genome,
    InnovationRegistry,
    Species,
    compatibility_distance,
    crossover,
    fitness_share,
    make_minimal_genome,
    mutate_add_connection,
    mutate_add_node,
    mutate_toggle_enable,
    mutate_weights,
    neat_param_size,
    speciate,
)
from evojax.util import create_logger


class NEAT(NEAlgorithm):
    """NEAT algorithm producing fixed-size flat parameter vectors.

    The flat encoding (shared with :class:`NEATPolicy`) is
    ``[weight, enabled, src, dst, bias, hidden_active]`` with
    ``param_size = 4 * max_connections + (n_inputs + n_outputs + max_hidden)
    + max_hidden``.
    """

    def __init__(self,
                 pop_size: int,
                 n_inputs: int,
                 n_outputs: int,
                 max_hidden: int = 16,
                 max_connections: int = 128,
                 weight_mutation_rate: float = 0.8,
                 weight_perturb_sigma: float = 0.1,
                 weight_replace_rate: float = 0.1,
                 add_connection_rate: float = 0.05,
                 add_node_rate: float = 0.03,
                 toggle_enable_rate: float = 0.01,
                 crossover_rate: float = 0.75,
                 disjoint_coef: float = 1.0,
                 excess_coef: float = 1.0,
                 weight_diff_coef: float = 0.4,
                 compat_threshold: float = 3.0,
                 species_elitism: int = 1,
                 stagnation_patience: int = 15,
                 parsimony_weight: float = 0.001,
                 tournament_size: int = 3,
                 init_weight_scale: float = 1.0,
                 seed: int = 0,
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self.logger = create_logger(name="NEAT")
        else:
            self.logger = logger

        self.pop_size = int(pop_size)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.max_hidden = int(max_hidden)
        self.max_connections = int(max_connections)

        self.weight_mutation_rate = weight_mutation_rate
        self.weight_perturb_sigma = weight_perturb_sigma
        self.weight_replace_rate = weight_replace_rate
        self.add_connection_rate = add_connection_rate
        self.add_node_rate = add_node_rate
        self.toggle_enable_rate = toggle_enable_rate
        self.crossover_rate = crossover_rate
        self.disjoint_coef = disjoint_coef
        self.excess_coef = excess_coef
        self.weight_diff_coef = weight_diff_coef
        self.compat_threshold = compat_threshold
        self.species_elitism = int(species_elitism)
        self.stagnation_patience = int(stagnation_patience)
        self.parsimony_weight = parsimony_weight
        self.tournament_size = int(tournament_size)
        self.init_weight_scale = init_weight_scale

        self.param_size = neat_param_size(self.n_inputs, self.n_outputs,
                                          self.max_hidden,
                                          self.max_connections)

        self._rng = np.random.default_rng(seed)
        self._innovation = InnovationRegistry()
        self._species: List[Species] = []
        self._generation = 0
        self._best_fitness: float = -np.inf
        self._best_genome: Optional[Genome] = None
        self._best_flat: np.ndarray = np.zeros(self.param_size, dtype=np.float32)

        self._population: List[Genome] = [
            make_minimal_genome(self.n_inputs, self.n_outputs,
                                self.max_hidden, self._innovation, self._rng,
                                self.init_weight_scale)
            for _ in range(self.pop_size)
        ]
        # Pre-alloc the flat encoding buffer; re-used every ``ask``.
        self._flat_buf = np.zeros((self.pop_size, self.param_size),
                                  dtype=np.float32)

        self.logger.info(
            "NEAT: pop_size=%d, n_inputs=%d, n_outputs=%d, max_hidden=%d, "
            "max_connections=%d, param_size=%d",
            self.pop_size, self.n_inputs, self.n_outputs, self.max_hidden,
            self.max_connections, self.param_size)

    # ------------------------------------------------------------------ encode

    def _encode_genome(self, g: Genome, out: np.ndarray) -> None:
        """Write genome ``g`` into the pre-allocated slice ``out``.

        Slot boundaries:
            [0,   C)        weight
            [C,   2C)       enabled
            [2C,  3C)       src
            [3C,  4C)       dst
            [4C,  4C + N)   bias
            [4C+N, end)     hidden_active
        """
        out.fill(0.0)
        C = self.max_connections
        N = self.n_inputs + self.n_outputs + self.max_hidden
        # Truncate if a genome over-generated (shouldn't happen, safety)
        conns = g.connections[:C]
        for i, c in enumerate(conns):
            out[i] = c.weight
            out[C + i] = 1.0 if c.enabled else 0.0
            out[2 * C + i] = float(c.src)
            out[3 * C + i] = float(c.dst)
        out[4 * C:4 * C + N] = g.biases
        hidden_start = self.n_inputs + self.n_outputs
        for h in g.hidden_nodes:
            out[4 * C + N + (h - hidden_start)] = 1.0

    # --------------------------------------------------------------- interface

    def ask(self) -> jnp.ndarray:
        for i, g in enumerate(self._population):
            self._encode_genome(g, self._flat_buf[i])
        # Copy so the returned jax array doesn't alias the re-used numpy
        # buffer we'll overwrite on the next ask().
        return jnp.asarray(self._flat_buf.copy())

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        raw_fit = np.asarray(fitness, dtype=np.float64)
        if raw_fit.shape != (self.pop_size,):
            raise ValueError(
                f"NEAT.tell expected fitness of shape ({self.pop_size},), "
                f"got {raw_fit.shape}")

        # Parsimony penalty on adjusted (maximization-oriented) fitness.
        n_conns = np.array(
            [sum(1 for c in g.connections if c.enabled)
             for g in self._population],
            dtype=np.float64)
        adj_fit = raw_fit - self.parsimony_weight * n_conns

        # Track best by RAW fitness (parsimony only shapes selection).
        best_idx = int(np.argmax(raw_fit))
        if raw_fit[best_idx] > self._best_fitness:
            self._best_fitness = float(raw_fit[best_idx])
            self._best_genome = self._population[best_idx].clone()
            self._encode_genome(self._best_genome, self._flat_buf[best_idx])
            self._best_flat = self._flat_buf[best_idx].copy()

        # --- Speciation ---
        self._species = speciate(self._population, self._species,
                                 self.compat_threshold,
                                 self.disjoint_coef, self.excess_coef,
                                 self.weight_diff_coef, self._rng)

        # Update species stagnation counters using raw max fitness per species.
        for sp in self._species:
            sp_best = max(raw_fit[m] for m in sp.members)
            if sp_best > sp.best_fitness:
                sp.best_fitness = float(sp_best)
                sp.stagnation = 0
            else:
                sp.stagnation += 1

        # Retire stagnant species (keep the global best-performing one).
        if len(self._species) > 1:
            # Species with the highest best_fitness is protected.
            protect_idx = int(np.argmax([s.best_fitness
                                         for s in self._species]))
            kept: List[Species] = []
            for i, sp in enumerate(self._species):
                if (sp.stagnation < self.stagnation_patience or
                        i == protect_idx):
                    kept.append(sp)
            if kept:
                self._species = kept

        # --- Fitness sharing (shift to positive first) ---
        fit_min = float(np.min(adj_fit))
        shifted = adj_fit - fit_min + 1e-6
        shared = fitness_share(shifted, self._species)

        # --- Offspring budget per species ---
        species_totals = np.array(
            [sum(shared[m] for m in sp.members) for sp in self._species],
            dtype=np.float64)
        total = float(species_totals.sum())
        if total <= 0.0 or len(self._species) == 0:
            budgets = np.full(max(1, len(self._species)),
                              self.pop_size // max(1, len(self._species)))
        else:
            budgets = np.floor(species_totals / total * self.pop_size).astype(
                int)
        # Distribute rounding remainder to the top species
        remainder = self.pop_size - int(budgets.sum())
        if remainder > 0 and len(budgets) > 0:
            order = np.argsort(-species_totals)
            for i in range(remainder):
                budgets[order[i % len(order)]] += 1
        elif remainder < 0:
            order = np.argsort(species_totals)
            deficit = -remainder
            for i in range(deficit):
                if budgets[order[i % len(order)]] > 0:
                    budgets[order[i % len(order)]] -= 1

        # --- Breed next generation ---
        next_pop: List[Genome] = []
        for sp, budget in zip(self._species, budgets):
            if budget <= 0 or not sp.members:
                continue
            # Sort species members by raw fitness, descending
            members_sorted = sorted(sp.members,
                                    key=lambda idx: raw_fit[idx],
                                    reverse=True)
            # Elitism: copy top-K verbatim
            n_elite = min(self.species_elitism, budget, len(members_sorted))
            for k in range(n_elite):
                next_pop.append(self._population[members_sorted[k]].clone())

            n_offspring = budget - n_elite
            # For tournament pool, use best half of the species (min 1).
            pool = members_sorted[:max(1, len(members_sorted) // 2)]
            for _ in range(n_offspring):
                child = self._breed_child(pool, raw_fit)
                next_pop.append(child)

        # Guard against off-by-one issues
        if len(next_pop) > self.pop_size:
            next_pop = next_pop[:self.pop_size]
        while len(next_pop) < self.pop_size:
            # Fall back: clone+mutate the global best if species budgeting left
            # us short (e.g. everything stagnated).
            seed_genome = (self._best_genome if self._best_genome is not None
                           else self._population[0])
            child = seed_genome.clone()
            self._apply_mutations(child)
            next_pop.append(child)

        self._population = next_pop
        self._generation += 1

    def _breed_child(self,
                     pool: List[int],
                     fit: np.ndarray) -> Genome:
        if self._rng.random() < self.crossover_rate and len(pool) >= 2:
            a = self._tournament(pool, fit)
            b = self._tournament(pool, fit)
            child = crossover(self._population[a], self._population[b],
                              float(fit[a]), float(fit[b]), self._rng)
        else:
            a = self._tournament(pool, fit)
            child = self._population[a].clone()
        self._apply_mutations(child)
        return child

    def _tournament(self, pool: List[int], fit: np.ndarray) -> int:
        k = min(self.tournament_size, len(pool))
        contenders = self._rng.choice(pool, size=k, replace=False)
        return int(max(contenders, key=lambda idx: fit[idx]))

    def _apply_mutations(self, g: Genome) -> None:
        mutate_weights(g, self._rng, self.weight_mutation_rate,
                       self.weight_perturb_sigma, self.weight_replace_rate)
        if self._rng.random() < self.add_connection_rate:
            mutate_add_connection(g, self._rng, self._innovation,
                                  self.max_connections)
        if self._rng.random() < self.add_node_rate:
            mutate_add_node(g, self._rng, self._innovation,
                            self.max_connections)
        mutate_toggle_enable(g, self._rng, self.toggle_enable_rate)

    # -------------------------------------------------------------- best/state

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.asarray(self._best_flat)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        flat = np.asarray(params, dtype=np.float32)
        if flat.shape != (self.param_size,):
            raise ValueError(
                f"best_params setter expected shape ({self.param_size},), "
                f"got {flat.shape}")
        self._best_flat = flat.copy()
        # We can't faithfully reconstruct a Genome from a flat vector
        # (species + innovation history is lost). This setter is intended
        # for best-parameter injection at evaluation time (e.g. Trainer
        # reloading best.npz for the demo rollout).
        self.logger.warning(
            "NEAT.best_params setter stores flat vector only; "
            "resuming training will re-speciate from the flat encoding.")

    def save_state(self) -> Any:
        state = {
            "population": copy.deepcopy(self._population),
            "species": copy.deepcopy(self._species),
            "innovation": self._innovation.state(),
            "best_fitness": self._best_fitness,
            "best_genome": copy.deepcopy(self._best_genome),
            "best_flat": self._best_flat.copy(),
            "generation": self._generation,
            "rng": self._rng.bit_generator.state,
        }
        return pickle.dumps(state)

    def load_state(self, saved_state: Any) -> None:
        state = pickle.loads(saved_state)
        self._population = state["population"]
        self._species = state["species"]
        self._innovation.load(state["innovation"])
        self._best_fitness = state["best_fitness"]
        self._best_genome = state["best_genome"]
        self._best_flat = state["best_flat"]
        self._generation = state["generation"]
        self._rng.bit_generator.state = state["rng"]
