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

"""NEAT genome representation plus mutation/crossover/speciation helpers.

Pure Python/NumPy. No JAX. The NEAT algorithm class drives this module
and then encodes genomes into a flat float32 vector so the evojax
``NEAlgorithm`` interface (fixed-size ``(pop_size, param_size)`` arrays)
can carry them to a paired ``NEATPolicy`` for evaluation.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np


# CPPN-style activation set. Index into this list = activation ID stored in
# the genome. Index 0 is tanh so the default (all-zeros) genome encoding
# behaves identically to plain NEAT.
CPPN_ACTIVATIONS: List[str] = [
    "tanh", "sin", "gauss", "abs", "sigmoid", "identity", "relu",
]


def neat_param_size(n_inputs: int,
                    n_outputs: int,
                    max_hidden: int,
                    max_connections: int,
                    include_activations: bool = False) -> int:
    """Size of the flat genome vector shared by algorithm and policy.

    Layout (all float32, concatenated in this order):
        weight         (max_connections,)
        enabled        (max_connections,)  # 0.0 / 1.0
        src            (max_connections,)  # integer-valued float
        dst            (max_connections,)  # integer-valued float
        bias           (n_inputs + n_outputs + max_hidden,)
        hidden_active  (max_hidden,)       # 0.0 / 1.0
        activations    (n_inputs + n_outputs + max_hidden,)
            # only when ``include_activations=True``; integer-valued float
            # ID into ``CPPN_ACTIVATIONS``. Used by HyperNEAT's CPPN.
    """
    n_nodes = n_inputs + n_outputs + max_hidden
    base = 4 * max_connections + n_nodes + max_hidden
    if include_activations:
        base += n_nodes
    return base


@dataclass
class ConnGene:
    src: int
    dst: int
    weight: float
    enabled: bool
    innovation: int


@dataclass
class Genome:
    n_inputs: int
    n_outputs: int
    max_hidden: int
    hidden_nodes: Set[int] = field(default_factory=set)
    connections: List[ConnGene] = field(default_factory=list)
    biases: Optional[np.ndarray] = None
    activations: Optional[np.ndarray] = None  # int8 ID per node; 0 = tanh

    def __post_init__(self) -> None:
        n_nodes = self.n_inputs + self.n_outputs + self.max_hidden
        if self.biases is None:
            self.biases = np.zeros(n_nodes, dtype=np.float32)
        if self.activations is None:
            self.activations = np.zeros(n_nodes, dtype=np.int8)

    @property
    def n_nodes(self) -> int:
        return self.n_inputs + self.n_outputs + self.max_hidden

    @property
    def input_range(self) -> range:
        return range(0, self.n_inputs)

    @property
    def output_range(self) -> range:
        return range(self.n_inputs, self.n_inputs + self.n_outputs)

    @property
    def hidden_range(self) -> range:
        start = self.n_inputs + self.n_outputs
        return range(start, start + self.max_hidden)

    def has_connection(self, src: int, dst: int) -> bool:
        for c in self.connections:
            if c.src == src and c.dst == dst:
                return True
        return False

    def clone(self) -> "Genome":
        return Genome(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            max_hidden=self.max_hidden,
            hidden_nodes=set(self.hidden_nodes),
            connections=[ConnGene(c.src, c.dst, c.weight, c.enabled,
                                  c.innovation) for c in self.connections],
            biases=self.biases.copy(),
            activations=self.activations.copy(),
        )


# -----------------------------------------------------------------------------
# Innovation tracking
# -----------------------------------------------------------------------------


class InnovationRegistry:
    """Maps (src, dst) pairs to a global innovation ID.

    Structurally identical mutations (same src/dst) get the same innovation ID
    across genomes and generations, which is the invariant NEAT crossover
    relies on when aligning gene lists.
    """

    def __init__(self) -> None:
        self._table: Dict[Tuple[int, int], int] = {}
        self._counter: int = 0

    def get(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._table:
            self._table[key] = self._counter
            self._counter += 1
        return self._table[key]

    def state(self) -> Dict:
        return {"table": dict(self._table), "counter": self._counter}

    def load(self, state: Dict) -> None:
        self._table = dict(state["table"])
        self._counter = state["counter"]


# -----------------------------------------------------------------------------
# Genome construction
# -----------------------------------------------------------------------------


def make_minimal_genome(n_inputs: int,
                        n_outputs: int,
                        max_hidden: int,
                        registry: InnovationRegistry,
                        rng: np.random.Generator,
                        init_weight_scale: float = 1.0,
                        activation_ids: Optional[List[int]] = None) -> Genome:
    """Fully connected inputs -> outputs with small random weights.

    When ``activation_ids`` has more than one entry, hidden-node activation
    IDs are sampled uniformly from it (CPPN init). IDs are positions into
    the global ``CPPN_ACTIVATIONS`` table so the policy can decode them
    without sharing config. Inputs and outputs keep activation 0 (tanh)
    so the substrate sees stable I/O regardless of CPPN mutations.
    """
    g = Genome(n_inputs=n_inputs, n_outputs=n_outputs, max_hidden=max_hidden)
    for src in g.input_range:
        for dst in g.output_range:
            w = float(rng.normal(0.0, init_weight_scale))
            innov = registry.get(src, dst)
            g.connections.append(
                ConnGene(src=src, dst=dst, weight=w, enabled=True,
                         innovation=innov))
    g.biases = rng.normal(0.0, 0.1, size=g.n_nodes).astype(np.float32)
    if activation_ids and len(activation_ids) > 1:
        hidden_start = n_inputs + n_outputs
        sampled = rng.choice(activation_ids, size=max_hidden)
        g.activations[hidden_start:] = sampled.astype(np.int8)
    return g


# -----------------------------------------------------------------------------
# Mutations
# -----------------------------------------------------------------------------


def mutate_weights(g: Genome,
                   rng: np.random.Generator,
                   mutation_rate: float,
                   perturb_sigma: float,
                   replace_rate: float,
                   weight_clip: float = 8.0) -> None:
    for c in g.connections:
        if rng.random() < mutation_rate:
            if rng.random() < replace_rate:
                c.weight = float(rng.normal(0.0, 1.0))
            else:
                c.weight += float(rng.normal(0.0, perturb_sigma))
            c.weight = float(np.clip(c.weight, -weight_clip, weight_clip))
    # Bias perturbation at a lower rate
    bias_mask = rng.random(g.biases.shape) < mutation_rate
    perturb = rng.normal(0.0, perturb_sigma, size=g.biases.shape)
    g.biases = np.where(bias_mask, g.biases + perturb, g.biases).astype(
        np.float32)
    np.clip(g.biases, -weight_clip, weight_clip, out=g.biases)


def _active_node_set(g: Genome) -> List[int]:
    return (list(g.input_range) + list(g.output_range) +
            sorted(g.hidden_nodes))


def mutate_add_connection(g: Genome,
                          rng: np.random.Generator,
                          registry: InnovationRegistry,
                          max_connections: int,
                          max_attempts: int = 20) -> bool:
    if len(g.connections) >= max_connections:
        return False
    nodes = _active_node_set(g)
    # Valid sources: inputs + hidden + outputs (outputs can be recurrent srcs)
    # Valid targets: outputs + hidden (never inputs)
    valid_targets = list(g.output_range) + sorted(g.hidden_nodes)
    if not valid_targets:
        return False
    for _ in range(max_attempts):
        src = int(rng.choice(nodes))
        dst = int(rng.choice(valid_targets))
        if src == dst:
            continue
        if g.has_connection(src, dst):
            continue
        innov = registry.get(src, dst)
        w = float(rng.normal(0.0, 1.0))
        g.connections.append(
            ConnGene(src=src, dst=dst, weight=w, enabled=True,
                     innovation=innov))
        return True
    return False


def mutate_add_node(g: Genome,
                    rng: np.random.Generator,
                    registry: InnovationRegistry,
                    max_connections: int) -> bool:
    # Splits an enabled connection into src -> new_hidden -> dst.
    if len(g.connections) + 2 > max_connections:
        return False
    # Need a free hidden slot
    free_hidden = [h for h in g.hidden_range if h not in g.hidden_nodes]
    if not free_hidden:
        return False
    enabled_conns = [c for c in g.connections if c.enabled]
    if not enabled_conns:
        return False
    conn = enabled_conns[int(rng.integers(len(enabled_conns)))]
    new_node = int(rng.choice(free_hidden))
    conn.enabled = False
    g.hidden_nodes.add(new_node)
    innov1 = registry.get(conn.src, new_node)
    innov2 = registry.get(new_node, conn.dst)
    g.connections.append(
        ConnGene(src=conn.src, dst=new_node, weight=1.0, enabled=True,
                 innovation=innov1))
    g.connections.append(
        ConnGene(src=new_node, dst=conn.dst, weight=conn.weight, enabled=True,
                 innovation=innov2))
    # Small bias for the newly activated hidden node
    g.biases[new_node] = float(rng.normal(0.0, 0.1))
    return True


def mutate_toggle_enable(g: Genome,
                         rng: np.random.Generator,
                         rate: float) -> None:
    for c in g.connections:
        if rng.random() < rate:
            c.enabled = not c.enabled


def mutate_activations(g: Genome,
                       rng: np.random.Generator,
                       rate: float,
                       activation_ids: List[int]) -> None:
    """Re-roll activation IDs on hidden nodes with probability ``rate``.

    Only mutates hidden nodes; inputs/outputs stay at activation 0 so the
    substrate query always reads the CPPN's output the same way. ``rate``
    is per-node, so ``len(activation_ids)*rate*max_hidden`` is roughly the
    expected per-genome activation churn.
    """
    if len(activation_ids) <= 1 or rate <= 0.0:
        return
    hidden_start = g.n_inputs + g.n_outputs
    for i in range(hidden_start, g.n_nodes):
        if rng.random() < rate:
            g.activations[i] = int(rng.choice(activation_ids))


# -----------------------------------------------------------------------------
# Crossover
# -----------------------------------------------------------------------------


def crossover(parent_a: Genome,
              parent_b: Genome,
              fit_a: float,
              fit_b: float,
              rng: np.random.Generator,
              disabled_inherit_prob: float = 0.75) -> Genome:
    """NEAT crossover aligned by innovation ID.

    Matching genes inherit from a random parent. Disjoint/excess genes inherit
    from the fitter parent; if fitnesses tie, they inherit randomly.
    """
    if fit_a >= fit_b:
        fitter, other = parent_a, parent_b
        fit_fitter, fit_other = fit_a, fit_b
    else:
        fitter, other = parent_b, parent_a
        fit_fitter, fit_other = fit_b, fit_a
    tie = fit_fitter == fit_other

    by_innov_a = {c.innovation: c for c in fitter.connections}
    by_innov_b = {c.innovation: c for c in other.connections}

    child = Genome(n_inputs=fitter.n_inputs, n_outputs=fitter.n_outputs,
                   max_hidden=fitter.max_hidden)
    child.biases = fitter.biases.copy()
    child.activations = fitter.activations.copy()
    # Average biases on matching nodes if the structural footprint overlaps.
    # For activations, inherit from a random parent on overlap (averaging
    # categorical IDs is meaningless).
    overlap = fitter.hidden_nodes & other.hidden_nodes
    for h in overlap:
        child.biases[h] = 0.5 * (fitter.biases[h] + other.biases[h])
        if rng.random() < 0.5:
            child.activations[h] = other.activations[h]

    all_innov = set(by_innov_a.keys()) | set(by_innov_b.keys())
    for innov in sorted(all_innov):
        in_a = innov in by_innov_a
        in_b = innov in by_innov_b
        if in_a and in_b:
            pick = by_innov_a[innov] if rng.random() < 0.5 else by_innov_b[innov]
            new_gene = ConnGene(src=pick.src, dst=pick.dst, weight=pick.weight,
                                enabled=pick.enabled, innovation=innov)
            if (not by_innov_a[innov].enabled) or (not by_innov_b[innov].enabled):
                if rng.random() < disabled_inherit_prob:
                    new_gene.enabled = False
                else:
                    new_gene.enabled = True
            child.connections.append(new_gene)
        elif in_a:
            child.connections.append(_copy_gene(by_innov_a[innov]))
        elif in_b and tie:
            child.connections.append(_copy_gene(by_innov_b[innov]))
        # else: disjoint/excess of the less fit parent is dropped

    # Rebuild hidden node set from the genes we kept
    for c in child.connections:
        for node in (c.src, c.dst):
            if node in fitter.hidden_range:
                child.hidden_nodes.add(node)
    return child


def _copy_gene(c: ConnGene) -> ConnGene:
    return ConnGene(src=c.src, dst=c.dst, weight=c.weight, enabled=c.enabled,
                    innovation=c.innovation)


# -----------------------------------------------------------------------------
# Speciation
# -----------------------------------------------------------------------------


def compatibility_distance(g_a: Genome,
                           g_b: Genome,
                           c1: float,
                           c2: float,
                           c3: float) -> float:
    by_innov_a = {c.innovation: c for c in g_a.connections}
    by_innov_b = {c.innovation: c for c in g_b.connections}
    innovs_a = set(by_innov_a)
    innovs_b = set(by_innov_b)
    if not innovs_a and not innovs_b:
        return 0.0
    max_innov_a = max(innovs_a) if innovs_a else -1
    max_innov_b = max(innovs_b) if innovs_b else -1
    common_max = min(max_innov_a, max_innov_b)

    matching = innovs_a & innovs_b
    excess = 0
    disjoint = 0
    for innov in innovs_a ^ innovs_b:
        if innov > common_max:
            excess += 1
        else:
            disjoint += 1

    if matching:
        w_diff = float(np.mean([
            abs(by_innov_a[i].weight - by_innov_b[i].weight)
            for i in matching
        ]))
    else:
        w_diff = 0.0

    n = max(len(innovs_a), len(innovs_b), 1)
    if n < 20:
        n = 1  # Classic NEAT: don't normalize small genomes
    return c1 * excess / n + c2 * disjoint / n + c3 * w_diff


@dataclass
class Species:
    representative: Genome
    members: List[int] = field(default_factory=list)  # indices into population
    best_fitness: float = -np.inf
    stagnation: int = 0
    # Stable, unique-across-the-run identifier. Assigned by NEAT after
    # speciate() places members; preserved across generations via
    # ``clone_shallow``. Defaults to -1 to flag "needs assignment".
    species_id: int = -1
    # Generation in which this species was first instantiated. Together with
    # the algorithm's current generation gives the species' age.
    birth_gen: int = -1

    def clone_shallow(self) -> "Species":
        s = Species(representative=self.representative.clone())
        s.best_fitness = self.best_fitness
        s.stagnation = self.stagnation
        s.species_id = self.species_id
        s.birth_gen = self.birth_gen
        return s


def speciate(population: List[Genome],
             prior_species: List[Species],
             compat_threshold: float,
             c1: float,
             c2: float,
             c3: float,
             rng: np.random.Generator) -> List[Species]:
    """Assign each genome in ``population`` to a species.

    Uses prior_species representatives if any; otherwise seeds new species.
    Returns fresh Species objects whose ``members`` list holds indices into
    ``population``.
    """
    active: List[Species] = [s.clone_shallow() for s in prior_species]
    for idx, genome in enumerate(population):
        placed = False
        for species in active:
            dist = compatibility_distance(genome, species.representative,
                                          c1, c2, c3)
            if dist < compat_threshold:
                species.members.append(idx)
                placed = True
                break
        if not placed:
            new_species = Species(representative=genome.clone())
            new_species.members.append(idx)
            active.append(new_species)

    # Drop empty species (reps whose cluster found no members this gen)
    active = [s for s in active if s.members]
    # Refresh representatives: random member from the newly-formed species
    for species in active:
        rep_idx = int(rng.choice(species.members))
        species.representative = population[rep_idx].clone()
    return active


def fitness_share(fitnesses: np.ndarray,
                  species: List[Species]) -> np.ndarray:
    shared = np.zeros_like(fitnesses, dtype=np.float64)
    for sp in species:
        if not sp.members:
            continue
        size = len(sp.members)
        for m in sp.members:
            shared[m] = fitnesses[m] / size
    return shared
