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

"""Per-species per-generation telemetry for NEAT.

The :class:`NEAT` algorithm appends one :class:`SpeciesGenStats` row per live
species per ``tell()`` call; on ``flush()`` the rows are written as a CSV.
This is the data source for diagnostic plots (per-species fitness
trajectories, improvement-rate histograms, retirement audits) and the
empirical evidence for whether protection-budget interventions help.

Kept off the JAX hot path — pure Python, runs on the CPU once per
generation, costs O(n_species) per call.
"""

import csv
from dataclasses import asdict
from dataclasses import dataclass
from typing import List
from typing import Optional


@dataclass
class SpeciesGenStats:
    """One row per (species, generation). Fields chosen to support the Phase 1
    diagnostic plots without re-running training to compute them."""
    generation: int
    species_id: int
    n_members: int
    age: int                    # generation - birth_gen
    max_fit: float              # raw best fitness in species this gen
    mean_fit: float
    fit_var: float
    n_conns_avg: float          # avg enabled connections across members
    n_hidden_avg: float         # avg active hidden nodes across members
    improvement_rate: float     # max_fit_t - max_fit_{t-window}, or NaN if too young
    n_offspring: int            # offspring budget assigned this gen
    retired: int                # 1 if species was retired this generation


class SpeciesHistory:
    """Append-only collector of per-species per-generation telemetry rows."""

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self._rows: List[SpeciesGenStats] = []

    def add(self, row: SpeciesGenStats) -> None:
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def flush(self, path: Optional[str] = None) -> str:
        """Write rows to CSV. Returns the path written to."""
        out = path or self.output_path
        if out is None:
            raise ValueError(
                "SpeciesHistory.flush requires a path (none configured)")
        if not self._rows:
            # Still write an empty file with a header so downstream readers
            # don't crash on missing files.
            fieldnames = list(SpeciesGenStats.__dataclass_fields__.keys())
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return out
        fieldnames = list(asdict(self._rows[0]).keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._rows:
                writer.writerow(asdict(r))
        return out
