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

"""Novelty score / archive for novelty search (Lehman & Stanley, 2011).

Pairs with any solver in :mod:`evojax.algo`: feed behavior descriptors from a
rollout, get back a novelty score per individual that can be used in place of
fitness. Pure NumPy — kept off the JAX hot path because the archive grows
across generations and would re-trigger compilation otherwise.
"""

from typing import Optional

import numpy as np


class NoveltyArchive:
    """Maintains a behavior-descriptor archive and scores novelty as mean k-NN distance.

    Following Lehman & Stanley, novelty for an individual is the mean Euclidean
    distance to its ``k`` nearest neighbors in (archive ∪ current population).
    Individuals whose novelty exceeds ``threshold`` are inserted into the
    archive — this keeps the archive growing in regions of behavior space the
    population has actually explored, so the score keeps mattering as the
    search progresses.
    """

    def __init__(self,
                 k: int = 15,
                 threshold: float = 6.0,
                 max_size: int = 2500,
                 bd_dim: Optional[int] = None):
        self.k = int(k)
        self.threshold = float(threshold)
        self.max_size = int(max_size)
        self.bd_dim = bd_dim
        self._archive: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return 0 if self._archive is None else self._archive.shape[0]

    def score(self, bds: np.ndarray) -> np.ndarray:
        """Novelty score per row of ``bds``, against archive ∪ bds itself.

        Including ``bds`` in the neighbor pool is the standard formulation —
        otherwise an early empty archive gives every individual a score of 0
        and the search stalls.
        """
        bds = np.asarray(bds, dtype=np.float32)
        n = bds.shape[0]
        if self._archive is None:
            pool = bds
        else:
            pool = np.concatenate([self._archive, bds], axis=0)
        # Pairwise distances bds (n, d) vs pool (m, d) -> (n, m)
        diff = bds[:, None, :] - pool[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        # Exclude self-matches (the bds entries that point at themselves in pool)
        # Self-matches sit at column indices [m_archive, m_archive+1, ...]
        m_archive = 0 if self._archive is None else self._archive.shape[0]
        self_idx = m_archive + np.arange(n)
        d2[np.arange(n), self_idx] = np.inf
        # k nearest (clip k to available neighbors)
        k = min(self.k, pool.shape[0] - 1)
        if k <= 0:
            return np.zeros(n, dtype=np.float32)
        # Partial sort for efficiency
        knn = np.partition(d2, k - 1, axis=1)[:, :k]
        return np.sqrt(knn).mean(axis=1).astype(np.float32)

    def update(self, bds: np.ndarray) -> int:
        """Insert any BDs whose novelty exceeds ``threshold``. Returns insert count."""
        bds = np.asarray(bds, dtype=np.float32)
        scores = self.score(bds)
        keep_mask = scores >= self.threshold
        # Always seed the archive with the first batch so subsequent novelty
        # scores have something to compare against.
        if self._archive is None:
            self._archive = bds[:1].copy()
            keep_mask[:1] = False  # already inserted
        to_add = bds[keep_mask]
        if to_add.shape[0] == 0:
            return 0
        self._archive = np.concatenate([self._archive, to_add], axis=0)
        if self._archive.shape[0] > self.max_size:
            # Drop the oldest — keeps the archive bounded without losing recent
            # exploration signal.
            self._archive = self._archive[-self.max_size:]
        return int(to_add.shape[0])
