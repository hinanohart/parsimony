"""Similarity-geometry primitives shared by eviction, compression, and the objective.

Coverage uses the facility-location notion: an item is "covered" by the best
(clipped, non-negative) cosine similarity to any kept item. These helpers are
pure functions of the (sorted ids, similarity matrix) pair, so they are
deterministic and reused everywhere.
"""

from __future__ import annotations

import numpy as np


def best_other_coverage(ids: list[str], sim: np.ndarray) -> dict[str, float]:
    """For each item, the best non-negative cosine similarity to any *other* item.

    This is how well the rest of the pool would still represent the item if it
    were removed (m2 in the docs).
    """
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 0.0}
    sc = np.clip(np.asarray(sim, dtype=np.float64), 0.0, None).copy()
    np.fill_diagonal(sc, -np.inf)
    m2 = sc.max(axis=1)
    return {ids[i]: float(max(0.0, m2[i])) if np.isfinite(m2[i]) else 0.0 for i in range(n)}


def removal_losses(ids: list[str], sim: np.ndarray) -> dict[str, float]:
    """First-order facility-location marginal loss of removing each item.

    With every item currently kept, removing item v only drops v's own coverage
    from 1 to its best-other-coverage, so loss = (1 - m2[v]) / n. Larger means
    "more unique, keep it".
    """
    n = len(ids)
    if n == 0:
        return {}
    cov = best_other_coverage(ids, sim)
    return {i: (1.0 - cov[i]) / n for i in cov}


def facility_location_value(ids: list[str], kept_idx: list[int], sim: np.ndarray) -> float:
    """Coverage of the full universe (all rows) by the kept columns, in [0, 1]."""
    n = len(ids)
    if n == 0 or not kept_idx:
        return 0.0
    sc = np.clip(np.asarray(sim, dtype=np.float64), 0.0, None)
    sub = sc[:, list(kept_idx)]
    return float(sub.max(axis=1).sum() / n)
