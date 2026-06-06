"""Evaluation metrics. Multiple signals — never a single proxy."""

from __future__ import annotations

import numpy as np

from ..geometry import facility_location_value


def gold_hit_any(kept: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return 1.0 if kept & gold else 0.0


def gold_hit_all(kept: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return 1.0 if gold <= kept else 0.0


def coverage_of(kept: set[str], ids: list[str], sim_clipped: np.ndarray) -> float:
    """Facility-location coverage of the whole set by the retained subset."""
    idx = [i for i, name in enumerate(ids) if name in kept]
    if not idx:
        return 0.0
    return facility_location_value(ids, idx, sim_clipped)


def redundancy(kept: set[str], ids: list[str], sim_clipped: np.ndarray) -> float:
    """Mean best-other-similarity inside the retained set (lower = less redundant)."""
    idx = [i for i, name in enumerate(ids) if name in kept]
    if len(idx) < 2:
        return 0.0
    sub = sim_clipped[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, 0.0)
    return float(sub.max(axis=1).mean())
