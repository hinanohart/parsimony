from __future__ import annotations

import numpy as np

from _helpers import make_item
from parsimony.geometry import best_other_coverage, facility_location_value, removal_losses
from parsimony.store import MemoryPool


def _pool(vecs: dict[str, list[float]]) -> MemoryPool:
    pool = MemoryPool(dim=len(next(iter(vecs.values()))))
    for k, v in vecs.items():
        pool.add(make_item(k, k, v))
    return pool


def test_redundant_item_has_low_removal_loss():
    pool = _pool({"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]})
    ids, sim = pool.sim_matrix()
    cov = best_other_coverage(ids, sim)
    assert abs(cov["a"] - 1.0) < 1e-5  # b is a perfect stand-in
    assert abs(cov["c"]) < 1e-5  # c is unique
    rl = removal_losses(ids, sim)
    assert rl["c"] > rl["a"]  # unique item is costlier to drop


def test_single_item_pool():
    pool = _pool({"a": [1.0, 0.0]})
    ids, sim = pool.sim_matrix()
    assert best_other_coverage(ids, sim) == {"a": 0.0}
    assert removal_losses(ids, sim) == {"a": 1.0}


def test_facility_location_value_monotone():
    pool = _pool({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]})
    ids, sim = pool.sim_matrix()
    full = facility_location_value(ids, [0, 1, 2], sim)
    dropped = facility_location_value(ids, [0, 1], sim)  # drop the redundant c
    assert full >= dropped
    assert abs(full - dropped) < 1e-6  # dropping a redundant item costs ~nothing


def test_empty():
    assert removal_losses([], np.zeros((0, 0))) == {}
