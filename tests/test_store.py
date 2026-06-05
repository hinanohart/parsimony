from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_item
from parsimony.store import MemoryPool, normalize


def test_normalize_unit_norm():
    v = normalize(np.array([3.0, 4.0]))
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6
    assert v.dtype == np.float32


def test_normalize_zero_vector_stable():
    v = normalize(np.zeros(3))
    assert float(np.linalg.norm(v)) == 0.0


def test_add_get_remove_len():
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "alpha", [1.0, 0.0]))
    pool.add(make_item("b", "beta", [0.0, 1.0]))
    assert len(pool) == 2
    assert "a" in pool
    got = pool.get("a")
    assert got is not None
    assert got.text == "alpha"
    removed = pool.remove("a")
    assert removed is not None
    assert removed.id == "a"
    assert len(pool) == 1
    assert pool.remove("missing") is None


def test_ids_sorted_deterministic():
    pool = MemoryPool(dim=2)
    for i in ["c", "a", "b"]:
        pool.add(make_item(i, i, [1.0, 0.0]))
    assert pool.ids() == ["a", "b", "c"]


def test_sim_matrix_cosine():
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "alpha", [1.0, 0.0]))
    pool.add(make_item("b", "beta", [0.0, 1.0]))
    pool.add(make_item("c", "gamma", [1.0, 1.0]))
    ids, s = pool.sim_matrix()
    assert ids == ["a", "b", "c"]
    # diagonal ~ 1
    for i in range(3):
        assert abs(float(s[i, i]) - 1.0) < 1e-5
    # a vs b orthogonal ~ 0
    assert abs(float(s[0, 1])) < 1e-5
    # a vs c = cos(45deg) ~ 0.707
    assert abs(float(s[0, 2]) - 0.70710677) < 1e-4


def test_dim_mismatch_rejected():
    pool = MemoryPool(dim=2)
    with pytest.raises(ValueError):
        pool.add(make_item("a", "x", [1.0, 0.0, 0.0]))


def test_empty_matrix_shapes():
    pool = MemoryPool(dim=3)
    ids, e = pool.matrix()
    assert ids == [] and e.shape == (0, 3)
    ids, s = pool.sim_matrix()
    assert ids == [] and s.shape == (0, 0)
