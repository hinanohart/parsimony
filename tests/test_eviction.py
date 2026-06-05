from __future__ import annotations

from _helpers import make_item
from parsimony.config import PolicyConfig
from parsimony.eviction import evict
from parsimony.store import MemoryPool


def _cluster_pool() -> MemoryPool:
    pool = MemoryPool(dim=3)
    pool.add(make_item("a1", "alpha one", [1.0, 0.0, 0.0]))
    pool.add(make_item("a2", "alpha two", [1.0, 0.0, 0.0]))  # dup of a1
    pool.add(make_item("b1", "beta one", [0.0, 1.0, 0.0]))
    pool.add(make_item("b2", "beta two", [0.0, 1.0, 0.0]))  # dup of b1
    pool.add(make_item("c1", "gamma one", [0.0, 0.0, 1.0]))  # unique
    return pool


def test_eviction_keeps_one_representative_per_cluster():
    pool = _cluster_pool()
    cfg = PolicyConfig(capacity=3)
    decisions = evict(pool, cfg)
    assert len(decisions) == 2
    kept = set(pool.ids())
    assert len(kept) == 3
    assert "c1" in kept  # unique item is never dropped
    assert len({"a1", "a2"} & kept) == 1  # exactly one alpha survives
    assert len({"b1", "b2"} & kept) == 1  # exactly one beta survives


def test_evicts_redundant_not_unique():
    pool = _cluster_pool()
    cfg = PolicyConfig(capacity=3)
    decisions = evict(pool, cfg)
    evicted_ids = {d.item_id for d in decisions}
    assert "c1" not in evicted_ids
    # each evicted duplicate is represented by its surviving twin at sim ~1
    for d in decisions:
        assert d.nearest_after_id is not None
        assert d.trace["nearest_after_sim"] > 0.99


def test_explicit_n():
    pool = _cluster_pool()
    cfg = PolicyConfig(capacity=1)
    decisions = evict(pool, cfg, n=1)
    assert len(decisions) == 1
    assert len(pool) == 4


def test_no_eviction_below_capacity():
    pool = _cluster_pool()
    cfg = PolicyConfig(capacity=10)
    assert evict(pool, cfg) == []
    assert len(pool) == 5


def test_eviction_deterministic():
    def run() -> list[str]:
        pool = _cluster_pool()
        return [d.item_id for d in evict(pool, PolicyConfig(capacity=3))]

    assert run() == run()
