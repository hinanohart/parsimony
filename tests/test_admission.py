from __future__ import annotations

from _helpers import make_item
from parsimony.admission import admit, find_victim
from parsimony.config import PolicyConfig
from parsimony.sketch import CountMinSketch4Bit
from parsimony.store import MemoryPool
from parsimony.types import DecisionType


def _cms(counts: dict[str, int], capacity: int = 8, seed: int = 0) -> CountMinSketch4Bit:
    cms = CountMinSketch4Bit(capacity=capacity, seed=seed)
    for k, c in counts.items():
        for _ in range(c):
            cms.increment(k)
    return cms


def test_below_capacity_admits():
    cfg = PolicyConfig(capacity=4)
    pool = MemoryPool(dim=2)
    cms = CountMinSketch4Bit(capacity=4)
    d = admit(make_item("x", "x", [1.0, 0.0]), pool, cms, cfg)
    assert d.admitted is True
    assert d.decision == DecisionType.ADMIT


def test_hot_candidate_evicts_cold_victim():
    cfg = PolicyConfig(capacity=2)
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "a", [1.0, 0.0]))
    pool.add(make_item("b", "b", [0.0, 1.0]))
    cms = _cms({"a": 1, "b": 1, "c": 14}, capacity=2)
    d = admit(make_item("c", "c", [1.0, 1.0]), pool, cms, cfg)
    assert d.admitted is True
    assert d.trace["victim_id"] in {"a", "b"}


def test_cold_candidate_rejected():
    cfg = PolicyConfig(capacity=2)
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "a", [1.0, 0.0]))
    pool.add(make_item("b", "b", [0.0, 1.0]))
    cms = _cms({"a": 14, "b": 14, "c": 1}, capacity=2)
    d = admit(make_item("c", "c", [1.0, 1.0]), pool, cms, cfg)
    assert d.admitted is False
    assert d.decision == DecisionType.REJECT


def test_find_victim_is_lowest_utility():
    cfg = PolicyConfig(capacity=3)
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "a", [1.0, 0.0]))
    pool.add(make_item("b", "b", [0.0, 1.0]))
    pool.add(make_item("c", "c", [1.0, 1.0]))
    cms = _cms({"a": 14, "b": 14, "c": 1}, capacity=3)
    assert find_victim(pool, cms, cfg) == "c"


def test_admission_deterministic():
    cfg = PolicyConfig(capacity=2)

    def run() -> tuple[bool, str]:
        pool = MemoryPool(dim=2)
        pool.add(make_item("a", "a", [1.0, 0.0]))
        pool.add(make_item("b", "b", [0.0, 1.0]))
        cms = _cms({"a": 1, "b": 1, "c": 14}, capacity=2)
        d = admit(make_item("c", "c", [1.0, 1.0]), pool, cms, cfg)
        return d.admitted, d.reason

    assert run() == run()
