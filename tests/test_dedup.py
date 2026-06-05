from __future__ import annotations

from _helpers import make_item
from parsimony.config import PolicyConfig
from parsimony.dedup import find_duplicate, merge
from parsimony.store import MemoryPool


def test_finds_near_duplicate():
    cfg = PolicyConfig(dedup_threshold=0.92)
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "the cat sat", [1.0, 0.0]))
    cand = make_item("b", "a cat was sitting", [0.99, 0.01])
    match, sim = find_duplicate(cand, pool, cfg)
    assert match == "a"
    assert sim >= 0.92


def test_distinct_is_not_duplicate():
    cfg = PolicyConfig(dedup_threshold=0.92)
    pool = MemoryPool(dim=2)
    pool.add(make_item("a", "the cat sat", [1.0, 0.0]))
    cand = make_item("b", "stock prices fell", [0.0, 1.0])
    match, sim = find_duplicate(cand, pool, cfg)
    assert match is None
    assert sim < 0.92


def test_empty_pool_no_duplicate():
    cfg = PolicyConfig()
    pool = MemoryPool(dim=2)
    match, sim = find_duplicate(make_item("a", "x", [1.0, 0.0]), pool, cfg)
    assert match is None and sim == 0.0


def test_merge_keeps_higher_salience_text_and_unions_provenance():
    existing = make_item("a", "old text", [1.0, 0.0], salience=1.0, created_at=5.0)
    existing = existing.evolve(source="sess1")
    candidate = make_item("b", "new better text", [1.0, 0.0], salience=2.0, created_at=3.0)
    candidate = candidate.evolve(source="sess2")
    m = merge(existing, candidate)
    assert m.id == "a"  # keeps existing id
    assert m.text == "new better text"  # higher salience wins
    assert m.salience == 2.0
    assert m.created_at == 3.0  # oldest origin
    assert m.source == "sess1+sess2"


def test_merge_is_deterministic():
    e = make_item("a", "x", [1.0, 0.0], salience=1.0, created_at=5.0)
    c = make_item("b", "y", [1.0, 0.0], salience=2.0, created_at=3.0)
    assert merge(e, c).text == merge(e, c).text
