from __future__ import annotations

from parsimony.sketch import CountMinSketch4Bit, Doorkeeper


def test_frequency_orders_hot_over_cold():
    cms = CountMinSketch4Bit(capacity=64, seed=0)
    for _ in range(10):
        cms.increment("hot")
    cms.increment("cold")
    assert cms.estimate("hot") >= 5
    assert cms.estimate("cold") == 1  # doorkeeper bit only
    assert cms.estimate("never-seen") == 0


def test_one_hit_wonder_does_not_pollute_sketch():
    cms = CountMinSketch4Bit(capacity=64, seed=0)
    cms.increment("wonder")  # first sighting -> doorkeeper only, counters untouched
    # every raw counter row is still zero for this key
    idx = cms._indices("wonder")
    assert all(int(cms.table[r, idx[r]]) == 0 for r in range(cms.depth))


def test_determinism_same_seed_same_state():
    a = CountMinSketch4Bit(capacity=64, seed=7)
    b = CountMinSketch4Bit(capacity=64, seed=7)
    for key in ["x", "y", "x", "z", "x", "y"]:
        a.increment(key)
        b.increment(key)
    assert (a.table == b.table).all()
    assert a.estimate("x") == b.estimate("x")


def test_different_seed_differs_in_layout():
    a = CountMinSketch4Bit(capacity=64, seed=1)
    b = CountMinSketch4Bit(capacity=64, seed=2)
    assert a._indices("same-key") != b._indices("same-key")


def test_aging_bounds_size():
    cms = CountMinSketch4Bit(capacity=4, sample_multiplier=10, seed=0)
    for _ in range(500):
        cms.increment("x")
    assert cms.size < cms.sample_size  # at least one aging reset happened
    assert cms.estimate("x") > 0


def test_doorkeeper_add_reports_presence():
    dk = Doorkeeper(capacity=32, seed=0)
    assert dk.add("k") is False
    assert dk.add("k") is True
    assert dk.contains("k") is True
    dk.clear()
    assert dk.contains("k") is False
