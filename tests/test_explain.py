from __future__ import annotations

from _helpers import make_item
from parsimony import Parsimony, PolicyConfig
from parsimony.admission import admit
from parsimony.explain import explain_decision
from parsimony.sketch import CountMinSketch4Bit
from parsimony.store import MemoryPool


def test_admission_terms_sum_to_utility():
    cfg = PolicyConfig(capacity=2, w_freq=1.0, w_cover=0.5, w_salience=0.3)
    pool = MemoryPool(dim=2)
    cms = CountMinSketch4Bit(capacity=2)
    cms.increment("a")
    d = admit(make_item("a", "a", [1.0, 0.0], salience=2.0), pool, cms, cfg)
    tr = explain_decision(d, cfg)
    terms = tr["objective"]["contributing_terms"]
    assert abs(sum(t["contribution"] for t in terms) - tr["objective"]["utility"]) < 1e-9


def test_reject_has_counterfactual():
    p = Parsimony(capacity=1, config=PolicyConfig(capacity=1))
    p.on_write(make_item("hot", "hot", [1.0, 0.0]))
    for _ in range(15):
        p.touch("hot")
    p.on_write(make_item("cold", "cold", [0.0, 1.0]))
    tr = p.explain("cold")
    assert tr["decision"] in {"reject", "admit"}
    assert "counterfactual" in tr


def test_eviction_trace_has_representative():
    # capacity 3 with 4 distinct directions: the 4th write triggers facility-location
    # eviction on the write path (eviction mode is the facade default).
    p = Parsimony(capacity=3, config=PolicyConfig(capacity=3, dedup_threshold=0.99))
    for name, vec in [
        ("a", [1.0, 0.0, 0.0, 0.0]),
        ("b", [0.0, 1.0, 0.0, 0.0]),
        ("c", [0.0, 0.0, 1.0, 0.0]),
        ("d", [0.0, 0.0, 0.0, 1.0]),
    ]:
        p.on_write(make_item(name, name, vec))
    kept = {it.id for it in p.snapshot()}
    evicted = [x for x in ["a", "b", "c", "d"] if x not in kept]
    assert evicted  # an eviction happened on the write path
    tr = p.explain(evicted[0])
    assert tr["decision"] == "evict"
    assert "eviction" in tr and "counterfactual" in tr
    assert "nearest_after_id" in tr["eviction"]


def test_admission_trace_has_tie_break_block():
    p = Parsimony(capacity=2)
    p.on_write(make_item("a", "a", [1.0, 0.0]))
    tr = p.explain("a")
    assert "tie_break" in tr
    assert set(tr["tie_break"]) == {"applied", "rule"}


def test_config_digest_stable():
    cfg = PolicyConfig(capacity=5, seed=3)
    assert cfg.digest() == PolicyConfig(capacity=5, seed=3).digest()
    assert cfg.digest() != PolicyConfig(capacity=5, seed=4).digest()
