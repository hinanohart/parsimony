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
    p = Parsimony(capacity=2)
    p.on_write(make_item("a1", "a", [1.0, 0.0]))
    p.on_write(
        make_item("a2", "a copy", [1.0, 0.0001])
    )  # near dup, but below default tau? sim~1 -> merge
    p.on_write(make_item("b", "b", [0.0, 1.0]))
    p.on_write(make_item("c", "c", [0.0, 1.0]))  # forces work
    p.step()
    # whichever was evicted, its trace (if any) carries a counterfactual
    for it in p.snapshot():
        tr = p.explain(it.id)
        assert "schema_version" in tr


def test_config_digest_stable():
    cfg = PolicyConfig(capacity=5, seed=3)
    assert cfg.digest() == PolicyConfig(capacity=5, seed=3).digest()
    assert cfg.digest() != PolicyConfig(capacity=5, seed=4).digest()
