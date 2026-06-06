from __future__ import annotations

from _helpers import make_item
from parsimony import Parsimony, PolicyConfig
from parsimony.types import DecisionType


def test_below_capacity_admits_and_grows():
    p = Parsimony(capacity=4)
    d = p.on_write(make_item("a", "alpha", [1.0, 0.0]))
    assert d.admitted is True
    assert len(p) == 1


def test_capacity_invariant_via_displacement():
    p = Parsimony(capacity=2)
    p.on_write(make_item("a", "a", [1.0, 0.0]))
    p.on_write(make_item("b", "b", [0.0, 1.0]))
    # reinforce a and b a little, then hammer a hot newcomer c
    for _ in range(15):
        p.touch("a")
    p.on_write(make_item("c", "c", [0.3, 0.7]))
    assert len(p) <= 2  # invariant never exceeded


def test_semantic_dedup_merges_paraphrase():
    p = Parsimony(capacity=10)
    p.on_write(make_item("a", "the cat sat on the mat", [1.0, 0.0]))
    d = p.on_write(make_item("b", "a cat is sitting on a mat", [0.999, 0.001]))
    assert d.decision == DecisionType.MERGE
    assert d.merged_into == "a"
    assert len(p) == 1  # no new item created


def test_distinct_writes_not_merged():
    p = Parsimony(capacity=10)
    p.on_write(make_item("a", "the cat sat", [1.0, 0.0]))
    d = p.on_write(make_item("b", "stocks fell sharply", [0.0, 1.0]))
    assert d.decision == DecisionType.ADMIT
    assert len(p) == 2


def test_step_runs_all_operators_deterministically():
    def build() -> Parsimony:
        p = Parsimony(capacity=3, config=PolicyConfig(capacity=3, dedup_threshold=0.95))
        p.on_write(make_item("a1", "alpha one two three four five", [1.0, 0.0, 0.0]))
        p.on_write(make_item("a2", "alpha six seven eight nine ten", [0.999, 0.001, 0.0]))
        p.on_write(make_item("b1", "beta uno dos tres", [0.0, 1.0, 0.0]))
        p.on_write(make_item("c1", "gamma 123 456 789 000 111", [0.0, 0.0, 1.0]))
        return p

    p1 = build()
    rep = p1.step()
    ids_after = [it.id for it in p1.snapshot()]
    p2 = build()
    p2.step()
    assert ids_after == [it.id for it in p2.snapshot()]
    # at least one of the operators acted
    assert rep.merged or rep.evicted or rep.compressed


def test_explain_returns_trace():
    p = Parsimony(capacity=2)
    p.on_write(make_item("a", "a", [1.0, 0.0]))
    tr = p.explain("a")
    assert tr["item_id"] == "a"
    assert tr["decision"] == "admit"
    assert "config_digest" in tr and "policy_seed" in tr


def test_snapshot_sorted():
    p = Parsimony(capacity=5)
    for i in ["c", "a", "b"]:
        p.on_write(make_item(i, i, [1.0, 0.0, 0.0]))
    # all distinct directions to avoid dedup
    p2 = Parsimony(capacity=5)
    p2.on_write(make_item("c", "c", [1.0, 0.0, 0.0]))
    p2.on_write(make_item("a", "a", [0.0, 1.0, 0.0]))
    p2.on_write(make_item("b", "b", [0.0, 0.0, 1.0]))
    assert [it.id for it in p2.snapshot()] == ["a", "b", "c"]
