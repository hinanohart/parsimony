"""End-to-end determinism: identical inputs and seed produce identical state."""

from __future__ import annotations

from _helpers import make_item
from parsimony import Parsimony, PolicyConfig

_VECS = [
    ("m0", "the quarterly revenue rose to 4200 dollars this year", [1.0, 0.0, 0.0]),
    ("m1", "a cat sat quietly on the warm mat", [0.0, 1.0, 0.0]),
    ("m2", "the revenue grew to 4200 usd in the quarter", [0.98, 0.02, 0.0]),
    ("m3", "stocks fell sharply after the announcement today", [0.0, 0.0, 1.0]),
    ("m4", "a kitten was resting on the rug all afternoon", [0.05, 0.99, 0.0]),
]


def _run() -> list[tuple[str, str, int]]:
    p = Parsimony(capacity=3, config=PolicyConfig(capacity=3, seed=11))
    for item_id, text, vec in _VECS:
        p.on_write(make_item(item_id, text, vec, salience=1.0 + len(item_id)))
        p.touch(item_id)
    p.step()
    return [(it.id, it.text, it.compression_level) for it in p.snapshot()]


def test_full_pipeline_deterministic():
    assert _run() == _run()


def test_explain_digest_stable_across_runs():
    p1 = Parsimony(capacity=3, config=PolicyConfig(capacity=3, seed=11))
    p2 = Parsimony(capacity=3, config=PolicyConfig(capacity=3, seed=11))
    for item_id, text, vec in _VECS[:3]:
        p1.on_write(make_item(item_id, text, vec))
        p2.on_write(make_item(item_id, text, vec))
    for item_id, _, _ in _VECS[:3]:
        t1, t2 = p1.explain(item_id), p2.explain(item_id)
        assert t1.get("config_digest") == t2.get("config_digest")
        assert t1.get("decision") == t2.get("decision")
