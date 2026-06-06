from __future__ import annotations

import json

import numpy as np

from parsimony.adapters.embedding import HashingEmbedder
from parsimony.loaders.longmemeval import date_key, load_longmemeval
from parsimony.metrics import coverage_of, gold_hit_all, gold_hit_any, redundancy
from parsimony.trace.adapter import generate_synthetic_trace, question_to_items


def _fake_lme(tmp_path):
    rec = {
        "question_id": "q1",
        "question_type": "single-session",
        "question": "when is the birthday?",
        "answer": "March 3",
        "question_date": "2023/05/01 (Mon) 10:00",
        "haystack_dates": ["2023/01/02 (Mon) 09:00", "2023/03/04 (Sat) 12:00"],
        "haystack_session_ids": ["s_old", "answer_gold"],
        "haystack_sessions": [
            [{"role": "user", "content": "old chatter about weather"}],
            [{"role": "user", "content": "my birthday is March 3"}],
        ],
        "answer_session_ids": ["answer_gold"],
    }
    p = tmp_path / "fake.json"
    p.write_text(json.dumps([rec]), encoding="utf-8")
    return p


def test_load_longmemeval(tmp_path):
    path = _fake_lme(tmp_path)
    qs, sha = load_longmemeval(path)
    assert len(qs) == 1
    assert qs[0].gold_session_ids == ["answer_gold"]
    assert len(sha) == 64


def test_date_key_orders():
    assert date_key("2023/01/02 (Mon) 09:00", 0) < date_key("2023/03/04 (Sat) 12:00", 1)
    assert date_key("garbage", 5)[0] == 9999


def test_question_to_items_chronological(tmp_path):
    path = _fake_lme(tmp_path)
    qs, _ = load_longmemeval(path)
    items, gold = question_to_items(qs[0], HashingEmbedder(dim=32))
    assert [it.id for it in items] == ["s_old", "answer_gold"]  # chronological
    assert gold == {"answer_gold"}


def test_synthetic_trace_deterministic_with_gold():
    a, ga = generate_synthetic_trace(n_items=30, n_clusters=5, dim=16, seed=7)
    b, gb = generate_synthetic_trace(n_items=30, n_clusters=5, dim=16, seed=7)
    assert [x.id for x in a] == [x.id for x in b]
    assert ga == gb and len(ga) >= 1


def test_metrics_basic():
    ids = ["a", "b", "c"]
    sim = np.array([[1.0, 0.0, 0.9], [0.0, 1.0, 0.0], [0.9, 0.0, 1.0]])
    assert gold_hit_any({"a"}, {"a", "z"}) == 1.0
    assert gold_hit_any({"a"}, {"z"}) == 0.0
    assert gold_hit_all({"a", "b"}, {"a", "b"}) == 1.0
    assert gold_hit_all({"a"}, {"a", "b"}) == 0.0
    assert coverage_of(set(), ids, sim) == 0.0
    assert coverage_of({"a", "b", "c"}, ids, sim) > 0.0
    assert redundancy({"a", "c"}, ids, sim) > redundancy({"a", "b"}, ids, sim)
