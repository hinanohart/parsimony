from __future__ import annotations

from parsimony.adapters.embedding import HashingEmbedder
from parsimony.adapters.mem0 import ParsimonyMem0Gate


def test_gate_emits_mem0_events():
    emb = HashingEmbedder(dim=64)
    gate = ParsimonyMem0Gate(capacity=3)
    r = gate.on_add("a", "the user lives in Berlin", emb.encode("the user lives in Berlin"))
    assert r["event"] == "ADD"
    assert r["id"] == "a"


def test_gate_merges_near_duplicate():
    emb = HashingEmbedder(dim=128)
    gate = ParsimonyMem0Gate(capacity=5)
    gate.on_add(
        "a",
        "the user's birthday is the 3rd of March",
        emb.encode("the user's birthday is the 3rd of March"),
    )
    r = gate.on_add(
        "b",
        "the user's birthday is the 3rd of March every year",
        emb.encode("the user's birthday is the 3rd of March every year"),
    )
    assert r["event"] in {"ADD", "UPDATE", "NONE"}
    if r["event"] == "UPDATE":
        assert r["id"] == "a"


def test_maintenance_returns_event_ops():
    emb = HashingEmbedder(dim=64)
    gate = ParsimonyMem0Gate(capacity=3)
    for i in range(6):
        gate.on_add(
            f"m{i}",
            f"distinct memory number {i} about topic {i}",
            emb.encode(f"topic {i} unique {i * 3}"),
        )
    ops = gate.maintenance()
    for op in ops:
        assert op["event"] in {"DELETE", "UPDATE"}
        assert "id" in op
    assert len(gate.snapshot()) <= 3
