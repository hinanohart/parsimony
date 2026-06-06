"""mem0 drop-in: route mem0's add / maintenance path through a parsimony policy.

mem0 (v3) extracts memories with an LLM, dedups by md5 of the text, then writes
each via ``_create_memory``; it has no capacity, eviction, or compression. This
gate adds those, translating parsimony's decisions into mem0's
``{id, memory, event}`` vocabulary (``ADD`` / ``UPDATE`` / ``DELETE`` / ``NONE``).

It does **not** import mem0 — pass the text and embedding you already compute, so
the integration works whether or not mem0 is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import PolicyConfig
from ..policy import Parsimony
from ..store import normalize
from ..types import DecisionType, MemoryItem


class ParsimonyMem0Gate:
    """Wrap a Parsimony policy behind mem0's add/maintenance contract."""

    def __init__(self, capacity: int, *, config: PolicyConfig | None = None, seed: int = 0):
        self.policy = Parsimony(capacity, config=config, seed=seed)

    def on_add(self, memory_id: str, text: str, embedding: np.ndarray) -> dict[str, Any]:
        """Decide a single incoming memory. Returns a mem0-style event dict."""
        item = MemoryItem(id=memory_id, text=text, embedding=normalize(np.asarray(embedding)))
        d = self.policy.on_write(item)
        if d.decision == DecisionType.MERGE:
            return {"event": "UPDATE", "id": d.merged_into, "reason": d.reason}
        if d.admitted:
            return {"event": "ADD", "id": memory_id, "reason": d.reason}
        return {"event": "NONE", "id": memory_id, "reason": d.reason}

    def maintenance(self) -> list[dict[str, Any]]:
        """Run a maintenance sweep; map evictions to DELETE and compressions to UPDATE."""
        report = self.policy.step()
        ops: list[dict[str, Any]] = []
        for e in report.evicted:
            ops.append({"event": "DELETE", "id": e.item_id, "reason": e.reason})
        for c in report.compressed:
            if c.chosen_level > 0:
                ops.append({"event": "UPDATE", "id": c.item_id, "reason": c.reason})
        return ops

    def explain(self, memory_id: str) -> dict[str, Any]:
        return self.policy.explain(memory_id)

    def snapshot(self) -> list[MemoryItem]:
        return self.policy.snapshot()
