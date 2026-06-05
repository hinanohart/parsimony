"""Shared test helpers (deterministic item construction)."""

from __future__ import annotations

import numpy as np

from parsimony.store import normalize
from parsimony.types import MemoryItem


def make_item(
    item_id: str,
    text: str,
    vec: list[float],
    *,
    created_at: float = 0.0,
    salience: float = 1.0,
    tokens: int = 0,
) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        text=text,
        embedding=normalize(np.asarray(vec, dtype=np.float32)),
        created_at=created_at,
        last_access=created_at,
        salience=salience,
        tokens=tokens or len(text.split()),
    )
