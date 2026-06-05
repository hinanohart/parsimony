"""The memory pool: a similarity-aware container with no policy of its own.

``MemoryPool`` holds items and exposes the embedding matrix and cosine
similarity matrix. Iteration order is always *sorted by id* so every downstream
algorithm is deterministic. The pool never decides what to keep or drop; that
is the job of the policy modules.
"""

from __future__ import annotations

import numpy as np

from .types import MemoryItem


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 copy of ``vec`` (zero vector unchanged)."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32)


class MemoryPool:
    def __init__(self, dim: int):
        self.dim = dim
        self.items: dict[str, MemoryItem] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.items

    def ids(self) -> list[str]:
        """Item ids in deterministic (sorted) order."""
        return sorted(self.items)

    def get(self, item_id: str) -> MemoryItem | None:
        return self.items.get(item_id)

    def add(self, item: MemoryItem) -> None:
        if item.embedding.shape[0] != self.dim:
            raise ValueError(f"embedding dim {item.embedding.shape[0]} != pool dim {self.dim}")
        self.items[item.id] = item

    def remove(self, item_id: str) -> MemoryItem | None:
        return self.items.pop(item_id, None)

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (sorted ids, E) where E is the (n, dim) embedding matrix."""
        ids = self.ids()
        if not ids:
            return ids, np.zeros((0, self.dim), dtype=np.float32)
        e = np.stack([self.items[i].embedding for i in ids]).astype(np.float32)
        return ids, e

    def sim_matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (sorted ids, S) where S[i, j] = cosine(item_i, item_j)."""
        ids, e = self.matrix()
        if e.shape[0] == 0:
            return ids, np.zeros((0, 0), dtype=np.float32)
        return ids, (e @ e.T).astype(np.float32)
