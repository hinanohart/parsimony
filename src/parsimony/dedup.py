"""(D) Write-time semantic near-duplicate merge.

mem0 deduplicates by md5 of the exact text. parsimony extends that to the
embedding space: if a new item is within ``dedup_threshold`` cosine of an
existing one, they are merged deterministically. This catches paraphrases and
contradictory updates that a hash never would.
"""

from __future__ import annotations

import numpy as np

from .config import PolicyConfig
from .objective import prefers
from .store import MemoryPool
from .types import MemoryItem


def find_duplicate(
    candidate: MemoryItem, pool: MemoryPool, cfg: PolicyConfig
) -> tuple[str | None, float]:
    """Return (best_match_id or None, best_similarity)."""
    ids, e = pool.matrix()
    if e.shape[0] == 0:
        return None, 0.0
    sims = e @ candidate.embedding
    j = int(np.argmax(sims))
    best = float(sims[j])
    if best >= cfg.dedup_threshold:
        return ids[j], best
    return None, best


def _merge_source(a: str, b: str) -> str:
    parts = sorted({p for p in (a, b) if p})
    return "+".join(parts)


def merge(existing: MemoryItem, candidate: MemoryItem) -> MemoryItem:
    """Deterministically fold ``candidate`` into ``existing`` (keeps existing.id)."""
    winner = candidate if prefers(candidate, existing) else existing
    return existing.evolve(
        text=winner.text,
        embedding=winner.embedding,
        salience=max(existing.salience, candidate.salience),
        created_at=min(existing.created_at, candidate.created_at),
        last_access=max(existing.last_access, candidate.last_access),
        source=_merge_source(existing.source, candidate.source),
        tokens=winner.tokens,
        content_hash=winner.content_hash,
    )
