"""Retention policies for benchmarking (parsimony + non-clairvoyant baselines).

Each policy consumes a chronological item stream under a capacity and returns the
final retained id set, the number of (simulated) LLM calls it made, and the
number of semantic merges it performed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import PolicyConfig
from ..dedup import find_duplicate, merge
from ..eviction import evict
from ..store import MemoryPool
from ..types import MemoryItem


@dataclass(frozen=True, slots=True)
class PolicyResult:
    kept: set[str]
    llm_calls: int = 0
    merges: int = 0


def _cfg(cfg: PolicyConfig | None, capacity: int, seed: int) -> PolicyConfig:
    return cfg if cfg is not None else PolicyConfig(capacity=capacity, seed=seed)


def parsimony_evict(
    items: list[MemoryItem], capacity: int, *, cfg: PolicyConfig | None = None, seed: int = 0
) -> PolicyResult:
    """Online facility-location eviction (no dedup): the coverage-faithful policy."""
    if not items:
        return PolicyResult(set())
    c = _cfg(cfg, capacity, seed)
    pool = MemoryPool(dim=items[0].embedding.shape[0])
    for it in items:
        pool.add(it)
        if len(pool) > capacity:
            evict(pool, c, n=len(pool) - capacity)
    return PolicyResult(set(pool.ids()))


def parsimony_full(
    items: list[MemoryItem], capacity: int, *, cfg: PolicyConfig | None = None, seed: int = 0
) -> PolicyResult:
    """Semantic dedup on write + facility-location eviction on overflow."""
    if not items:
        return PolicyResult(set())
    c = _cfg(cfg, capacity, seed)
    pool = MemoryPool(dim=items[0].embedding.shape[0])
    merges = 0
    for it in items:
        match, _sim = find_duplicate(it, pool, c)
        if match is not None:
            pool.items[match] = merge(pool.items[match], it)
            merges += 1
        else:
            pool.add(it)
            if len(pool) > capacity:
                evict(pool, c, n=len(pool) - capacity)
    return PolicyResult(set(pool.ids()), merges=merges)


def recency_keep_last(items: list[MemoryItem], capacity: int, **_: object) -> PolicyResult:
    return PolicyResult({it.id for it in items[-capacity:]})


def oldest_keep_first(items: list[MemoryItem], capacity: int, **_: object) -> PolicyResult:
    return PolicyResult({it.id for it in items[:capacity]})


def random_keep(
    items: list[MemoryItem], capacity: int, *, seed: int = 0, **_: object
) -> PolicyResult:
    n = len(items)
    if n == 0:
        return PolicyResult(set())
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(capacity, n), replace=False)
    return PolicyResult({items[int(i)].id for i in idx})


def mock_judge(items: list[MemoryItem], capacity: int, **_: object) -> PolicyResult:
    """An LLM-judge stand-in: keeps the k 'most informative' (longest) items.

    Charges one LLM call per item, so its cost is visible next to parsimony's zero.
    """
    scored = sorted(items, key=lambda it: (-it.tokens, it.id))[:capacity]
    return PolicyResult({it.id for it in scored}, llm_calls=len(items))


Policy = Callable[..., PolicyResult]

REGISTRY: dict[str, Policy] = {
    "parsimony": parsimony_evict,
    "parsimony+dedup": parsimony_full,
    "recency": recency_keep_last,
    "oldest": oldest_keep_first,
    "random": random_keep,
    "mock_judge": mock_judge,
}
