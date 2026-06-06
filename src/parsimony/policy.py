"""The Parsimony facade: one object that runs all four operators on a pool.

Write path uses TinyLFU admission + semantic dedup; periodic ``step`` uses
facility-location eviction and rate-distortion compression. Every decision is
recorded as an ExplainTrace retrievable via ``explain(item_id)``. No method ever
calls a language model.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .admission import admit
from .compression import apply_compression, compress_item
from .config import PolicyConfig
from .dedup import find_duplicate, merge
from .eviction import evict as _evict_op
from .explain import explain_decision
from .geometry import best_other_coverage
from .sketch import CountMinSketch4Bit
from .store import MemoryPool
from .types import (
    AdmitDecision,
    CompressDecision,
    DecisionType,
    EvictDecision,
    MemoryItem,
    StepReport,
)


class EmbeddingBackend(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class Parsimony:
    """A deterministic forgetting policy over a bounded memory pool."""

    def __init__(
        self,
        capacity: int,
        *,
        config: PolicyConfig | None = None,
        embedder: EmbeddingBackend | None = None,
        seed: int = 0,
    ):
        self.cfg = config if config is not None else PolicyConfig(capacity=capacity, seed=seed)
        self.cms = CountMinSketch4Bit(
            capacity=self.cfg.capacity,
            depth=self.cfg.cms_depth,
            sample_multiplier=self.cfg.sample_multiplier,
            seed=self.cfg.seed,
            doorkeeper_bits_per_item=self.cfg.doorkeeper_bits_per_item,
            doorkeeper_hashes=self.cfg.doorkeeper_hashes,
        )
        self.pool: MemoryPool | None = None
        self.embedder = embedder
        self.llm_calls = 0
        self._explain: dict[str, dict[str, Any]] = {}

    # --- internals ---
    def _ensure_pool(self, dim: int) -> MemoryPool:
        if self.pool is None:
            self.pool = MemoryPool(dim=dim)
        return self.pool

    def _record(self, decision: object, item_id: str) -> None:
        self._explain[item_id] = explain_decision(decision, self.cfg)

    def _embed_fn(self):  # type: ignore[no-untyped-def]
        return None if self.embedder is None else self.embedder.encode

    # --- write path ---
    def on_write(self, item: MemoryItem) -> AdmitDecision:
        """Admit, reject, or merge a new memory."""
        self.cms.increment(item.id)
        pool = self._ensure_pool(item.embedding.shape[0])

        match, sim = find_duplicate(item, pool, self.cfg)
        if match is not None:
            existing = pool.items[match]
            pool.items[match] = merge(existing, item)
            self.cms.increment(match)
            decision = AdmitDecision(
                item.id,
                DecisionType.MERGE,
                admitted=False,
                reason=f"merged into {match} (cosine {sim:.3f} >= {self.cfg.dedup_threshold})",
                merged_into=match,
                trace={"dedup": {"merged_into": match, "sim": sim}},
            )
            self._record(decision, item.id)
            return decision

        decision = admit(item, pool, self.cms, self.cfg)
        if decision.admitted:
            if len(pool) >= self.cfg.capacity:
                victim_id = decision.trace.get("victim_id")
                if victim_id is not None and victim_id in pool:
                    pool.remove(victim_id)
                    ev = EvictDecision(
                        victim_id,
                        decision=DecisionType.EVICT,
                        reason=f"displaced by {item.id} (TinyLFU admission)",
                        trace={"displaced_by": item.id},
                    )
                    self._record(ev, victim_id)
            pool.add(item)
        self._record(decision, item.id)
        return decision

    def touch(self, item_id: str) -> None:
        """Register a read access (reinforces frequency)."""
        self.cms.increment(item_id)
        if self.pool is not None and item_id in self.pool:
            it = self.pool.items[item_id]
            self.pool.items[item_id] = it.evolve(last_access=it.last_access + 1.0)

    # --- maintenance path ---
    def evict(self, n: int | None = None) -> list[EvictDecision]:
        """Facility-location eviction (default: down to capacity)."""
        if self.pool is None:
            return []
        decisions = _evict_op(self.pool, self.cfg, n)
        for d in decisions:
            self._record(d, d.item_id)
        return decisions

    def compress(self, item_id: str) -> CompressDecision:
        """Rate-distortion compression of a single item (applied in place)."""
        if self.pool is None or item_id not in self.pool:
            raise KeyError(item_id)
        item = self.pool.items[item_id]
        ids, sim = self.pool.sim_matrix()
        cov = best_other_coverage(ids, sim).get(item_id, 0.0)
        decision = compress_item(item, self.cfg, embed_fn=self._embed_fn(), coverage_residual=cov)
        self.pool.items[item_id] = apply_compression(item, self.cfg, decision.chosen_level)
        self._record(decision, item_id)
        return decision

    def _dedup_pass(self) -> list[AdmitDecision]:
        assert self.pool is not None
        merged: list[AdmitDecision] = []
        removed: set[str] = set()
        ids = self.pool.ids()
        for a in ids:
            if a in removed or a not in self.pool:
                continue
            for b in ids:
                if b <= a or b in removed or b not in self.pool:
                    continue
                ia, ib = self.pool.items[a], self.pool.items[b]
                s = float(ia.embedding @ ib.embedding)
                if s >= self.cfg.dedup_threshold:
                    self.pool.items[a] = merge(ia, ib)
                    self.pool.remove(b)
                    removed.add(b)
                    d = AdmitDecision(
                        b,
                        DecisionType.MERGE,
                        admitted=False,
                        reason=f"step-merged into {a} (cosine {s:.3f})",
                        merged_into=a,
                        trace={"dedup": {"merged_into": a, "sim": s}},
                    )
                    merged.append(d)
                    self._record(d, b)
        return merged

    def step(self) -> StepReport:
        """A deterministic maintenance sweep: dedup -> evict -> compress."""
        if self.pool is None:
            return StepReport()
        merged = self._dedup_pass()
        evicted = self.evict()
        compressed: list[CompressDecision] = []
        for item_id in self.pool.ids():
            compressed.append(self.compress(item_id))
        return StepReport(merged=merged, evicted=evicted, compressed=compressed)

    # --- inspection ---
    def explain(self, item_id: str) -> dict[str, Any]:
        """The ExplainTrace of the last decision about ``item_id`` ({} if none)."""
        return self._explain.get(item_id, {})

    def snapshot(self) -> list[MemoryItem]:
        """All current items in deterministic (id-sorted) order."""
        if self.pool is None:
            return []
        return [self.pool.items[i] for i in self.pool.ids()]

    def __len__(self) -> int:
        return 0 if self.pool is None else len(self.pool)
