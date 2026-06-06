"""Core data structures: memory items, access events, and decision records.

All structures are frozen/slots dataclasses. Embeddings are excluded from
equality/repr so items remain hashable and comparable by identity fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np


class DecisionType(str, Enum):
    """The kind of decision the policy made about an item."""

    ADMIT = "admit"
    REJECT = "reject"
    KEEP = "keep"
    MERGE = "merge"
    EVICT = "evict"
    COMPRESS = "compress"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """A single memory under management.

    ``embedding`` must be L2-normalized so that ``e_a @ e_b`` is cosine
    similarity. Frequency is *not* stored here; it lives in the Count-Min
    sketch so it survives item deletion.
    """

    id: str
    text: str
    embedding: np.ndarray = field(repr=False, compare=False)
    created_at: float = 0.0
    last_access: float = 0.0
    salience: float = 1.0
    source: str = ""
    tokens: int = 0
    compression_level: int = 0
    content_hash: str = ""
    weight: float = 1.0  # coverage mass this item stands for (grows when it absorbs evicted items)

    def evolve(self, **changes: Any) -> MemoryItem:
        """Return a copy with ``changes`` applied (frozen-safe)."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class AccessEvent:
    """A write or read touch in an access trace."""

    id: str
    kind: str = "write"  # "write" | "read"
    timestamp: float = 0.0
    text: str | None = None
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AdmitDecision:
    item_id: str
    decision: DecisionType
    admitted: bool
    reason: str = ""
    merged_into: str | None = None
    trace: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class EvictDecision:
    item_id: str
    decision: DecisionType = DecisionType.EVICT
    removal_loss: float = 0.0
    covered_count: int = 0
    nearest_after_id: str | None = None
    reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CompressDecision:
    item_id: str
    decision: DecisionType = DecisionType.KEEP
    chosen_level: int = 0
    lam: float = 0.0
    distortion: float = 0.0
    rate_saved: int = 0
    reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(slots=True)
class StepReport:
    """Outcome of a periodic maintenance sweep."""

    merged: list[AdmitDecision] = field(default_factory=list)
    evicted: list[EvictDecision] = field(default_factory=list)
    compressed: list[CompressDecision] = field(default_factory=list)
