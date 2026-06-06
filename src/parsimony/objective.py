"""The one objective every operator shares.

All four operators (admission, dedup, eviction, compression) read from a single
utility/cost plane so a decision can always be explained as a linear
combination of the same terms. The ``a1`` defaults (``w_freq=1``, others ``0``)
make ``utility`` exactly a normalized W-TinyLFU frequency.
"""

from __future__ import annotations

from typing import Any

from .config import PolicyConfig
from .types import MemoryItem

MAX_FREQ = 16  # 4-bit counter (15) + doorkeeper bit


def freq_norm(est: int) -> float:
    """Normalize a sketch estimate into [0, 1]."""
    return min(max(est, 0), MAX_FREQ) / MAX_FREQ


def utility(cfg: PolicyConfig, *, freq_est: int, removal_loss: float, salience: float) -> float:
    """The shared keep-value of an item: higher = more worth keeping."""
    return cfg.w_freq * freq_norm(freq_est) + cfg.w_cover * removal_loss + cfg.w_salience * salience


def utility_terms(
    cfg: PolicyConfig, *, freq_est: int, removal_loss: float, salience: float
) -> tuple[float, list[dict[str, Any]]]:
    """Return (utility, contributing_terms) where contributions sum to utility."""
    fn = freq_norm(freq_est)
    c_freq = cfg.w_freq * fn
    c_cover = cfg.w_cover * removal_loss
    c_sal = cfg.w_salience * salience
    terms: list[dict[str, Any]] = [
        {
            "term": "frequency",
            "weight": cfg.w_freq,
            "raw": float(freq_est),
            "normalized": fn,
            "contribution": c_freq,
        },
        {
            "term": "coverage_uniqueness",
            "weight": cfg.w_cover,
            "raw": float(removal_loss),
            "normalized": float(removal_loss),
            "contribution": c_cover,
        },
        {
            "term": "salience",
            "weight": cfg.w_salience,
            "raw": float(salience),
            "normalized": float(salience),
            "contribution": c_sal,
        },
    ]
    total = c_freq + c_cover + c_sal
    return total, terms


def prefers(a: MemoryItem, b: MemoryItem) -> bool:
    """Deterministic tie-break: True iff ``a`` should be kept over ``b``.

    Order: higher salience, then older (smaller created_at), then lexicographic id.
    """
    if a.salience != b.salience:
        return a.salience > b.salience
    if a.created_at != b.created_at:
        return a.created_at < b.created_at
    return a.id < b.id
