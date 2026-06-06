"""(C) Per-memory rate-distortion compression.

For each item we pick the compression level that minimizes ``rate + lambda *
distortion``, where ``lambda`` rises with salience and content type (numbers and
proper nouns resist compression). The drop level's distortion is discounted by
how well the rest of the pool already covers the item, so the three operators
share information: an item others can reconstruct is cheaper to drop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .compress_ops import extractive
from .config import PolicyConfig
from .store import normalize
from .types import CompressDecision, DecisionType, MemoryItem

EmbedFn = Callable[[str], np.ndarray]


def _type_weight(text: str, cfg: PolicyConfig) -> float:
    return cfg.numeric_type_weight if any(ch.isdigit() for ch in text) else cfg.default_type_weight


def _lexical_distortion(verbatim: str, compressed: str) -> float:
    weights: dict[str, float] = {}
    for tok in verbatim.split():
        weights[tok] = extractive.token_weight(tok)
    total = sum(weights.values())
    if total == 0:
        return 0.0
    kept = set(compressed.split())
    retained = sum(w for t, w in weights.items() if t in kept)
    return max(0.0, min(1.0, 1.0 - retained / total))


def compress_item(
    item: MemoryItem,
    cfg: PolicyConfig,
    *,
    embed_fn: EmbedFn | None = None,
    coverage_residual: float | None = None,
) -> CompressDecision:
    """Choose the rate-distortion-optimal compression level for one item.

    ``coverage_residual`` is how well the rest of the pool already covers this
    item (computed from its verbatim embedding); it discounts the drop level's
    distortion so an item others can reconstruct is cheaper to drop.
    """
    verbatim_tokens = max(1, extractive.token_count(item.text))
    lam = cfg.lambda0 * item.salience * _type_weight(item.text, cfg)
    levels = cfg.compression_levels

    curve: list[dict[str, Any]] = []
    for level in levels:
        text_l = extractive.render(item.text, level, levels, cfg.skeleton_keep_ratio)
        rate_tokens = extractive.token_count(text_l)
        rate = rate_tokens / verbatim_tokens
        if level >= levels[-1]:
            distortion = (1.0 - coverage_residual) if coverage_residual is not None else 1.0
            distortion = max(0.0, min(1.0, distortion))
        elif level <= levels[0]:
            distortion = 0.0
        elif embed_fn is not None:
            emb = normalize(embed_fn(text_l))
            distortion = max(0.0, min(1.0, 1.0 - float(np.dot(emb, item.embedding))))
        else:
            distortion = _lexical_distortion(item.text, text_l)
        cost = rate + lam * distortion
        curve.append(
            {
                "level": int(level),
                "rate": rate,
                "distortion": distortion,
                "cost": cost,
                "tokens": rate_tokens,
            }
        )

    best = min(curve, key=lambda c: (c["cost"], c["level"]))
    chosen = int(best["level"])
    if chosen <= levels[0]:
        dtype = DecisionType.KEEP
    elif chosen >= levels[-1]:
        dtype = DecisionType.DROP
    else:
        dtype = DecisionType.COMPRESS
    rate_saved = verbatim_tokens - int(best["tokens"])
    return CompressDecision(
        item_id=item.id,
        decision=dtype,
        chosen_level=chosen,
        lam=lam,
        distortion=float(best["distortion"]),
        rate_saved=rate_saved,
        reason=(
            f"level {chosen} ({dtype.value}); lambda {lam:.3f}; "
            f"rate {best['rate']:.2f}, distortion {best['distortion']:.3f}"
        ),
        trace={
            "rd_curve": curve,
            "lambda": lam,
            "type_weight": _type_weight(item.text, cfg),
            "verbatim_tokens": verbatim_tokens,
        },
    )


def apply_compression(item: MemoryItem, cfg: PolicyConfig, level: int) -> MemoryItem:
    """Return ``item`` rendered at ``level`` (records the new level and token count)."""
    text_l = extractive.render(item.text, level, cfg.compression_levels, cfg.skeleton_keep_ratio)
    return item.evolve(
        text=text_l,
        tokens=extractive.token_count(text_l),
        compression_level=level,
    )
