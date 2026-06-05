"""Extractive (non-generative) text reduction.

Nothing here calls a model: skeletons are built by selecting whole sentences,
and the drop-stub keeps a short prefix. Same input always yields same output.
"""

from __future__ import annotations

import math
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HAS_DIGIT = re.compile(r"\d")
_PROPER = re.compile(r"^[A-Z][a-zA-Z]")


def token_count(text: str) -> int:
    return len(text.split())


def token_weight(tok: str) -> float:
    """Numeric and proper-noun-like tokens are worth more (kept preferentially)."""
    if _HAS_DIGIT.search(tok) or _PROPER.match(tok):
        return 2.0
    return 1.0


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def skeleton(text: str, keep_ratio: float) -> str:
    """Keep the highest-weight sentences (in original order)."""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    scored = []
    for i, s in enumerate(sents):
        weight = sum(token_weight(t) for t in s.split())
        position_bonus = 1.0 / (1.0 + i)
        scored.append((weight + position_bonus, i, s))
    k = max(1, math.ceil(keep_ratio * len(sents)))
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:k]
    keep_idx = sorted(i for _, i, _ in top)
    return " ".join(sents[i] for i in keep_idx)


def drop_stub(text: str, words: int = 8) -> str:
    toks = text.split()
    if len(toks) <= words:
        return text
    return " ".join(toks[:words]) + " […]"


def render(text: str, level: int, levels: tuple[int, ...], skeleton_keep_ratio: float) -> str:
    """Render ``text`` at a compression ``level`` (min=verbatim, max=drop-stub)."""
    if level <= levels[0]:
        return text
    if level >= levels[-1]:
        return drop_stub(text)
    return skeleton(text, skeleton_keep_ratio)
