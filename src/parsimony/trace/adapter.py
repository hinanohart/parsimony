"""Turn corpora (or synthetic generators) into ordered MemoryItem write traces."""

from __future__ import annotations

import numpy as np

from ..adapters.embedding import HashingEmbedder
from ..loaders.longmemeval import LMEQuestion, date_key
from ..store import normalize
from ..types import MemoryItem


def question_to_items(
    q: LMEQuestion, embedder: HashingEmbedder
) -> tuple[list[MemoryItem], set[str]]:
    """Sessions as items in chronological write order; gold = answer sessions."""
    order = sorted(
        range(len(q.sessions)),
        key=lambda i: date_key(q.sessions[i][1], i),
    )
    items: list[MemoryItem] = []
    for rank, i in enumerate(order):
        sid, _date, text = q.sessions[i]
        items.append(
            MemoryItem(
                id=sid,
                text=text,
                embedding=embedder.encode(text),
                created_at=float(rank),
                last_access=float(rank),
                salience=1.0,
                tokens=len(text.split()),
            )
        )
    return items, set(q.gold_session_ids)


def generate_synthetic_trace(
    n_items: int = 60,
    n_clusters: int = 8,
    dim: int = 32,
    seed: int = 0,
) -> tuple[list[MemoryItem], set[str]]:
    """A Zipf-popularity, clustered synthetic trace (honest stand-in for a real corpus).

    Each item belongs to a semantic cluster (a random center plus noise). Cluster
    popularity is Zipf-distributed; the single rarest cluster's first item is the
    "gold" memory a recency policy would discard but a coverage policy keeps.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    pop = 1.0 / np.arange(1, n_clusters + 1)
    pop = pop / pop.sum()
    items: list[MemoryItem] = []
    rarest = n_clusters - 1
    gold: set[str] = set()
    first_rarest_seen = False
    for t in range(n_items):
        c = int(rng.choice(n_clusters, p=pop))
        vec = centers[c] + 0.15 * rng.standard_normal(dim).astype(np.float32)
        item_id = f"s{t:04d}_c{c}"
        items.append(
            MemoryItem(
                id=item_id,
                text=f"synthetic memory {t} cluster {c}",
                embedding=normalize(vec),
                created_at=float(t),
                last_access=float(t),
                salience=1.0,
                tokens=5,
            )
        )
        if c == rarest and not first_rarest_seen:
            gold.add(item_id)
            first_rarest_seen = True
    if not gold and items:
        gold.add(items[0].id)
    return items, gold
