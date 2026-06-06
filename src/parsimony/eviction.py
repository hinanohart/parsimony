"""(B) Facility-location submodular eviction with ghost-mass weighting.

Each kept item carries a ``weight`` = how many original items it now stands for.
When an item is evicted its weight is handed to its nearest surviving
representative, so an online stream of evictions keeps approximating the
coverage of *everything seen* without storing the evicted embeddings (bounded
memory). At each step we drop the item whose removal costs the least
weighted coverage.
"""

from __future__ import annotations

import numpy as np

from .config import PolicyConfig
from .store import MemoryPool
from .types import DecisionType, EvictDecision


class _Coverage:
    """Weighted facility-location over the live set (universe == facilities == kept)."""

    def __init__(self, sim_clipped: np.ndarray, weights: np.ndarray):
        self.sim = sim_clipped
        self.n = sim_clipped.shape[0]
        self.w = np.asarray(weights, dtype=np.float64).copy()
        self.kept = [True] * self.n
        self.best = np.zeros(self.n)
        self.second = np.zeros(self.n)
        self.argbest = np.full(self.n, -1, dtype=int)
        for u in range(self.n):
            self._recompute_row(u)

    def _recompute_row(self, u: int) -> None:
        if not self.kept[u]:
            self.best[u] = 0.0
            self.second[u] = 0.0
            self.argbest[u] = -1
            return
        b1 = -1.0
        b2 = -1.0
        a1 = -1
        row = self.sim[u]
        for j in range(self.n):
            if not self.kept[j]:
                continue
            s = float(row[j])
            if s > b1 or (s == b1 and (a1 == -1 or j < a1)):
                b2, b1, a1 = b1, s, j
            elif s > b2:
                b2 = s
        self.best[u] = max(b1, 0.0)
        self.second[u] = max(b2, 0.0)
        self.argbest[u] = a1

    def live(self) -> list[int]:
        return [j for j in range(self.n) if self.kept[j]]

    def removal_loss(self) -> dict[int, float]:
        loss = {j: 0.0 for j in self.live()}
        for u in range(self.n):
            if not self.kept[u]:
                continue
            j = int(self.argbest[u])
            if j >= 0 and self.kept[j]:
                loss[j] += self.w[u] * float(self.best[u] - self.second[u])
        return loss

    def remove_and_transfer(self, victim: int) -> tuple[int, float]:
        """Evict ``victim``; hand its weight to the nearest survivor. Returns (rep, sim)."""
        nearest = -1
        nsim = -1.0
        for j in range(self.n):
            if j == victim or not self.kept[j]:
                continue
            s = float(self.sim[victim, j])
            if s > nsim:
                nsim = s
                nearest = j
        self.kept[victim] = False
        if nearest >= 0:
            self.w[nearest] += self.w[victim]
        for u in range(self.n):
            if int(self.argbest[u]) == victim:
                self._recompute_row(u)
        return nearest, (nsim if nearest >= 0 else 0.0)


def _evict_order(
    ids: list[str], sim_clipped: np.ndarray, weights: np.ndarray, n_remove: int
) -> tuple[list[dict], np.ndarray]:
    cov = _Coverage(sim_clipped, weights)
    out: list[dict] = []
    for _ in range(n_remove):
        live = cov.live()
        if len(live) <= 1:
            break
        loss = cov.removal_loss()
        victim = min(live, key=lambda j: (loss[j], ids[j]))
        covered = sum(1 for j in live if j != victim and sim_clipped[victim, j] > 0.0)
        victim_weight = float(cov.w[victim])
        nearest, nsim = cov.remove_and_transfer(victim)
        out.append(
            {
                "id": ids[victim],
                "removal_loss": float(loss[victim]),
                "nearest_after_id": ids[nearest] if nearest >= 0 else None,
                "nearest_after_sim": float(nsim),
                "covered_count": covered,
                "transferred_weight": victim_weight,
            }
        )
    return out, cov.w


def evict(pool: MemoryPool, cfg: PolicyConfig, n: int | None = None) -> list[EvictDecision]:
    """Evict items (mutating the pool). Default brings the pool down to capacity."""
    if len(pool) == 0:
        return []
    target = (len(pool) - cfg.capacity) if n is None else n
    target = min(max(target, 0), len(pool) - 1)
    if target <= 0:
        return []
    ids, sim = pool.sim_matrix()
    sim_clipped = np.clip(np.asarray(sim, dtype=np.float64), 0.0, None)
    weights = np.array([pool.items[i].weight for i in ids], dtype=np.float64)
    order, final_w = _evict_order(ids, sim_clipped, weights, target)
    evicted = {rec["id"] for rec in order}

    for k, item_id in enumerate(ids):
        if item_id in evicted:
            pool.remove(item_id)
        elif final_w[k] != pool.items[item_id].weight:
            pool.items[item_id] = pool.items[item_id].evolve(weight=float(final_w[k]))

    decisions: list[EvictDecision] = []
    for rec in order:
        decisions.append(
            EvictDecision(
                item_id=rec["id"],
                decision=DecisionType.EVICT,
                removal_loss=rec["removal_loss"],
                covered_count=rec["covered_count"],
                nearest_after_id=rec["nearest_after_id"],
                reason=(
                    f"least weighted coverage loss {rec['removal_loss']:.5f}; "
                    f"mass {rec['transferred_weight']:.0f} handed to {rec['nearest_after_id']} "
                    f"(sim {rec['nearest_after_sim']:.3f})"
                ),
                trace={
                    "removal_loss": rec["removal_loss"],
                    "nearest_after_id": rec["nearest_after_id"],
                    "nearest_after_sim": rec["nearest_after_sim"],
                    "covered_count": rec["covered_count"],
                    "transferred_weight": rec["transferred_weight"],
                },
            )
        )
    return decisions
