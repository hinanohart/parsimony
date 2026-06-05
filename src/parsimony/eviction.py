"""(B) Facility-location submodular eviction.

To shrink a full pool we repeatedly drop the item whose removal least reduces
how well the *kept* set still represents the *whole* set (a facility-location
coverage objective). Coverage uses clipped cosine similarity; each universe item
is represented by its best-matching kept item.

The greedy maintains a per-row top-2 (best and runner-up kept representative) and
only recomputes the rows whose representative was just removed, so it does not
rescan the full matrix every step.
"""

from __future__ import annotations

import numpy as np

from .config import PolicyConfig
from .store import MemoryPool
from .types import DecisionType, EvictDecision


class _Coverage:
    """Incremental top-2 representative tracker over a fixed universe."""

    def __init__(self, sim_clipped: np.ndarray):
        self.sim = sim_clipped
        self.n = sim_clipped.shape[0]
        self.kept = [True] * self.n
        self.best = np.zeros(self.n)
        self.second = np.zeros(self.n)
        self.argbest = np.full(self.n, -1, dtype=int)
        for u in range(self.n):
            self._recompute_row(u)

    def _recompute_row(self, u: int) -> None:
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
        """Coverage that each live facility uniquely provides (its marginal loss)."""
        loss = {j: 0.0 for j in self.live()}
        for u in range(self.n):
            j = int(self.argbest[u])
            if j >= 0 and self.kept[j]:
                loss[j] += float(self.best[u] - self.second[u])
        return loss

    def remove(self, j: int) -> None:
        self.kept[j] = False
        for u in range(self.n):
            if int(self.argbest[u]) == j:
                self._recompute_row(u)


def _evict_order(ids: list[str], sim_clipped: np.ndarray, n_remove: int) -> list[dict]:
    cov = _Coverage(sim_clipped)
    out: list[dict] = []
    for _ in range(n_remove):
        live = cov.live()
        if len(live) <= 1:
            break
        loss = cov.removal_loss()
        victim = min(live, key=lambda j: (loss[j], ids[j]))
        # nearest surviving representative of the victim (best other kept item)
        others = [(float(sim_clipped[victim, j]), ids[j]) for j in live if j != victim]
        nearest = max(others, default=(0.0, None))
        covered = sum(1 for j in live if j != victim and sim_clipped[victim, j] > 0.0)
        out.append(
            {
                "id": ids[victim],
                "removal_loss": float(loss[victim]),
                "nearest_after_id": nearest[1],
                "nearest_after_sim": nearest[0],
                "covered_count": covered,
            }
        )
        cov.remove(victim)
    return out


def evict(pool: MemoryPool, cfg: PolicyConfig, n: int | None = None) -> list[EvictDecision]:
    """Evict items (mutating the pool). Default brings the pool down to capacity."""
    target = (len(pool) - cfg.capacity) if n is None else n
    if target <= 0 or len(pool) == 0:
        return []
    target = min(target, len(pool) - 1) if n is None else min(target, len(pool))
    ids, sim = pool.sim_matrix()
    sim_clipped = np.clip(np.asarray(sim, dtype=np.float64), 0.0, None)
    order = _evict_order(ids, sim_clipped, target)
    decisions: list[EvictDecision] = []
    for rec in order:
        pool.remove(rec["id"])
        decisions.append(
            EvictDecision(
                item_id=rec["id"],
                decision=DecisionType.EVICT,
                removal_loss=rec["removal_loss"],
                covered_count=rec["covered_count"],
                nearest_after_id=rec["nearest_after_id"],
                reason=(
                    f"least coverage loss {rec['removal_loss']:.5f}; "
                    f"represented by {rec['nearest_after_id']} "
                    f"(sim {rec['nearest_after_sim']:.3f})"
                ),
                trace={
                    "removal_loss": rec["removal_loss"],
                    "nearest_after_id": rec["nearest_after_id"],
                    "nearest_after_sim": rec["nearest_after_sim"],
                    "covered_count": rec["covered_count"],
                },
            )
        )
    return decisions
