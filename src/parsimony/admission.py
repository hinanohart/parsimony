"""(A) TinyLFU admission.

When the pool is full, a new item is admitted only if its keep-utility beats the
current weakest resident's (the TinyLFU "is the newcomer hotter than the
victim?" test, generalized to the shared utility). Below capacity, admit freely.
"""

from __future__ import annotations

import numpy as np

from .config import PolicyConfig
from .geometry import removal_losses
from .objective import prefers, utility, utility_terms
from .sketch import CountMinSketch4Bit
from .store import MemoryPool
from .types import AdmitDecision, DecisionType, MemoryItem


def _candidate_removal_loss(candidate: MemoryItem, pool: MemoryPool) -> float:
    ids, e = pool.matrix()
    if e.shape[0] == 0:
        return 1.0
    sims = np.clip(e @ candidate.embedding, 0.0, None)
    m2 = float(sims.max())
    return (1.0 - m2) / (len(ids) + 1)


def _resident_removal_losses(pool: MemoryPool, cfg: PolicyConfig) -> dict[str, float]:
    if cfg.w_cover == 0.0:
        return {i: 0.0 for i in pool.ids()}
    ids, sim = pool.sim_matrix()
    return removal_losses(ids, sim)


def find_victim(pool: MemoryPool, cms: CountMinSketch4Bit, cfg: PolicyConfig) -> str | None:
    """The resident with the lowest keep-utility (deterministic tie-break)."""
    ids = pool.ids()
    if not ids:
        return None
    rl = _resident_removal_losses(pool, cfg)

    def key(i: str) -> tuple[float, float, float, str]:
        it = pool.items[i]
        u = utility(
            cfg, freq_est=cms.estimate(i), removal_loss=rl.get(i, 0.0), salience=it.salience
        )
        # min utility, then weakest tie-break (low salience, newer, larger id)
        return (u, it.salience, -it.created_at, i)

    return min(ids, key=key)


def admit(
    candidate: MemoryItem,
    pool: MemoryPool,
    cms: CountMinSketch4Bit,
    cfg: PolicyConfig,
) -> AdmitDecision:
    """Decide whether ``candidate`` enters a (possibly full) pool."""
    if len(pool) < cfg.capacity:
        u_c, terms = utility_terms(
            cfg,
            freq_est=cms.estimate(candidate.id),
            removal_loss=_candidate_removal_loss(candidate, pool) if cfg.w_cover else 0.0,
            salience=candidate.salience,
        )
        return AdmitDecision(
            candidate.id,
            DecisionType.ADMIT,
            True,
            reason="pool below capacity",
            trace={"utility": u_c, "contributing_terms": terms, "capacity": cfg.capacity},
        )

    v = find_victim(pool, cms, cfg)
    assert v is not None  # pool is full -> non-empty
    rl = _resident_removal_losses(pool, cfg)
    u_c, terms_c = utility_terms(
        cfg,
        freq_est=cms.estimate(candidate.id),
        removal_loss=_candidate_removal_loss(candidate, pool) if cfg.w_cover else 0.0,
        salience=candidate.salience,
    )
    victim_item = pool.items[v]
    u_v, _ = utility_terms(
        cfg,
        freq_est=cms.estimate(v),
        removal_loss=rl.get(v, 0.0),
        salience=victim_item.salience,
    )
    admit_it = (u_c > u_v) or (u_c == u_v and prefers(candidate, victim_item))
    decision = DecisionType.ADMIT if admit_it else DecisionType.REJECT
    reason = f"utility {u_c:.4f} {'>' if admit_it else '<='} victim {v} {u_v:.4f}"
    return AdmitDecision(
        candidate.id,
        decision,
        admit_it,
        reason=reason,
        trace={
            "utility": u_c,
            "victim_id": v,
            "victim_utility": u_v,
            "contributing_terms": terms_c,
        },
    )
