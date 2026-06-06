"""Knapsack-coverage Bélády: the offline-optimal retained set.

Classic furthest-in-future eviction is not optimal here: items have variable
"size" (compressibility) and many-to-many semantic overlap, so the right offline
oracle keeps the size-k subset that maximizes facility-location coverage of the
whole stream. This is a *non-causal lower bound on regret* — a yardstick, not an
implementable policy.

* ``opt_static_coverage`` — exact ILP (pulp/CBC), small k.
* ``opt_online_coverage`` — clairvoyant lazy-greedy ((1 - 1/e) of optimal).
"""

from __future__ import annotations

import numpy as np

from ..geometry import facility_location_value


def opt_online_coverage(ids: list[str], sim_clipped: np.ndarray, k: int) -> tuple[list[int], float]:
    """Clairvoyant greedy: pick k items maximizing coverage of the whole set."""
    n = len(ids)
    if n == 0:
        return [], 0.0
    if k >= n:
        return list(range(n)), facility_location_value(ids, list(range(n)), sim_clipped)
    selected: list[int] = []
    best_cov = np.zeros(n)
    remaining = set(range(n))
    for _ in range(k):
        best_j = -1
        best_gain = -1.0
        for j in sorted(remaining):
            gain = float(np.maximum(best_cov, sim_clipped[:, j]).sum() - best_cov.sum())
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
        remaining.discard(best_j)
        best_cov = np.maximum(best_cov, sim_clipped[:, best_j])
    return selected, float(best_cov.sum() / n)


def opt_static_coverage(
    ids: list[str], sim_clipped: np.ndarray, k: int, time_limit: float = 5.0
) -> tuple[list[int], float, str]:
    """Exact max-coverage subset of size k via ILP. Falls back to greedy if pulp absent."""
    n = len(ids)
    if n == 0:
        return [], 0.0, "empty"
    if k >= n:
        return list(range(n)), facility_location_value(ids, list(range(n)), sim_clipped), "trivial"
    try:
        import pulp
    except ImportError:
        sel, cov = opt_online_coverage(ids, sim_clipped, k)
        return sel, cov, "greedy-fallback"

    prob = pulp.LpProblem("max_facility_location", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n)]
    x = {
        (u, j): pulp.LpVariable(f"x_{u}_{j}", lowBound=0, upBound=1)
        for u in range(n)
        for j in range(n)
        if sim_clipped[u, j] > 0.0
    }
    prob += pulp.lpSum(float(sim_clipped[u, j]) * x[(u, j)] for (u, j) in x)
    prob += pulp.lpSum(y) == k
    for u in range(n):
        terms = [x[(u, j)] for j in range(n) if (u, j) in x]
        if terms:
            prob += pulp.lpSum(terms) <= 1
    for u, j in x:
        prob += x[(u, j)] <= y[j]
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    sel = [j for j in range(n) if y[j].value() and y[j].value() > 0.5]
    if not sel:
        sel, cov = opt_online_coverage(ids, sim_clipped, k)
        return sel, cov, "greedy-fallback-no-solution"
    cov = facility_location_value(ids, sel, sim_clipped)
    return sel, cov, pulp.LpStatus[prob.status]
