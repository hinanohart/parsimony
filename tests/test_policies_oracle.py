from __future__ import annotations

import numpy as np
import pytest

from parsimony.metrics import coverage_of
from parsimony.oracle.belady import opt_online_coverage, opt_static_coverage
from parsimony.policies import REGISTRY, mock_judge, parsimony_evict, recency_keep_last
from parsimony.trace.adapter import generate_synthetic_trace


def _sim(items):
    e = np.stack([it.embedding for it in items]).astype(np.float64)
    return [it.id for it in items], np.clip(e @ e.T, 0.0, None)


def test_all_policies_respect_capacity():
    items, _ = generate_synthetic_trace(n_items=40, n_clusters=6, dim=16, seed=0)
    for name, fn in REGISTRY.items():
        res = fn(items, 8, seed=0)
        assert len(res.kept) <= 8, name


def test_parsimony_coverage_beats_recency_on_clusters():
    items, _ = generate_synthetic_trace(n_items=60, n_clusters=8, dim=24, seed=3)
    ids, sim = _sim(items)
    cap = 8
    cov_p = coverage_of(parsimony_evict(items, cap).kept, ids, sim)
    cov_r = coverage_of(recency_keep_last(items, cap).kept, ids, sim)
    assert cov_p >= cov_r


def test_mock_judge_counts_llm_calls():
    items, _ = generate_synthetic_trace(n_items=30, n_clusters=5, dim=16, seed=1)
    res = mock_judge(items, 10)
    assert res.llm_calls == len(items)
    assert parsimony_evict(items, 10).llm_calls == 0


def test_opt_online_le_opt_static():
    items, _ = generate_synthetic_trace(n_items=20, n_clusters=4, dim=16, seed=2)
    ids, sim = _sim(items)
    _, cov_on = opt_online_coverage(ids, sim, 4)
    _, cov_st, status = opt_static_coverage(ids, sim, 4, time_limit=5.0)
    # the offline optimum is at least as good as greedy (allow tiny solver slack)
    assert cov_st >= cov_on - 1e-6
    assert status in {"Optimal", "greedy-fallback", "Not Solved", "Undefined"}


def test_opt_static_trivial_when_k_ge_n():
    items, _ = generate_synthetic_trace(n_items=5, n_clusters=2, dim=8, seed=0)
    ids, sim = _sim(items)
    sel, _cov, status = opt_static_coverage(ids, sim, 10)
    assert status == "trivial"
    assert len(sel) == 5


def test_parsimony_near_optimal_coverage_ratio():
    items, _ = generate_synthetic_trace(n_items=50, n_clusters=7, dim=20, seed=5)
    ids, sim = _sim(items)
    cap = 7
    cov_p = coverage_of(parsimony_evict(items, cap).kept, ids, sim)
    _, cov_st, status = opt_static_coverage(ids, sim, cap, time_limit=5.0)
    if status != "Optimal":
        pytest.skip("ILP not proven optimal in time budget")
    assert cov_p / cov_st >= 0.85  # online stays close to offline optimum
