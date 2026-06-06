"""Run all policies on a corpus and score retention against the Bélády oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.embedding import HashingEmbedder
from ..loaders.longmemeval import load_longmemeval
from ..metrics import coverage_of, gold_hit_all, gold_hit_any, redundancy
from ..oracle.belady import opt_online_coverage, opt_static_coverage
from ..policies import REGISTRY
from ..trace.adapter import generate_synthetic_trace, question_to_items


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _vs(value: float, source: str, note: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"value": round(value, 4), "source": source}
    if note:
        out["note"] = note
    return out


def run_bench(
    *,
    data_path: str | None = None,
    n: int = 120,
    capacity: int = 10,
    ilp_sample: int = 40,
    seed: int = 0,
    embed_dim: int = 256,
    ilp_time_limit: float = 5.0,
) -> dict[str, Any]:
    """Return a results dict with provenance, headline ratios, and per-policy stats."""
    embedder = HashingEmbedder(dim=embed_dim, seed=seed)

    if data_path and Path(data_path).exists():
        questions, sha = load_longmemeval(data_path)
        mode = "real-longmemeval"
        dataset = Path(data_path).stem
        source = f"{dataset}@{sha[:8]}"
        traces = [question_to_items(q, embedder) for q in questions[:n]]
    else:
        mode = "synthetic-honest"
        dataset = "synthetic"
        sha = "n/a"
        source = "synthetic-honest"
        traces = [
            generate_synthetic_trace(n_items=60, n_clusters=8, dim=embed_dim, seed=seed + i)
            for i in range(n)
        ]

    names = list(REGISTRY)
    agg: dict[str, dict[str, list[float]]] = {
        p: {
            "coverage": [],
            "hit_any": [],
            "hit_all": [],
            "llm_calls": [],
            "merges": [],
            "redund": [],
        }
        for p in names
    }
    ratio_online: list[float] = []
    ratio_ilp: list[float] = []
    ilp_optimal = 0
    ilp_done = 0

    for qi, (items, gold) in enumerate(traces):
        if not items:
            continue
        ids = [it.id for it in items]
        emb = np.stack([it.embedding for it in items]).astype(np.float64)
        sim = np.clip(emb @ emb.T, 0.0, None)

        for name in names:
            res = REGISTRY[name](items, capacity, seed=seed)
            agg[name]["coverage"].append(coverage_of(res.kept, ids, sim))
            agg[name]["hit_any"].append(gold_hit_any(res.kept, gold))
            agg[name]["hit_all"].append(gold_hit_all(res.kept, gold))
            agg[name]["llm_calls"].append(float(res.llm_calls))
            agg[name]["merges"].append(float(res.merges))
            agg[name]["redund"].append(redundancy(res.kept, ids, sim))

        _, cov_online = opt_online_coverage(ids, sim, capacity)
        if cov_online > 0:
            ratio_online.append(agg["parsimony"]["coverage"][-1] / cov_online)

        if qi < ilp_sample:
            _, cov_ilp, status = opt_static_coverage(ids, sim, capacity, time_limit=ilp_time_limit)
            ilp_done += 1
            if status == "Optimal":
                ilp_optimal += 1
            if cov_ilp > 0:
                ratio_ilp.append(agg["parsimony"]["coverage"][-1] / cov_ilp)

    policies_out: dict[str, Any] = {}
    for name in names:
        a = agg[name]
        policies_out[name] = {
            "coverage": _vs(_mean(a["coverage"]), source),
            "gold_hit_any": _vs(_mean(a["hit_any"]), source),
            "gold_hit_all": _vs(_mean(a["hit_all"]), source),
            "intra_set_redundancy": _vs(_mean(a["redund"]), source),
            "avg_llm_calls": _vs(_mean(a["llm_calls"]), source),
            "avg_merges": _vs(_mean(a["merges"]), source),
        }

    return {
        "provenance": {
            "mode": mode,
            "dataset": dataset,
            "sha256": sha,
            "source": source,
            "n_questions": len([t for t in traces if t[0]]),
            "capacity": capacity,
            "embedder": f"hashing-dim{embed_dim}",
            "seed": seed,
            "ilp_solved": ilp_done,
            "ilp_optimal": ilp_optimal,
        },
        "headline": {
            "competitive_ratio_vs_belady_ilp": _vs(
                _mean(ratio_ilp),
                source,
                note=(
                    "parsimony online coverage / OPT-static ILP coverage; "
                    f"ILP on {ilp_done} questions, {ilp_optimal} proven optimal"
                ),
            ),
            "competitive_ratio_vs_belady_greedy": _vs(
                _mean(ratio_online),
                source,
                note="parsimony online coverage / clairvoyant greedy coverage (all questions)",
            ),
        },
        "policies": policies_out,
    }
