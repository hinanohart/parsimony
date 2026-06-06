"""ExplainTrace: the auditable record of every decision.

Each decision is rendered as a JSON-serializable dict whose ``contributing_terms``
sum exactly to the utility, plus a one-line counterfactual ("it would have been
kept if ..."). This is the audit trail an LLM judge cannot produce.
"""

from __future__ import annotations

from typing import Any

from .config import PolicyConfig
from .types import AdmitDecision, CompressDecision, EvictDecision

SCHEMA_VERSION = "1"


def _base(item_id: str, decision_value: str, cfg: PolicyConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "decision": decision_value,
        "policy_seed": cfg.seed,
        "config_digest": cfg.digest(),
    }


def explain_admission(d: AdmitDecision, cfg: PolicyConfig) -> dict[str, Any]:
    out = _base(d.item_id, d.decision.value, cfg)
    out["objective"] = {
        "utility": d.trace.get("utility"),
        "contributing_terms": d.trace.get("contributing_terms", []),
    }
    out["admission"] = {
        "admitted": d.admitted,
        "victim_id": d.trace.get("victim_id"),
        "victim_utility": d.trace.get("victim_utility"),
    }
    out["dedup"] = {"merged_into": d.merged_into, "sim": d.trace.get("dedup", {}).get("sim")}
    out["tie_break"] = d.trace.get(
        "tie_break", {"applied": False, "rule": "salience desc, then created_at asc, then id asc"}
    )
    if d.merged_into:
        out["counterfactual"] = (
            f"would stay a separate memory if its cosine to {d.merged_into} "
            f"were below {cfg.dedup_threshold}"
        )
    elif d.trace.get("mode") == "eviction":
        out["admission"]["evicted"] = d.trace.get("evicted", [])
        out["counterfactual"] = (
            "would be kept if it added more facility-location coverage than the "
            "least-covering resident"
            if not d.admitted
            else "admitted; over capacity, the least-covering item was evicted instead"
        )
    elif d.decision.value == "reject":
        out["counterfactual"] = (
            f"would be admitted if its utility exceeded victim "
            f"{d.trace.get('victim_id')} ({d.trace.get('victim_utility')})"
        )
    else:
        vid = d.trace.get("victim_id")
        out["counterfactual"] = (
            f"would be rejected if its utility fell below victim {vid}"
            if vid
            else "admitted unconditionally (pool below capacity)"
        )
    out["reason"] = d.reason
    return out


def explain_eviction(d: EvictDecision, cfg: PolicyConfig) -> dict[str, Any]:
    out = _base(d.item_id, d.decision.value, cfg)
    out["eviction"] = {
        "removal_loss": d.removal_loss,
        "covered_count": d.covered_count,
        "nearest_after_id": d.nearest_after_id,
        "nearest_after_sim": d.trace.get("nearest_after_sim"),
        "displaced_by": d.trace.get("displaced_by"),
    }
    out["counterfactual"] = (
        f"would be kept if removing it cost more coverage than removing some other item "
        f"(it is represented by {d.nearest_after_id} after removal)"
    )
    out["reason"] = d.reason
    return out


def explain_compression(d: CompressDecision, cfg: PolicyConfig) -> dict[str, Any]:
    out = _base(d.item_id, d.decision.value, cfg)
    out["compression"] = {
        "chosen_level": d.chosen_level,
        "lambda": d.lam,
        "distortion": d.distortion,
        "rate_saved": d.rate_saved,
        "rd_curve": d.trace.get("rd_curve", []),
    }
    out["counterfactual"] = (
        f"would be compressed less aggressively if lambda (now {d.lam:.3f}) were larger "
        f"(raise salience or content weight)"
    )
    out["reason"] = d.reason
    return out


def explain_decision(d: object, cfg: PolicyConfig) -> dict[str, Any]:
    if isinstance(d, AdmitDecision):
        return explain_admission(d, cfg)
    if isinstance(d, EvictDecision):
        return explain_eviction(d, cfg)
    if isinstance(d, CompressDecision):
        return explain_compression(d, cfg)
    raise TypeError(f"not a decision: {type(d)!r}")
