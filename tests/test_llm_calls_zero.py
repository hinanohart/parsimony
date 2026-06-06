"""The policy path must never touch an LLM (or torch). Enforced at runtime."""

from __future__ import annotations

import sys

from _helpers import make_item
from parsimony import Parsimony

_BANNED = ["torch", "openai", "anthropic", "litellm", "cohere"]


def test_core_imports_no_llm_or_torch():
    import parsimony  # noqa: F401

    for mod in _BANNED:
        assert mod not in sys.modules, f"{mod} must not be imported by the parsimony core"


def test_llm_call_counter_zero_after_workload():
    p = Parsimony(capacity=3)
    for i in range(12):
        p.on_write(make_item(f"i{i}", f"text number {i} value {i * 7}", [1.0, float(i % 3), 0.0]))
        p.touch(f"i{i % 3}")
    p.step()
    assert p.llm_calls == 0
    # still no LLM/torch after running everything
    for mod in _BANNED:
        assert mod not in sys.modules
