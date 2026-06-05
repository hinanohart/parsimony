"""parsimony — a deterministic, LLM-call-free forgetting policy for agent memory.

The public surface is assembled as the package stabilizes. Importing
``parsimony`` never imports an LLM SDK, ``torch``, or any network client.
"""

from __future__ import annotations

from .config import PolicyConfig
from .store import MemoryPool, normalize
from .types import (
    AccessEvent,
    AdmitDecision,
    CompressDecision,
    DecisionType,
    EvictDecision,
    MemoryItem,
    StepReport,
)

__version__ = "0.1.0a1"

__all__ = [
    "PolicyConfig",
    "MemoryPool",
    "normalize",
    "MemoryItem",
    "AccessEvent",
    "DecisionType",
    "AdmitDecision",
    "EvictDecision",
    "CompressDecision",
    "StepReport",
    "__version__",
]
