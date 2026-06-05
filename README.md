# parsimony

> agent long-term memory forgetting policy — **not** a KV/attention cache.

Deterministic, LLM-call-free **forgetting policy** for AI agent long-term memory.
It decides *what to admit, what to evict, how far to compress, and what to merge*
through one explainable objective function — not an LLM judge, not LRU/TTL.

**Status:** pre-alpha (`v0.1.0a1`). Scaffold in progress; see CI for what is verified.

- CPU-only, `torch`-free core (depends on `numpy`).
- Zero LLM calls in the policy path (mechanically enforced in CI).
- Deterministic: a single seed governs every hash and tie-break.
- A *policy shim* you plug into the memory store you already have (mem0/Letta/Zep) — not a replacement store.

Full documentation lands as the package stabilizes. License: MIT.
