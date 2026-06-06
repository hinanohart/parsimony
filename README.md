# parsimony

> A deterministic, LLM-call-free **forgetting policy** for AI agent long-term memory.
> *(agent memory — **not** a KV/attention cache.)*

`parsimony` decides **what to admit, what to evict, how far to compress, and what
to merge** through one explainable objective function — not an LLM judge, not LRU
with a TTL. It is a *policy shim*: plug it into the memory store you already have
(mem0 / Letta / Zep), keep your store, swap in a principled forget decision.

**Status: pre-alpha (`v0.1.0a1`).** APIs may change. Everything claimed below is
checked in CI; the benchmark numbers are reproducible with `parsimony bench`.

```text
- CPU-only, torch-free core (depends on numpy).
- Zero LLM calls in the policy path — enforced at runtime and in CI.
- Deterministic: one seed governs every hash and tie-break; decisions are reproducible.
- Explainable: every decision emits an ExplainTrace whose terms sum to the utility.
```

## What it is good at (and what it is not)

parsimony optimizes the **coverage / diversity** of the retained set. On
LongMemEval it keeps the **highest-coverage, least-redundant** subset of any
online policy we tested, at **zero LLM cost**, and stays within **~97% of the
offline coverage-optimal (ILP) subset**.

It is **not** an answer-recall maximizer. If your only goal is "did the exact
gold session survive", a recency baseline or an LLM judge retains more (see the
honest table below). Coverage and answer-recall are different objectives, and we
report both rather than hiding the one we lose.

## Install

```bash
pip install "parsimony-mem @ git+https://github.com/hinanohart/parsimony"
# import name is `parsimony`; the PyPI distribution name is `parsimony-mem`
# (the name `parsimony` on PyPI is an unrelated caching library).
```

## Quickstart

```python
from parsimony import Parsimony
from parsimony.adapters.embedding import HashingEmbedder
from parsimony.types import MemoryItem

emb = HashingEmbedder(dim=64)
p = Parsimony(capacity=3)

def item(i, text):
    return MemoryItem(id=i, text=text, embedding=emb.encode(text))

p.on_write(item("m1", "the user's birthday is on the 3rd of March"))
p.on_write(item("m2", "the user's birthday is on the 3rd of March every year"))  # -> merge
p.on_write(item("m3", "the user likes hiking in the mountains"))
p.on_write(item("m4", "the user prefers tea over coffee"))                       # -> evict/reject

print([it.id for it in p.snapshot()])
print("llm calls:", p.llm_calls)          # 0
print(p.explain("m1")["counterfactual"])  # why this decision, and what would flip it
```

Run the worked example and the benchmark yourself:

```bash
parsimony demo
parsimony bench --data data/longmemeval_s.json --n 150 --capacity 10
```

## The four operators (one shared objective)

| operator | module | what it does |
|---|---|---|
| **A. admission** | `admission.py` | TinyLFU: a full pool admits a newcomer only if its keep-utility beats the weakest resident's |
| **B. eviction** | `eviction.py` | facility-location submodular greedy with ghost-mass weighting (keeps a coverage-maximal subset) |
| **C. compression** | `compression.py` | per-item rate-distortion level choice (`rate + λ·distortion`); extractive, non-generative |
| **D. dedup** | `dedup.py` | write-time semantic near-duplicate merge (cosine ≥ τ) — mem0 dedups by md5; this works in embedding space |

All four read the same `utility / cost` plane (`objective.py`), so every decision
is a linear combination of the same terms and is reported in an `ExplainTrace`.

## Benchmark (real data, reproducible)

LongMemEval (MIT, `xiaowu0162/longmemeval`), `longmemeval_s`, 150 questions,
capacity = 10, deterministic hashing embeddings (dim 256), seed 0. Gold is
machine-checkable (`answer_session_ids`) — **no LLM judge involved**. ILP solved
to proven optimality on the 50-question sample.

**Headline — coverage competitive ratio (parsimony's strength):**

- parsimony online coverage / **Bélády ILP-optimal coverage = 0.97**
- parsimony online coverage / clairvoyant greedy coverage = 0.97

**Per-policy (the full, honest picture):**

| policy | coverage ↑ | redundancy ↓ | gold-recall (any) ↑ | gold-recall (all) ↑ | LLM calls ↓ |
|---|---|---|---|---|---|
| **parsimony** | **0.744** | **0.564** | 0.220 | 0.040 | **0.0** |
| recency | 0.725 | 0.672 | 0.380 | 0.173 | 0.0 |
| oldest | 0.724 | 0.658 | 0.360 | 0.113 | 0.0 |
| random | 0.723 | 0.660 | 0.373 | 0.113 | 0.0 |
| mock_judge (LLM stand-in) | 0.719 | 0.786 | **0.613** | **0.353** | 50.5 |

Reading this honestly:

- **parsimony wins coverage and redundancy** — it keeps the most representative,
  least-redundant set, and it does so within 3% of the offline optimum, with no
  model calls.
- **parsimony loses gold-recall** to recency and to the LLM-judge stand-in. Its
  coverage objective deliberately drops moderately-redundant items, and on
  LongMemEval the answer session is often one of them. If recall is your goal,
  use recency or blend it in; parsimony is for diversity / dedup / curation.
- On this corpus **dedup makes no merges** (full sessions are not paraphrases of
  each other), so `parsimony` and `parsimony+dedup` coincide. Dedup's value shows
  on paraphrase-heavy memories (see `parsimony demo`).

## CLAIM / NON-CLAIM

| we claim | we do **not** claim |
|---|---|
| deterministic, zero-LLM, explainable decisions | that it improves downstream QA accuracy |
| ~97% of ILP-optimal coverage online (measured) | that it beats recency on answer-recall (it does not) |
| highest coverage / lowest redundancy of tested online policies | generative summarization (compression is extractive) |
| a measured Bélády competitive ratio for *coverage* | a competitive ratio for task recall |

## mem0 drop-in

`parsimony.adapters.mem0` wraps mem0's add/update path so the forget decision is
parsimony's while the store stays mem0's. It is an optional import; the core never
imports mem0.

## How the numbers stay honest

`ci/check_claims.sh` runs in CI and fails the build if: a forbidden hype phrase
appears; the core imports `torch` or an LLM SDK; the policy engine uses a
nondeterministic RNG; or a committed `bench/results.json` reports a value without
a `source` provenance tag. README benchmark numbers are produced by
`parsimony bench` from `bench/results.json` (LongMemEval real trace, or a clearly
labelled synthetic trace when no corpus is present).

## Related

Part of the hinanohart memory toolkit, each at a different layer:
[memcanon](https://github.com/hinanohart/memcanon) (post-hoc audit/dedup),
[chronospect](https://github.com/hinanohart/chronospect) (in-weight timescale
measurement). parsimony is the *write/evict-time policy*. A companion idea,
beladymem, scores forgetting policies against Bélády-MIN; parsimony's oracle is
self-contained (no code dependency).

## License

MIT.
