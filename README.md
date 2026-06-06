# parsimony

> A deterministic, LLM-call-free **forgetting policy** for AI agent long-term memory.
> *(agent memory — **not** a KV/attention cache.)*

`parsimony` decides **what to admit, what to evict, how far to compress, and what
to merge** through one explainable objective function — not an LLM judge, not LRU
with a TTL. It is a *policy shim*: plug it into the memory store you already have
(mem0 / Letta / Zep), keep your store, swap in a principled forget decision.

**Status: pre-alpha (`v0.1.0a2`).** APIs may change. Everything claimed below is
checked in CI; the benchmark numbers are reproducible with `parsimony bench`.

```text
- CPU-only, torch-free core (depends on numpy).
- Zero LLM calls in the policy path — enforced at runtime and in CI.
- Deterministic: one seed governs every hash and tie-break; decisions are reproducible.
- Explainable: every decision emits an ExplainTrace with a counterfactual; admission/dedup traces carry contributing_terms that sum to the utility.
```

## What it is good at (and what it is not)

parsimony optimizes the **diversity** of the retained set: its ghost-mass
facility-location eviction drops the most redundant memory each time, so on
LongMemEval it keeps the **least-redundant subset of any online policy we
tested** (intra-set redundancy **0.41 vs 0.66–0.79** for recency / random /
LLM-judge — a ~37% reduction), at **zero LLM cost**, while its facility-location
coverage stays within **~94% of the offline coverage-optimal (ILP) subset**.

It is **not** a coverage winner or an answer-recall maximizer. On this corpus
every online policy scores ~0.72 coverage (≈94–95% of the ILP optimum), so
coverage does not separate them; if your goal is "did the exact gold session
survive", a recency baseline or an LLM judge retains more (see the honest table
below). We report every metric, including the ones we do not win.

## Install

```bash
pip install "parsimony-mem @ git+https://github.com/hinanohart/parsimony"
# import name is `parsimony`; the PyPI distribution name is `parsimony-mem`
# (the name `parsimony` on PyPI is an unrelated caching library).

# to run the ILP benchmark you also need the [bench] extra (pulp, datasets):
pip install "parsimony-mem[bench] @ git+https://github.com/hinanohart/parsimony"
# (without pulp, the Belady ILP gracefully falls back to the clairvoyant greedy oracle)
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
p.on_write(item("m2", "the user's birthday is on the 3rd of March"))  # same content -> merge
p.on_write(item("m3", "the user likes hiking in the mountains"))
p.on_write(item("m4", "the user prefers tea over coffee"))           # full pool -> evict/reject

print([it.id for it in p.snapshot()])
print("llm calls:", p.llm_calls)          # 0
print(p.explain("m1")["counterfactual"])  # why this decision, and what would flip it
```

Run the worked example and the benchmark yourself:

```bash
parsimony demo
parsimony bench --data data/longmemeval_s.json --n 150 --capacity 10 --ilp-sample 50
```

## The four operators (one shared objective)

| operator | module | what it does |
|---|---|---|
| **A. admission** | `admission.py` | TinyLFU: a full pool admits a newcomer only if its keep-utility beats the weakest resident's (opt-in via `admission_control=True`) |
| **B. eviction** | `eviction.py` | facility-location submodular greedy with ghost-mass weighting (keeps a coverage-maximal subset) — the **default** `on_write` overflow behaviour |
| **C. compression** | `compression.py` | per-item rate-distortion level choice (`rate + λ·distortion`); extractive, non-generative |
| **D. dedup** | `dedup.py` | write-time semantic near-duplicate merge (cosine ≥ τ) — mem0 dedups by md5; this works in embedding space |

The four operators are built from the same signals (frequency, coverage,
salience, rate) and one shared config. Admission and dedup route through the
unified `utility()` (`objective.py`); eviction (facility-location) and
compression (rate-distortion) apply their own coverage and rate-distortion
criteria directly. Every decision is reported as an `ExplainTrace` with a
one-line counterfactual; admission and dedup traces additionally carry
`contributing_terms` that sum to the utility, while eviction and compression
traces carry their coverage and rate-distortion fields.

## Benchmark (real data, reproducible)

LongMemEval (MIT, `xiaowu0162/longmemeval`), `longmemeval_s`, 150 questions,
capacity = 10, deterministic hashing embeddings (dim 256), seed 0. Gold is
machine-checkable (`answer_session_ids`) — **no LLM judge involved**. ILP solved
to proven optimality on the 50-question sample.

**Headline — intra-set redundancy (parsimony's strength):**

- parsimony retained-set redundancy **0.41** vs **0.66–0.79** for every other
  online policy (lower = more diverse) — a ~37% reduction, by design.
- it pays almost nothing in coverage for that diversity: facility-location
  coverage **0.72 = 94% of the offline ILP optimum** (0.76), on par with the
  baselines (which reach ~95%). Coverage does not separate online policies here.

**Per-policy (the full, honest picture):**

| policy | coverage ↑ | redundancy ↓ | gold-recall (any) ↑ | gold-recall (all) ↑ | LLM calls ↓ |
|---|---|---|---|---|---|
| **parsimony** | 0.719 | **0.414** | 0.213 | 0.053 | **0.0** |
| recency | 0.725 | 0.672 | 0.380 | 0.173 | 0.0 |
| oldest | 0.724 | 0.658 | 0.360 | 0.113 | 0.0 |
| random | 0.723 | 0.660 | 0.373 | 0.113 | 0.0 |
| mock_judge (LLM stand-in) | 0.719 | 0.786 | **0.613** | **0.353** | 50.5 |

Reading this honestly:

- **parsimony wins redundancy by a wide margin** — 0.41 vs 0.66–0.79. It keeps
  the most diverse retained set of any policy tested, by design (it evicts the
  most redundant memory each step), with no model calls.
- **parsimony does not win coverage.** All online policies land at ~0.72 (94–95%
  of the offline ILP optimum); coverage does not separate them on this corpus, so
  we make no superiority claim there — only that the diversity gain costs almost
  nothing in coverage (still 94% of optimal).
- **parsimony loses gold-recall** to recency and to the LLM-judge stand-in. It
  deliberately drops redundant items, and on LongMemEval the answer session is
  often a redundant one. If recall is your goal, use recency or blend it in;
  parsimony is for diversity / dedup / curation.
- On this corpus **dedup makes no merges** (full sessions are not paraphrases of
  each other), so `parsimony` and `parsimony+dedup` coincide. Dedup's value shows
  on paraphrase-heavy memories (see `parsimony demo`).

## CLAIM / NON-CLAIM

| we claim | we do **not** claim |
|---|---|
| deterministic, zero-LLM, explainable decisions | that it improves downstream QA accuracy |
| **lowest intra-set redundancy** (most diverse retained set) of tested online policies (measured) | that it beats recency on answer-recall (it does not) |
| coverage within ~94% of the offline ILP optimum (measured) | that it has the **highest** coverage (baselines tie or edge it) |
| a measured Bélády competitive ratio for *coverage* | generative summarization (compression is extractive); a competitive ratio for task recall |

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
