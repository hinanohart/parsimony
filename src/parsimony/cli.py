"""Command-line interface: ``parsimony bench`` and ``parsimony demo``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def _cmd_bench(args: argparse.Namespace) -> int:
    from .harness.run import run_bench

    results = run_bench(
        data_path=args.data,
        n=args.n,
        capacity=args.capacity,
        ilp_sample=args.ilp_sample,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    prov = results["provenance"]
    print(
        f"mode={prov['mode']} source={prov['source']} n={prov['n_questions']} k={prov['capacity']}"
    )
    print("headline:")
    for key, rec in results["headline"].items():
        print(f"  {key}: {rec['value']}")
    print("coverage / gold_hit_any / avg_llm_calls by policy:")
    for name, rec in results["policies"].items():
        print(
            f"  {name:16s} cov={rec['coverage']['value']:.3f} "
            f"hit_any={rec['gold_hit_any']['value']:.3f} "
            f"llm={rec['avg_llm_calls']['value']:.1f} "
            f"merges={rec['avg_merges']['value']:.1f}"
        )
    print(f"results written to {out}")
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    from .adapters.embedding import HashingEmbedder
    from .policy import Parsimony
    from .types import MemoryItem

    emb = HashingEmbedder(dim=64)
    p = Parsimony(capacity=3)

    def item(i: str, text: str, salience: float = 1.0) -> MemoryItem:
        return MemoryItem(id=i, text=text, embedding=emb.encode(text), salience=salience)

    writes = [
        ("m1", "the user's birthday is on the 3rd of March"),
        ("m2", "the user's birthday is on the 3rd of March every year"),  # near-dup -> merge
        ("m3", "the user likes hiking in the mountains"),
        ("m4", "the user prefers tea over coffee"),
        ("m5", "the user lives in Berlin near the river"),
    ]
    for i, text in writes:
        d = p.on_write(item(i, text))
        print(f"write {i}: {d.decision.value:7s} - {d.reason}")
    print("\nretained:", [it.id for it in p.snapshot()])
    print(f"llm calls: {p.llm_calls}")
    last = p.snapshot()[-1].id
    print(f"\nexplain({last}):")
    print(json.dumps(p.explain(last), indent=2)[:800])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parsimony",
        description="Deterministic, LLM-call-free forgetting policy for agent memory.",
    )
    parser.add_argument("--version", action="version", version=f"parsimony {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="run the retention benchmark vs Belady")
    b.add_argument("--data", default=None, help="path to LongMemEval json (else synthetic)")
    b.add_argument("--n", type=int, default=120)
    b.add_argument("--capacity", type=int, default=10)
    b.add_argument("--ilp-sample", type=int, default=40, dest="ilp_sample")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="bench/results.json")
    b.set_defaults(func=_cmd_bench)

    d = sub.add_parser("demo", help="run a tiny worked example")
    d.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
