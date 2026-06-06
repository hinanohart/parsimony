from __future__ import annotations

import json

import pytest

from parsimony.cli import main
from parsimony.harness.run import run_bench


def test_run_bench_synthetic_structure():
    res = run_bench(data_path=None, n=6, capacity=5, ilp_sample=3, embed_dim=16)
    assert res["provenance"]["mode"] == "synthetic-honest"
    assert "competitive_ratio_vs_belady_ilp" in res["headline"]
    for name in ("parsimony", "recency", "mock_judge"):
        assert name in res["policies"]
        assert "source" in res["policies"][name]["coverage"]
    ratio = res["headline"]["competitive_ratio_vs_belady_ilp"]["value"]
    assert 0.0 <= ratio <= 1.5


def test_parsimony_zero_llm_in_bench():
    res = run_bench(data_path=None, n=5, capacity=4, ilp_sample=2, embed_dim=16)
    assert res["policies"]["parsimony"]["avg_llm_calls"]["value"] == 0.0
    assert res["policies"]["mock_judge"]["avg_llm_calls"]["value"] > 0.0


def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_demo_runs():
    assert main(["demo"]) == 0


def test_cli_bench_writes_results(tmp_path):
    out = tmp_path / "results.json"
    rc = main(["bench", "--n", "4", "--capacity", "4", "--ilp-sample", "2", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "headline" in data and "provenance" in data
