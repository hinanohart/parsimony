from __future__ import annotations

from _helpers import make_item
from parsimony.compress_ops import extractive
from parsimony.compression import apply_compression, compress_item
from parsimony.config import PolicyConfig
from parsimony.types import DecisionType

LONG = (
    "The quarterly revenue rose to 4200 dollars. "
    "Marketing spend stayed flat over the period. "
    "The team shipped three features last month. "
    "Customer churn was broadly unchanged."
)


def test_low_salience_compresses():
    item = make_item("a", LONG, [1.0, 0.0], salience=0.01)
    cfg = PolicyConfig()
    d = compress_item(item, cfg)
    assert d.chosen_level > 0  # cheap to lose -> compress hard
    assert d.rate_saved > 0


def test_high_salience_numeric_stays_verbatim():
    item = make_item("a", LONG, [1.0, 0.0], salience=50.0)
    cfg = PolicyConfig()
    d = compress_item(item, cfg)
    assert d.chosen_level == 0
    assert d.decision == DecisionType.KEEP
    assert d.rate_saved == 0


def test_rd_curve_rate_monotone():
    item = make_item("a", LONG, [1.0, 0.0], salience=1.0)
    d = compress_item(item, PolicyConfig())
    rates = [c["rate"] for c in d.trace["rd_curve"]]
    assert rates == sorted(rates, reverse=True)  # higher level -> lower rate
    assert abs(rates[0] - 1.0) < 1e-9  # verbatim rate is 1.0


def test_coverage_residual_discounts_drop():
    item = make_item("a", LONG, [1.0, 0.0], salience=1.0)
    cfg = PolicyConfig()
    no_cov = compress_item(item, cfg)
    high_cov = compress_item(item, cfg, coverage_residual=0.95)
    drop_d_no = no_cov.trace["rd_curve"][-1]["distortion"]
    drop_d_cov = high_cov.trace["rd_curve"][-1]["distortion"]
    assert drop_d_cov < drop_d_no  # well-covered item is cheaper to drop


def test_short_text_not_split():
    item = make_item("a", "single short note", [1.0, 0.0], salience=0.01)
    cfg = PolicyConfig()
    d = compress_item(item, cfg)
    # skeleton of a 1-sentence text equals the text; only verbatim or drop apply
    assert d.chosen_level in {0, 2}


def test_apply_compression_records_level():
    item = make_item("a", LONG, [1.0, 0.0], salience=1.0)
    cfg = PolicyConfig()
    out = apply_compression(item, cfg, level=1)
    assert out.compression_level == 1
    assert out.tokens == extractive.token_count(out.text)
    assert out.tokens <= item.tokens


def test_deterministic():
    item = make_item("a", LONG, [1.0, 0.0], salience=0.5)
    cfg = PolicyConfig()
    assert compress_item(item, cfg).chosen_level == compress_item(item, cfg).chosen_level
