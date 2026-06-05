from __future__ import annotations

from _helpers import make_item
from parsimony.config import PolicyConfig
from parsimony.objective import freq_norm, prefers, utility, utility_terms


def test_freq_norm_bounds():
    assert freq_norm(0) == 0.0
    assert freq_norm(16) == 1.0
    assert freq_norm(100) == 1.0
    assert freq_norm(-5) == 0.0


def test_a1_utility_is_pure_frequency():
    cfg = PolicyConfig()  # a1 defaults
    u = utility(cfg, freq_est=8, removal_loss=0.9, salience=5.0)
    assert u == freq_norm(8)  # cover/salience weights are 0


def test_utility_terms_sum_to_utility():
    cfg = PolicyConfig(w_freq=1.0, w_cover=0.5, w_salience=0.3)
    total, terms = utility_terms(cfg, freq_est=8, removal_loss=0.4, salience=2.0)
    assert abs(total - sum(t["contribution"] for t in terms)) < 1e-12
    assert abs(total - utility(cfg, freq_est=8, removal_loss=0.4, salience=2.0)) < 1e-12


def test_prefers_ordering():
    hi = make_item("a", "x", [1.0, 0.0], salience=2.0, created_at=10.0)
    lo = make_item("b", "y", [1.0, 0.0], salience=1.0, created_at=1.0)
    assert prefers(hi, lo) is True  # salience wins
    old = make_item("a", "x", [1.0, 0.0], salience=1.0, created_at=1.0)
    new = make_item("b", "y", [1.0, 0.0], salience=1.0, created_at=9.0)
    assert prefers(old, new) is True  # older wins on salience tie
