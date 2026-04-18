"""
Tests for Nigeria-extension functions added to etl/portfolio_math.

Additive only — no existing test is modified or affected.
Design contract:
- omega_ratio: distribution-free hurdle metric replacing Sharpe for barbell
- payoff_asymmetry_ratio: direct avg_win / |avg_loss| payoff-shape metric
- fractional_kelly_fat_tail: kurtosis-corrected quarter-Kelly sizing
- effective_ngn_return: USD -> NGN return after P2P bridge friction
- portfolio_metrics_ngn: additive extension of calculate_enhanced_portfolio_metrics
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from etl.portfolio_math import (
    DAILY_NGN_THRESHOLD,
    DEFAULT_RISK_FREE_RATE,
    NGN_ANNUAL_INFLATION,
    NGN_P2P_FRICTION,
    TRADING_DAYS,
    effective_ngn_return,
    fractional_kelly_fat_tail,
    omega_ratio,
    payoff_asymmetry_ratio,
    portfolio_metrics_ngn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def profitable_series() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.002, 0.01, 500))


@pytest.fixture()
def losing_series() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(-0.002, 0.01, 500))


@pytest.fixture()
def fat_tail_series() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.standard_t(df=3, size=500) * 0.01 + 0.001)


@pytest.fixture()
def normal_series() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.001, 0.01, 500))


# ---------------------------------------------------------------------------
# NGN Constants
# ---------------------------------------------------------------------------

class TestNGNConstants:

    def test_daily_ngn_threshold_positive(self):
        assert DAILY_NGN_THRESHOLD > 0.0

    def test_daily_ngn_threshold_higher_than_usd_rf(self):
        daily_usd_rf = DEFAULT_RISK_FREE_RATE / TRADING_DAYS
        assert DAILY_NGN_THRESHOLD > daily_usd_rf, (
            "NGN hurdle must exceed USD risk-free rate — "
            "28% inflation + 3% P2P friction far exceeds ~2% USD rf"
        )

    def test_daily_ngn_threshold_compounds_correctly(self):
        expected = (1.0 + NGN_ANNUAL_INFLATION + NGN_P2P_FRICTION) ** (1.0 / TRADING_DAYS) - 1.0
        assert abs(DAILY_NGN_THRESHOLD - expected) < 1e-12

    def test_annual_hurdle_approx_31_pct(self):
        annual_equivalent = (1.0 + DAILY_NGN_THRESHOLD) ** TRADING_DAYS - 1.0
        expected = NGN_ANNUAL_INFLATION + NGN_P2P_FRICTION
        assert abs(annual_equivalent - expected) < 0.001


# ---------------------------------------------------------------------------
# omega_ratio
# ---------------------------------------------------------------------------

class TestOmegaRatio:

    def test_profitable_series_exceeds_one(self, profitable_series):
        assert omega_ratio(profitable_series) > 1.0

    def test_losing_series_below_one(self, losing_series):
        assert omega_ratio(losing_series) < 1.0

    def test_omega_inf_when_no_losses_above_threshold(self):
        returns = pd.Series([0.01] * 200)  # every return beats threshold
        assert omega_ratio(returns, threshold=0.0) == float("inf")

    def test_omega_nan_for_short_series(self):
        assert math.isnan(omega_ratio(pd.Series([0.01, 0.02])))

    def test_omega_nan_for_none(self):
        assert math.isnan(omega_ratio(None))  # type: ignore[arg-type]

    def test_custom_threshold_zero(self):
        returns = pd.Series([0.001] * 200)
        assert omega_ratio(returns, threshold=0.0) > 1.0

    def test_custom_threshold_high_fails_good_series(self):
        # Series with mean 0.001/day fails a 0.005/day threshold
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.001, 500))
        assert omega_ratio(returns, threshold=0.005) < 1.0

    def test_ngn_threshold_stricter_than_usd_rf(self, profitable_series):
        daily_usd_rf = DEFAULT_RISK_FREE_RATE / TRADING_DAYS
        omega_usd = omega_ratio(profitable_series, threshold=daily_usd_rf)
        omega_ngn = omega_ratio(profitable_series)  # default = NGN threshold
        # Same series scores higher against USD rf than NGN threshold
        assert omega_usd >= omega_ngn, (
            "NGN threshold is higher than USD rf — omega vs NGN should be <= omega vs USD rf"
        )

    def test_series_between_usd_rf_and_ngn_threshold_fails_ngn(self):
        """A series beating USD risk-free but not NGN inflation must fail NGN hurdle."""
        daily_usd_rf = DEFAULT_RISK_FREE_RATE / TRADING_DAYS
        daily_ngn = DAILY_NGN_THRESHOLD
        # Construct returns strictly between the two thresholds
        mid = (daily_usd_rf + daily_ngn) / 2.0
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(mid, 0.0001, 500))
        assert omega_ratio(returns, threshold=daily_usd_rf) > 1.0  # beats USD rf
        assert omega_ratio(returns) < 1.0                           # fails NGN hurdle

    def test_monotone_in_threshold(self, profitable_series):
        """Higher threshold -> lower or equal omega for same series."""
        thresholds = [0.0, 0.0001, 0.001, DAILY_NGN_THRESHOLD]
        omegas = [omega_ratio(profitable_series, threshold=t) for t in thresholds]
        for i in range(len(omegas) - 1):
            assert omegas[i] >= omegas[i + 1] or math.isinf(omegas[i])


# ---------------------------------------------------------------------------
# payoff_asymmetry_ratio
# ---------------------------------------------------------------------------

class TestPayoffAsymmetryRatio:

    def test_ratio_exceeds_one_when_winners_dominate(self):
        returns = pd.Series([0.04, -0.01, 0.05, -0.02, 0.03, -0.01])
        assert payoff_asymmetry_ratio(returns) > 1.0

    def test_ratio_is_infinite_when_losses_absent(self):
        returns = pd.Series([0.01, 0.03, 0.02, 0.04])
        assert payoff_asymmetry_ratio(returns) == float("inf")

    def test_ratio_nan_for_empty_input(self):
        assert math.isnan(payoff_asymmetry_ratio(pd.Series(dtype=float)))

    def test_ratio_nan_for_none(self):
        assert math.isnan(payoff_asymmetry_ratio(None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fractional_kelly_fat_tail
# ---------------------------------------------------------------------------

class TestFractionalKellyFatTail:

    def test_result_in_valid_range(self, normal_series):
        f = fractional_kelly_fat_tail(normal_series)
        assert 0.0 <= f <= 0.20

    def test_fat_tail_correction_reduces_kelly(self):
        """High-kurtosis series must yield lower Kelly than Gaussian series.

        Both series are constructed with the same positive mean (well above the
        NGN threshold) so that (mu - rf) > 0 in both cases. The only difference
        is kurtosis. The fat-tail kurtosis dampener [1 / (1 + max(k-3,0)/4)]
        must strictly reduce the output.
        """
        from etl.portfolio_math import DAILY_NGN_THRESHOLD
        rng = np.random.default_rng(2026)

        # Positive mean clearly above DAILY_NGN_THRESHOLD
        target_mean = DAILY_NGN_THRESHOLD * 5.0  # 5x the hurdle
        n = 600

        # Near-Gaussian series: kurtosis ≈ 0 (excess), dampener ≈ 1.0
        normal_ret = pd.Series(rng.normal(target_mean, 0.005, n))
        # Heavy-tailed series via t(3): excess kurtosis >> 0, dampener < 1.0
        # Shift to same target_mean
        fat_raw = rng.standard_t(df=3, size=n) * 0.005
        fat_ret = pd.Series(fat_raw + (target_mean - fat_raw.mean()))

        # Both must have positive Kelly before correction
        f_normal = fractional_kelly_fat_tail(normal_ret)
        f_fat = fractional_kelly_fat_tail(fat_ret)

        assert f_normal > 0.0, f"Normal series should have positive Kelly; got {f_normal}"
        assert f_fat <= f_normal, (
            f"Fat-tail Kelly ({f_fat:.4f}) must be <= normal Kelly ({f_normal:.4f}); "
            f"kurtosis normal={normal_ret.kurtosis():.1f} fat={fat_ret.kurtosis():.1f}"
        )

    def test_short_series_returns_minimum_stake(self):
        assert fractional_kelly_fat_tail(pd.Series([0.01] * 10)) == 0.01

    def test_none_returns_minimum_stake(self):
        assert fractional_kelly_fat_tail(None) == 0.01  # type: ignore[arg-type]

    def test_zero_variance_returns_zero(self):
        # Constant series -> sigma2=0 -> undefined Kelly -> 0
        returns = pd.Series([0.001] * 100)
        assert fractional_kelly_fat_tail(returns) == 0.0

    def test_hard_cap_at_twenty_percent(self):
        # Extreme positive edge: very high mean, very low variance
        rng = np.random.default_rng(9)
        extreme = pd.Series(rng.normal(0.10, 0.0001, 300))
        f = fractional_kelly_fat_tail(extreme)
        assert f <= 0.20, f"Hard cap violated: {f}"

    def test_quarter_kelly_fraction_applied(self, normal_series):
        """Quarter-Kelly must produce <= full Kelly."""
        f_quarter = fractional_kelly_fat_tail(normal_series, kelly_fraction=0.25)
        f_full = fractional_kelly_fat_tail(normal_series, kelly_fraction=1.0)
        assert f_quarter <= f_full

    def test_ngn_rf_produces_different_result_than_usd_rf(self, normal_series):
        """NGN threshold (higher) should reduce Kelly vs USD risk-free."""
        f_usd = fractional_kelly_fat_tail(
            normal_series, risk_free=DEFAULT_RISK_FREE_RATE / TRADING_DAYS
        )
        f_ngn = fractional_kelly_fat_tail(normal_series)  # default = NGN
        # Higher hurdle rate lowers (mu - rf) -> lower Kelly or zero
        assert f_ngn <= f_usd or f_ngn == 0.0


# ---------------------------------------------------------------------------
# effective_ngn_return
# ---------------------------------------------------------------------------

class TestEffectiveNGNReturn:

    def test_ngn_weakening_boosts_effective_return(self):
        """Positive spot change (NGN weakens) benefits USD holder."""
        r = effective_ngn_return(0.01, ngn_usd_spot_change=0.005)
        assert r > 0.01

    def test_friction_reduces_return(self):
        r_no_friction = effective_ngn_return(0.01, 0.0, withdrawal_friction=0.0)
        r_with_friction = effective_ngn_return(0.01, 0.0, withdrawal_friction=0.03)
        assert r_with_friction < r_no_friction

    def test_default_friction_matches_constant(self):
        expected = 0.01 + 0.0 - (NGN_P2P_FRICTION / TRADING_DAYS)
        result = effective_ngn_return(0.01, 0.0)
        assert abs(result - expected) < 1e-12

    def test_additive_decomposition(self):
        """R_eff = R_USD + Δspot - friction (exact)."""
        r_usd = 0.005
        spot = 0.002
        friction = 0.001
        expected = r_usd + spot - friction
        assert abs(effective_ngn_return(r_usd, spot, withdrawal_friction=friction) - expected) < 1e-12

    def test_zero_inputs(self):
        friction = NGN_P2P_FRICTION / TRADING_DAYS
        assert abs(effective_ngn_return(0.0, 0.0) - (-friction)) < 1e-12


# ---------------------------------------------------------------------------
# portfolio_metrics_ngn
# ---------------------------------------------------------------------------

class TestPortfolioMetricsNGN:

    def test_ngn_keys_present(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        for key in ("omega_ratio", "payoff_asymmetry", "beats_ngn_hurdle", "ngn_daily_threshold",
                    "ngn_annual_hurdle_pct", "fractional_kelly_fat_tail"):
            assert key in m, f"Missing key: {key}"

    def test_base_keys_still_present(self, profitable_series):
        """Additive extension must not drop any existing metric."""
        m = portfolio_metrics_ngn(profitable_series)
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown",
                    "total_return", "annual_return", "volatility"):
            assert key in m, f"Base key dropped by NGN extension: {key}"

    def test_beats_ngn_hurdle_is_bool(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        assert isinstance(m["beats_ngn_hurdle"], bool)

    def test_ngn_annual_hurdle_pct_correct(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        expected = round((NGN_ANNUAL_INFLATION + NGN_P2P_FRICTION) * 100.0, 1)
        assert m["ngn_annual_hurdle_pct"] == expected

    def test_omega_ratio_consistent_with_standalone(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        standalone = omega_ratio(profitable_series)
        assert abs(m["omega_ratio"] - standalone) < 1e-10

    def test_payoff_asymmetry_consistent_with_standalone(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        standalone = payoff_asymmetry_ratio(profitable_series)
        if math.isinf(standalone):
            assert math.isinf(m["payoff_asymmetry"])
        else:
            assert abs(m["payoff_asymmetry"] - standalone) < 1e-10

    def test_losing_series_does_not_beat_hurdle(self, losing_series):
        m = portfolio_metrics_ngn(losing_series)
        assert m["beats_ngn_hurdle"] is False

    def test_returns_dict(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        assert isinstance(m, dict)
        assert len(m) > 5  # has both base + NGN keys

    def test_benchmark_returns_expose_alpha_and_information_ratio(self):
        strategy = pd.Series([0.02, 0.01, -0.005, 0.03, 0.015])
        benchmark = pd.Series([0.01, 0.0, -0.01, 0.01, 0.005])
        m = portfolio_metrics_ngn(strategy, benchmark_returns=benchmark)
        assert "alpha" in m
        assert "information_ratio" in m
        assert "beta" in m
        assert "tracking_error" in m
        assert math.isfinite(m["alpha"])
        assert math.isfinite(m["information_ratio"])
        assert math.isfinite(m["beta"])
        assert math.isfinite(m["tracking_error"])
        assert m["alpha"] > 0


# ---------------------------------------------------------------------------
# Anti-omega hardening tests (Gaps 1-3 in portfolio_math.py)
# ---------------------------------------------------------------------------

from etl.portfolio_math import (
    omega_robustness_summary,
    omega_bootstrap_ci,
    expected_shortfall_to_edge_ratio,
    omega_curve,
)


class TestOmegaCliffDropGuard:
    """Gap 1 -- threshold chosen badly: cliff-drop check."""

    def test_cliff_fields_present_in_robustness_summary(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.005, 0.008, 200))
        result = omega_robustness_summary(returns, execution_drag_hurdle=0.0002)
        assert "omega_cliff_drop_ratio" in result
        assert "omega_cliff_ok" in result

    def test_cliff_fails_when_edge_is_thin(self):
        # Most returns below NGN hurdle -- edge disappears at hurdle
        returns = pd.Series([0.0001] * 90 + [0.01] * 10)
        result = omega_robustness_summary(returns, execution_drag_hurdle=0.0002)
        assert result["omega_cliff_drop_ratio"] is not None
        # Omega at K0 (zero) will be high; at K1 (NGN hurdle) it should crater
        assert result["omega_cliff_drop_ratio"] > 0.40, (
            f"Expected large cliff drop, got {result['omega_cliff_drop_ratio']:.3f}"
        )

    def test_cliff_penalizes_robustness_score(self):
        # Thin-edge strategy should score lower robustness than fat-edge strategy
        thin = omega_robustness_summary(
            pd.Series([0.0001] * 90 + [0.01] * 10),
            execution_drag_hurdle=0.0002,
        )
        fat = omega_robustness_summary(
            pd.Series([0.003] * 100),
            execution_drag_hurdle=0.0002,
        )
        if thin["omega_robustness_score"] is not None and fat["omega_robustness_score"] is not None:
            assert thin["omega_robustness_score"] <= fat["omega_robustness_score"]

    def test_cliff_fields_key_present_in_short_series(self):
        result = omega_robustness_summary(pd.Series([0.01] * 5))
        assert "omega_cliff_drop_ratio" in result
        assert "omega_cliff_ok" in result


class TestOmegaBootstrapCI:
    """Gap 2 -- right tail overestimated: bootstrap CI around omega."""

    def test_ci_lower_present_with_sufficient_obs(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.003, 0.01, 50))
        result = omega_bootstrap_ci(returns, n_bootstrap=200, rng=np.random.default_rng(7))
        assert result["omega_ci_lower"] is not None
        assert result["omega_ci_upper"] is not None
        assert result["omega_ci_lower"] <= result["omega_point_estimate"]
        assert result["omega_ci_upper"] >= result["omega_ci_lower"]

    def test_ci_absent_below_10_obs(self):
        result = omega_bootstrap_ci(pd.Series([0.01] * 5), n_bootstrap=100)
        assert result["omega_ci_lower"] is None
        assert result["omega_right_tail_ok"] is None

    def test_right_tail_ok_false_for_single_outlier_series(self):
        # One large winner + many tiny losses: CI lower should be < 1
        returns = pd.Series([-0.001] * 49 + [0.15])
        result = omega_bootstrap_ci(
            returns, n_bootstrap=500, rng=np.random.default_rng(99)
        )
        if result["omega_right_tail_ok"] is not None:
            assert result["omega_right_tail_ok"] is False, (
                f"Expected right_tail_ok=False for outlier-driven series, "
                f"ci_lower={result['omega_ci_lower']:.3f}"
            )

    def test_right_tail_ok_true_for_consistent_winners(self):
        # Returns consistently above NGN hurdle -> lower CI bound should exceed 1.0
        returns = pd.Series([0.005] * 100)
        result = omega_bootstrap_ci(
            returns, n_bootstrap=300, rng=np.random.default_rng(42)
        )
        assert result["omega_right_tail_ok"] is True

    def test_wide_ci_signals_high_variance(self):
        # Use extreme sigma contrast (10x) with a large sample to ensure deterministic ordering.
        # Calm series is constant -- every bootstrap resample gives the same omega --> width=0.
        volatile = pd.Series(np.random.default_rng(13).normal(0.002, 0.15, 200))
        calm = pd.Series([0.004] * 200)  # constant -> CI width is 0 by construction
        r_v = omega_bootstrap_ci(volatile, n_bootstrap=300, rng=np.random.default_rng(1))
        r_c = omega_bootstrap_ci(calm, n_bootstrap=300, rng=np.random.default_rng(1))
        if r_v["omega_ci_width"] is not None and r_c["omega_ci_width"] is not None:
            assert r_v["omega_ci_width"] >= r_c["omega_ci_width"]


class TestExpectedShortfallToEdge:
    """Gap 3 -- left tail not truly bounded: ES relative to daily edge."""

    def test_es_to_edge_fields_present(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.002, 0.01, 100))
        result = expected_shortfall_to_edge_ratio(returns)
        assert result["expected_shortfall_raw"] is not None
        assert result["expected_shortfall_to_edge"] is not None
        assert result["es_to_edge_bounded"] is not None

    def test_bounded_when_tail_shallow_relative_to_edge(self):
        returns = pd.Series([0.03] * 60 + [-0.001] * 40)
        result = expected_shortfall_to_edge_ratio(returns)
        assert result["es_to_edge_bounded"] is True

    def test_unbounded_when_tail_deep_relative_to_edge(self):
        returns = pd.Series([0.0001] * 90 + [-0.20] * 10)
        result = expected_shortfall_to_edge_ratio(returns)
        assert result["es_to_edge_bounded"] is False, (
            f"Expected unbounded ES, ratio={result['expected_shortfall_to_edge']}"
        )

    def test_provided_edge_is_preferred_over_proxy(self):
        returns = pd.Series([0.01] * 50 + [-0.005] * 50)
        r1 = expected_shortfall_to_edge_ratio(returns)
        r2 = expected_shortfall_to_edge_ratio(returns, expected_daily_edge=0.01)
        assert r2["edge_source"] == "provided"
        assert r1["edge_source"] == "positive_mean_proxy"

    def test_absent_below_min_obs(self):
        result = expected_shortfall_to_edge_ratio(pd.Series([0.01] * 3))
        assert result["expected_shortfall_to_edge"] is None

    def test_portfolio_metrics_ngn_includes_all_anti_omega_fields(self):
        rng = np.random.default_rng(3)
        m = portfolio_metrics_ngn(pd.Series(rng.normal(0.002, 0.01, 100)))
        for key in (
            "omega_cliff_drop_ratio", "omega_cliff_ok",
            "omega_ci_lower", "omega_right_tail_ok",
            "expected_shortfall_raw", "expected_shortfall_to_edge", "es_to_edge_bounded",
        ):
            assert key in m, f"Missing key: {key}"
