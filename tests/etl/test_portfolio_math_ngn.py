"""
Tests for Nigeria-extension functions added to etl/portfolio_math.

Additive only — no existing test is modified or affected.
These functions are Phase 11 stubs; wiring (Phases B-E) begins after
THIN_LINKAGE >= 10 (gate warmup expires 2026-04-15).

Design contract:
- omega_ratio: distribution-free hurdle metric replacing Sharpe for barbell
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
        for key in ("omega_ratio", "beats_ngn_hurdle", "ngn_daily_threshold",
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

    def test_losing_series_does_not_beat_hurdle(self, losing_series):
        m = portfolio_metrics_ngn(losing_series)
        assert m["beats_ngn_hurdle"] is False

    def test_returns_dict(self, profitable_series):
        m = portfolio_metrics_ngn(profitable_series)
        assert isinstance(m, dict)
        assert len(m) > 5  # has both base + NGN keys
