"""
tests/forcester_ts/test_var_backtest.py
----------------------------------------
Unit tests for VaRBacktester — Kupiec, Christoffersen, pinball loss.
"""
import numpy as np
import pytest

from forcester_ts.var_backtest import VaRBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    vol = np.abs(rng.standard_normal(n)) * 0.01 + 0.005
    return returns, vol


# ---------------------------------------------------------------------------
# compute_var
# ---------------------------------------------------------------------------

class TestComputeVar:
    def test_normal_var_positive(self):
        bt = VaRBacktester()
        vol = np.array([0.01, 0.02, 0.03])
        var = bt.compute_var(vol, confidence_level=0.99)
        assert all(v > 0 for v in var), "VaR should be positive"

    def test_normal_var_scales_with_sigma(self):
        bt = VaRBacktester()
        vol1 = np.array([0.01])
        vol2 = np.array([0.02])
        var1 = bt.compute_var(vol1, 0.99)[0]
        var2 = bt.compute_var(vol2, 0.99)[0]
        assert var2 == pytest.approx(2 * var1, rel=1e-6)

    def test_t_dist_larger_var_than_normal(self):
        bt = VaRBacktester()
        vol = np.array([0.01] * 10)
        var_norm = bt.compute_var(vol, 0.99, dist="normal")
        var_t = bt.compute_var(vol, 0.99, dist="t", nu=5)
        assert all(var_t[i] > var_norm[i] for i in range(len(vol))), (
            "t-dist VaR should be larger than normal VaR at same confidence"
        )

    def test_unknown_dist_falls_back_to_normal(self):
        bt = VaRBacktester()
        vol = np.array([0.01])
        var_norm = bt.compute_var(vol, 0.99, dist="normal")
        var_bad = bt.compute_var(vol, 0.99, dist="bogus")
        assert var_bad[0] == pytest.approx(var_norm[0])

    def test_higher_confidence_gives_larger_var(self):
        bt = VaRBacktester()
        vol = np.array([0.01] * 5)
        var_95 = bt.compute_var(vol, 0.95)
        var_99 = bt.compute_var(vol, 0.99)
        assert all(var_99[i] > var_95[i] for i in range(len(vol)))


# ---------------------------------------------------------------------------
# kupiec_test
# ---------------------------------------------------------------------------

class TestKupiecTest:
    def test_no_violations_when_var_is_huge(self):
        bt = VaRBacktester()
        returns = np.full(100, -0.001)  # small losses
        var = np.full(100, 1.0)         # enormous VaR → no violations
        result = bt.kupiec_test(returns, var, 0.99)
        assert result["violations"] == 0
        assert result["violation_rate"] == 0.0
        assert result["T"] == 100

    def test_all_violations_when_var_is_zero(self):
        bt = VaRBacktester()
        returns = np.full(100, -0.01)
        var = np.zeros(100)
        result = bt.kupiec_test(returns, var, 0.99)
        assert result["violations"] == 100
        assert result["violation_rate"] == 1.0

    def test_expected_rate_matches_confidence(self):
        bt = VaRBacktester()
        returns, vol = _make_data()
        var = bt.compute_var(vol, 0.99)
        result = bt.kupiec_test(returns, var, 0.99)
        assert result["expected_rate"] == pytest.approx(0.01)

    def test_result_keys_present(self):
        bt = VaRBacktester()
        returns, vol = _make_data()
        var = bt.compute_var(vol, 0.99)
        result = bt.kupiec_test(returns, var, 0.99)
        for key in ("violations", "T", "violation_rate", "expected_rate",
                    "lr_stat", "p_value", "reject_null"):
            assert key in result, f"Missing key: {key}"

    def test_well_calibrated_var_does_not_reject(self):
        """
        With large N and actual violation rate close to alpha,
        the test should NOT reject H0 (high p-value).
        """
        rng = np.random.default_rng(0)
        n = 10_000
        alpha = 0.01
        returns = rng.standard_normal(n) * 0.01
        # Set VaR so exactly ~1% of returns violate
        from scipy.stats import norm
        var = -norm.ppf(alpha) * 0.01 * np.ones(n)
        bt = VaRBacktester()
        result = bt.kupiec_test(returns, var, 1 - alpha)
        # p-value should be reasonably high (not rejecting) — soft check
        assert result["p_value"] > 1e-4  # not drastically wrong


# ---------------------------------------------------------------------------
# christoffersen_test
# ---------------------------------------------------------------------------

class TestChristoffersenTest:
    def test_result_keys_present(self):
        bt = VaRBacktester()
        returns, vol = _make_data()
        var = bt.compute_var(vol, 0.99)
        result = bt.christoffersen_test(returns, var, 0.99)
        for key in ("n00", "n01", "n10", "n11", "pi01", "pi11",
                    "lr_ind", "lr_cc", "p_value_ind", "p_value_cc",
                    "reject_independence", "reject_coverage"):
            assert key in result, f"Missing key: {key}"

    def test_transition_counts_sum_to_T_minus_1(self):
        bt = VaRBacktester()
        returns, vol = _make_data(n=100)
        var = bt.compute_var(vol, 0.99)
        r = bt.christoffersen_test(returns, var, 0.99)
        assert r["n00"] + r["n01"] + r["n10"] + r["n11"] == 99  # T-1

    def test_insufficient_data_returns_error(self):
        bt = VaRBacktester()
        result = bt.christoffersen_test(np.array([0.01]), np.array([0.02]))
        assert "error" in result

    def test_lr_cc_is_lr_ind_plus_lr_pof(self):
        bt = VaRBacktester()
        returns, vol = _make_data()
        var = bt.compute_var(vol, 0.99)
        r = bt.christoffersen_test(returns, var, 0.99)
        kupiec = bt.kupiec_test(returns, var, 0.99)
        expected_cc = r["lr_ind"] + kupiec["lr_stat"]
        assert r["lr_cc"] == pytest.approx(expected_cc, abs=1e-9)


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_median_loss_is_half_mae(self):
        """At tau=0.5 pinball = 0.5 * MAE."""
        actual = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, -1.0, 0.0])
        bt = VaRBacktester()
        result = bt.pinball_loss(actual, {0.5: q})
        mae = float(np.mean(np.abs(actual - q)))
        assert result[0.5] == pytest.approx(0.5 * mae, abs=1e-10)

    def test_perfect_median_forecast_has_zero_loss(self):
        actual = np.array([1.0, 2.0, 3.0])
        bt = VaRBacktester()
        result = bt.pinball_loss(actual, {0.5: actual.copy()})
        assert result[0.5] == pytest.approx(0.0, abs=1e-12)

    def test_mean_key_present(self):
        actual = np.ones(10)
        q = {0.1: np.zeros(10), 0.9: np.ones(10) * 2}
        bt = VaRBacktester()
        result = bt.pinball_loss(actual, q)
        assert "mean" in result

    def test_multiple_taus(self):
        actual, vol = _make_data(n=100)
        bt = VaRBacktester()
        taus = [0.01, 0.05, 0.50, 0.95, 0.99]
        q_forecasts = {tau: np.zeros(100) for tau in taus}
        result = bt.pinball_loss(actual, q_forecasts)
        assert len(result) == len(taus) + 1  # + "mean"

    def test_empty_quantile_dict_has_only_mean(self):
        bt = VaRBacktester()
        result = bt.pinball_loss(np.ones(10), {})
        assert result == {}

    def test_tau_asymmetry(self):
        """Tau=0.9 penalizes underprediction more than tau=0.1."""
        actual = np.ones(20) * 0.05  # positive, forecast below
        q = np.zeros(20)             # forecast = 0 (always underpredicting)
        bt = VaRBacktester()
        result = bt.pinball_loss(actual, {0.1: q, 0.9: q})
        # tau=0.9 loss = 0.9 * mean(|y - q|); tau=0.1 loss = 0.1 * mean(|y - q|)
        assert result[0.9] > result[0.1]


# ---------------------------------------------------------------------------
# full_report
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_full_report_structure(self):
        returns, vol = _make_data()
        bt = VaRBacktester()
        report = bt.full_report(returns, vol, confidence_levels=(0.95, 0.99),
                                taus=(0.01, 0.50, 0.99))
        assert "confidence_levels" in report
        assert "pinball" in report
        assert 0.95 in report["confidence_levels"]
        assert 0.99 in report["confidence_levels"]
        for cl in (0.95, 0.99):
            assert "kupiec" in report["confidence_levels"][cl]
            assert "christoffersen" in report["confidence_levels"][cl]

    def test_full_report_pinball_mean_present(self):
        returns, vol = _make_data()
        bt = VaRBacktester()
        report = bt.full_report(returns, vol, confidence_levels=(0.99,),
                                taus=(0.01, 0.50, 0.99))
        assert "mean" in report["pinball"]

    def test_full_report_uses_supplied_empirical_quantiles_without_parametric_fallback(self):
        class _NoParametricBacktester(VaRBacktester):
            def compute_var(self, *args, **kwargs):
                raise AssertionError("compute_var should not run when full empirical quantiles are supplied")

        bt = _NoParametricBacktester()
        actual = np.array([-0.01, 0.0, 0.01, -0.005])
        quantiles = {
            0.01: np.full(4, -0.02),
            0.50: np.zeros(4),
            0.99: np.full(4, 0.02),
        }

        report = bt.full_report(
            actual,
            None,
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=quantiles,
        )

        assert report["confidence_levels"][0.99]["source"] == "empirical_quantile"
        assert report["confidence_levels"][0.99]["tau"] == pytest.approx(0.01)
        assert report["pinball_sources"][0.01] == "empirical_quantile"
        assert report["pinball_sources"][0.5] == "empirical_quantile"
        assert report["pinball_sources"][0.99] == "empirical_quantile"

    def test_full_report_reports_mixed_sources_explicitly(self):
        bt = VaRBacktester()
        actual, vol = _make_data(n=16)
        quantiles = {
            0.50: np.zeros(16),
        }

        report = bt.full_report(
            actual,
            vol,
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=quantiles,
        )

        assert report["confidence_levels"][0.99]["source"] == "parametric_var"
        assert report["pinball_sources"][0.01] == "parametric_var"
        assert report["pinball_sources"][0.5] == "empirical_quantile"
        assert report["pinball_sources"][0.99] == "parametric_var"

    def test_full_report_marks_missing_sources_without_silent_fallback(self):
        bt = VaRBacktester()
        actual = np.array([-0.01, 0.0, 0.01, -0.005])
        quantiles = {
            0.50: np.zeros(4),
        }

        report = bt.full_report(
            actual,
            None,
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=quantiles,
        )

        assert report["confidence_levels"][0.99]["source"] == "missing"
        assert "error" in report["confidence_levels"][0.99]["kupiec"]
        assert report["pinball_sources"][0.01] == "missing"
        assert report["pinball_sources"][0.5] == "empirical_quantile"
        assert report["pinball_sources"][0.99] == "missing"
        assert np.isnan(report["pinball"][0.01])
        assert np.isnan(report["pinball"][0.99])

    def test_full_report_decouples_coverage_quantiles_from_pinball_quantiles(self):
        bt = VaRBacktester()
        actual, vol = _make_data(n=16)
        quantiles = {
            0.01: np.full(16, -0.02),
            0.50: np.zeros(16),
            0.99: np.full(16, 0.02),
        }

        report = bt.full_report(
            actual,
            vol,
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=quantiles,
            coverage_quantile_forecasts={},
        )

        assert report["confidence_levels"][0.99]["source"] == "parametric_var"
        assert report["pinball_sources"][0.01] == "empirical_quantile"
        assert report["pinball_sources"][0.5] == "empirical_quantile"
        assert report["pinball_sources"][0.99] == "empirical_quantile"

    def test_full_report_treats_nan_only_quantiles_as_missing(self):
        bt = VaRBacktester()
        actual = np.array([-0.01, 0.0, 0.01, -0.005])
        quantiles = {
            0.01: np.full(4, np.nan),
            0.50: np.zeros(4),
            0.99: np.full(4, np.nan),
        }

        report = bt.full_report(
            actual,
            None,
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=quantiles,
        )

        assert report["confidence_levels"][0.99]["source"] == "missing"
        assert report["pinball_sources"][0.01] == "missing"
        assert report["pinball_sources"][0.5] == "empirical_quantile"
        assert report["pinball_sources"][0.99] == "missing"

    def test_full_report_treats_nan_only_volatility_as_missing(self):
        bt = VaRBacktester()
        actual = np.array([-0.01, 0.0, 0.01, -0.005])

        report = bt.full_report(
            actual,
            np.full(4, np.nan),
            confidence_levels=(0.99,),
            taus=(0.01, 0.50, 0.99),
            quantile_forecasts=None,
        )

        assert report["confidence_levels"][0.99]["source"] == "missing"
        assert report["pinball_sources"][0.01] == "missing"
        assert report["pinball_sources"][0.5] == "missing"
        assert report["pinball_sources"][0.99] == "missing"
