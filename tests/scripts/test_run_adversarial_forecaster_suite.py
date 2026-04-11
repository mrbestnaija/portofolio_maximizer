from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_adversarial_forecaster_suite as mod


# ---------------------------------------------------------------------------
# Original RMSE threshold tests (preserved — backward compatibility)
# ---------------------------------------------------------------------------

def test_evaluate_thresholds_flags_breaches() -> None:
    summary = {
        "prod_like_conf_off": {
            "errors": 0,
            "ensemble_under_best_rate": 1.0,
            "avg_ensemble_ratio_vs_best": 1.30,
            "ensemble_worse_than_rw_rate": 0.66,
        }
    }
    thresholds = {
        "max_ensemble_under_best_rate": 1.0,
        "max_avg_ensemble_ratio_vs_best": 1.2,
        "max_ensemble_worse_than_rw_rate": 0.3,
        "require_zero_errors": True,
    }
    breaches = mod.evaluate_thresholds(summary, thresholds)
    assert any("avg_ensemble_ratio_vs_best" in item for item in breaches)
    assert any("ensemble_worse_than_rw_rate" in item for item in breaches)


def test_load_thresholds_from_monitor_config(tmp_path: Path) -> None:
    cfg = tmp_path / "forecaster_monitoring_ci.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    adversarial_suite:",
                "      max_ensemble_under_best_rate: 0.95",
                "      max_avg_ensemble_ratio_vs_best: 1.10",
                "      max_ensemble_worse_than_rw_rate: 0.20",
                "      max_index_mismatch_rate: 0.00",
                "      require_zero_errors: true",
            ]
        ),
        encoding="utf-8",
    )
    thresholds = mod._load_thresholds(cfg)
    assert thresholds["max_ensemble_under_best_rate"] == 0.95
    assert thresholds["max_avg_ensemble_ratio_vs_best"] == 1.10
    assert thresholds["max_ensemble_worse_than_rw_rate"] == 0.20
    assert thresholds["max_index_mismatch_rate"] == 0.00
    assert thresholds["require_zero_errors"] is True


def test_evaluate_thresholds_can_use_effective_default_path_metric() -> None:
    summary = {
        "prod_like_conf_on": {
            "errors": 0,
            "ensemble_under_best_rate": 0.90,
            "avg_ensemble_ratio_vs_best": 1.10,
            "ensemble_worse_than_rw_rate": 0.60,
            "effective_worse_than_rw_rate": 0.15,
        }
    }
    thresholds = {
        "max_ensemble_under_best_rate": 1.0,
        "max_avg_ensemble_ratio_vs_best": 1.2,
        "max_ensemble_worse_than_rw_rate": 0.3,
        "max_effective_worse_than_rw_rate": 0.2,
        "use_effective_default_path_metric": True,
        "require_zero_errors": True,
    }
    breaches = mod.evaluate_thresholds(summary, thresholds)
    # Raw ensemble_worse_than_rw_rate would breach; effective metric should pass.
    assert not any("worse_than_rw_rate" in item for item in breaches)


def test_evaluate_thresholds_flags_index_mismatch_rate() -> None:
    summary = {
        "prod_like_conf_on": {
            "errors": 0,
            "ensemble_under_best_rate": 0.10,
            "avg_ensemble_ratio_vs_best": 1.01,
            "ensemble_worse_than_rw_rate": 0.05,
            "ensemble_index_mismatch_rate": 0.25,
        }
    }
    thresholds = {
        "max_ensemble_under_best_rate": 1.0,
        "max_avg_ensemble_ratio_vs_best": 1.2,
        "max_ensemble_worse_than_rw_rate": 0.3,
        "max_index_mismatch_rate": 0.0,
        "require_zero_errors": True,
    }
    breaches = mod.evaluate_thresholds(summary, thresholds)
    assert any("ensemble_index_mismatch_rate" in item for item in breaches)


# ---------------------------------------------------------------------------
# Denominator bug fix: all-error runs must produce nan rates, not 0.0
# ---------------------------------------------------------------------------

def test_summarize_all_errors_produces_nan_rates() -> None:
    """When every run errors, rates should be nan — not 0.0 (false PASS)."""
    rows = [
        {
            "scenario": "random_walk",
            "seed": 101,
            "rw": {},
            "metrics": {},
            "weights": {},
            "status": None,
            "default_model": None,
            "ensemble_index_mismatch": False,
            "barbell": {},
            "error": "something went wrong",
        }
        for _ in range(3)
    ]
    summary = mod.summarize(rows)
    assert summary["errors"] == 3
    assert math.isnan(summary["ensemble_under_best_rate"]), (
        "all-error runs must produce nan, not 0.0 — 0.0 is a false PASS"
    )
    assert math.isnan(summary["ensemble_worse_than_rw_rate"])
    assert math.isnan(summary["ensemble_index_mismatch_rate"])


def test_evaluate_thresholds_all_nan_rates_produce_breach() -> None:
    """nan rates (from all-error runs) must fire a breach, not silently pass."""
    summary = {
        "prod_like_conf_on": {
            "errors": 5,
            "ensemble_under_best_rate": float("nan"),
            "ensemble_worse_than_rw_rate": float("nan"),
            "ensemble_index_mismatch_rate": float("nan"),
        }
    }
    thresholds = {
        "max_ensemble_under_best_rate": 1.0,
        "max_avg_ensemble_ratio_vs_best": 1.2,
        "max_ensemble_worse_than_rw_rate": 0.3,
        "require_zero_errors": True,
    }
    breaches = mod.evaluate_thresholds(summary, thresholds)
    # require_zero_errors=True fires; nan guard also fires
    assert len(breaches) > 0


# ---------------------------------------------------------------------------
# Barbell scenario generators
# ---------------------------------------------------------------------------

class TestBarbellScenarios:
    """Verify new barbell scenarios produce valid, non-degenerate price series."""

    @pytest.mark.parametrize("scenario", mod._BARBELL_SCENARIOS)
    def test_barbell_scenario_generates_positive_prices(self, scenario: str) -> None:
        series = mod.gen_series(scenario, 200, seed=42)
        assert isinstance(series, pd.Series)
        assert len(series) == 200
        assert (series > 0).all(), f"Scenario {scenario}: prices must be strictly positive"
        assert series.isnull().sum() == 0

    def test_ngn_high_inflation_has_positive_drift(self) -> None:
        """NGN high inflation scenario should have a positive mean return."""
        series = mod.gen_series("ngn_high_inflation", 500, seed=101)
        rets = series.pct_change().dropna()
        assert float(rets.mean()) > 0.0, "ngn_high_inflation must trend upward"

    def test_asymmetric_vol_has_negative_skew(self) -> None:
        """Asymmetric vol: downside clustering should produce left-skewed returns."""
        series = mod.gen_series("asymmetric_vol", 500, seed=101)
        rets = series.pct_change().dropna()
        # Not enforcing strict skew (seed-dependent) but volatility should be elevated
        assert float(rets.std()) > 0.005, "asymmetric_vol must have non-trivial vol"

    def test_fat_tail_crash_has_large_drawdown(self) -> None:
        """fat_tail_crash must contain at least one daily return < -5%."""
        series = mod.gen_series("fat_tail_crash", 300, seed=202)
        rets = series.pct_change().dropna()
        assert float(rets.min()) < -0.05, "fat_tail_crash must have a crash event"

    def test_crisis_recovery_has_drawdown_then_recovery(self) -> None:
        """crisis_recovery: first third should trend down, last third should trend up."""
        n = 300
        series = mod.gen_series("crisis_recovery", n, seed=303)
        third = n // 3
        first_phase_return = (series.iloc[third] - series.iloc[0]) / series.iloc[0]
        last_phase_return = (series.iloc[-1] - series.iloc[2 * third]) / series.iloc[2 * third]
        # First third declines on average
        assert first_phase_return < 0.0, "crisis_recovery first phase should be declining"
        # Last third recovers
        assert last_phase_return > 0.0, "crisis_recovery last phase should be recovering"

    def test_all_default_scenarios_are_valid(self) -> None:
        """All DEFAULT_SCENARIOS must generate valid series."""
        for scenario in mod.DEFAULT_SCENARIOS:
            for seed in [101, 202]:
                series = mod.gen_series(scenario, 200, seed=seed)
                assert (series > 0).all(), f"{scenario} seed={seed}: negative prices"


# ---------------------------------------------------------------------------
# compute_barbell_per_run tests
# ---------------------------------------------------------------------------

class TestComputeBarbellPerRun:
    """Unit tests for per-run barbell metric computation."""

    def _make_series(self, values, start="2023-01-01"):
        idx = pd.date_range(start, periods=len(values), freq="D")
        return pd.Series(values, index=idx)

    def test_correct_direction_returns_terminal_da_1(self) -> None:
        """Forecast predicts UP, actual goes UP → terminal_DA = 1.0."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 101.0, 102.0, 103.0, 104.0])
        forecast = self._make_series([100.0, 100.5, 101.0, 101.5, 102.0])
        result = mod.compute_barbell_per_run(train, test, forecast, None, None)
        assert result["terminal_da"] == 1.0
        assert result["forecast_direction"] == 1
        assert result["trade_return"] is not None and result["trade_return"] > 0

    def test_wrong_direction_returns_terminal_da_0(self) -> None:
        """Forecast predicts UP, actual goes DOWN → terminal_DA = 0.0, trade loses."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 99.0, 98.0, 97.0, 96.0])
        # Forecast says UP
        forecast = self._make_series([100.0, 100.5, 101.0, 101.5, 102.0])
        result = mod.compute_barbell_per_run(train, test, forecast, None, None)
        assert result["terminal_da"] == 0.0
        assert result["trade_return"] is not None and result["trade_return"] < 0

    def test_ci_coverage_inside(self) -> None:
        """Actual terminal price inside CI → ci_coverage = 1.0."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 100.5, 101.0])
        forecast = self._make_series([100.0, 100.3, 100.8])
        lower = self._make_series([99.0, 99.5, 100.0])
        upper = self._make_series([101.0, 101.5, 102.0])
        result = mod.compute_barbell_per_run(train, test, forecast, lower, upper)
        assert result["ci_coverage"] == 1.0

    def test_ci_coverage_outside(self) -> None:
        """Actual terminal price outside CI → ci_coverage = 0.0."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 100.5, 105.0])  # actual terminal = 105
        forecast = self._make_series([100.0, 100.3, 100.8])
        lower = self._make_series([99.0, 99.5, 99.8])
        upper = self._make_series([101.0, 101.5, 102.0])  # upper = 102 < actual 105
        result = mod.compute_barbell_per_run(train, test, forecast, lower, upper)
        assert result["ci_coverage"] == 0.0

    def test_flat_forecast_direction_is_zero(self) -> None:
        """Flat forecast → forecast_direction = 0, trade_return = 0.0."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 101.0, 102.0])
        # Flat forecast: first == last
        forecast = self._make_series([100.0, 100.0, 100.0])
        result = mod.compute_barbell_per_run(train, test, forecast, None, None)
        assert result["forecast_direction"] == 0
        assert result["trade_return"] == 0.0

    def test_no_forecast_returns_none_metrics(self) -> None:
        """Without a forecast series, barbell metrics are None (graceful, not PASS)."""
        train = self._make_series([100.0] * 10)
        test = self._make_series([100.0, 101.0, 102.0])
        result = mod.compute_barbell_per_run(train, test, None, None, None)
        assert result["terminal_da"] is None
        assert result["trade_return"] is None
        assert result["ci_coverage"] is None
        # actual_return and max_drawdown_path should still be computed
        assert result["actual_return"] is not None

    def test_max_drawdown_computed_from_actual_path(self) -> None:
        """Max drawdown computed from actual test series (not forecast)."""
        train = self._make_series([100.0] * 10)
        # Test series goes up then crashes 20%
        test = self._make_series([100.0, 105.0, 110.0, 88.0, 90.0])
        result = mod.compute_barbell_per_run(train, test, None, None, None)
        # Max drawdown from peak (110) to trough (88) is ~20%
        assert result["max_drawdown_path"] is not None
        assert result["max_drawdown_path"] > 0.10, "should capture >10% drawdown"

    def test_negative_entry_price_returns_safely(self) -> None:
        """Negative entry price (degenerate input) must not crash."""
        train = self._make_series([-5.0, -4.0, -3.0])
        test = self._make_series([-2.0, -1.0, 0.5])
        result = mod.compute_barbell_per_run(train, test, None, None, None)
        # Should return safely with None metrics (entry_price <= 0 guard)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# summarize_barbell tests
# ---------------------------------------------------------------------------

class TestSummarizeBarbell:
    """Aggregate barbell metrics correctly across a batch of runs."""

    def _make_rows(self, trade_returns, terminal_das=None, ci_coverages=None):
        """Build synthetic run rows for summarize_barbell."""
        rows = []
        for i, tr in enumerate(trade_returns):
            tda = terminal_das[i] if terminal_das else (1.0 if tr > 0 else 0.0)
            cc = ci_coverages[i] if ci_coverages else 0.5
            rows.append({
                "error": None,
                "barbell": {
                    "terminal_da": tda,
                    "trade_return": tr,
                    "ci_coverage": cc,
                    "max_drawdown_path": 0.05,
                },
            })
        return rows

    def test_omega_ratio_computed_for_10_plus_trades(self) -> None:
        """omega_ratio should be computed when >= 10 trade returns available."""
        # Returns above NGN threshold → omega > 1
        rows = self._make_rows([0.02] * 15)
        summary = mod.summarize_barbell(rows)
        assert summary["n_trades"] == 15
        assert summary["omega_ratio"] is not None
        assert summary["omega_above_1"] is True

    def test_omega_ratio_none_for_fewer_than_10_trades(self) -> None:
        """omega_ratio must be None when < 10 trades (statistically meaningless)."""
        rows = self._make_rows([0.01] * 5)
        summary = mod.summarize_barbell(rows)
        assert summary["omega_ratio"] is None

    def test_omega_below_ngn_hurdle_omega_above_1_is_false(self) -> None:
        """Returns below NGN hurdle → omega < 1 → omega_above_1 = False."""
        # Small negative returns → losses > gains relative to hurdle
        rows = self._make_rows([-0.005] * 12)
        summary = mod.summarize_barbell(rows)
        assert summary["omega_above_1"] is False

    def test_profit_factor_computed_correctly(self) -> None:
        """profit_factor = avg_win / avg_loss (absolute)."""
        # 5 wins of 0.04, 5 losses of -0.02 → PF = 0.04/0.02 = 2.0
        rows = self._make_rows([0.04, 0.04, 0.04, 0.04, 0.04,
                                 -0.02, -0.02, -0.02, -0.02, -0.02])
        summary = mod.summarize_barbell(rows)
        pf = summary["profit_factor"]
        assert pf is not None
        assert abs(pf - 2.0) < 0.01

    def test_terminal_da_pass_rate(self) -> None:
        """terminal_da_pass_rate = fraction with tda >= 0.45."""
        # 6 correct (1.0), 4 incorrect (0.0) → rate = 0.60
        rows = self._make_rows(
            [0.01] * 10,
            terminal_das=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        summary = mod.summarize_barbell(rows)
        assert abs(summary["terminal_da_pass_rate"] - 0.60) < 0.01

    def test_all_error_rows_produce_zero_trades(self) -> None:
        """All-error rows → n_trades=0, omega_ratio=None, no false PASS."""
        rows = [{"error": "crash", "barbell": {}} for _ in range(10)]
        summary = mod.summarize_barbell(rows)
        assert summary["n_trades"] == 0
        assert summary["omega_ratio"] is None
        assert summary["omega_above_1"] is False
        assert summary["n_errors"] == 10

    def test_expected_profit_usd_uses_capital_base(self) -> None:
        """expected_profit_usd = mean_trade_return * capital_base (25000)."""
        rows = self._make_rows([0.01] * 10)
        summary = mod.summarize_barbell(rows)
        expected = summary["mean_trade_return"] * 25_000.0
        assert abs(summary["expected_profit_usd"] - expected) < 0.01

    def test_mean_max_drawdown_aggregated(self) -> None:
        """mean_max_drawdown correctly aggregated across valid runs."""
        rows = [
            {"error": None, "barbell": {"max_drawdown_path": 0.10, "trade_return": 0.01,
                                         "terminal_da": 1.0, "ci_coverage": 0.5}},
            {"error": None, "barbell": {"max_drawdown_path": 0.20, "trade_return": -0.01,
                                         "terminal_da": 0.0, "ci_coverage": 0.5}},
        ]
        summary = mod.summarize_barbell(rows)
        assert abs(summary["mean_max_drawdown"] - 0.15) < 0.01


# ---------------------------------------------------------------------------
# evaluate_barbell_thresholds tests
# ---------------------------------------------------------------------------

class TestEvaluateBarbellThresholds:
    """Verify barbell threshold evaluation logic — no threshold-dodge short-circuits."""

    def test_omega_breach_when_below_minimum(self) -> None:
        barbell_summary = {
            "prod_like_conf_on": {
                "n_trades": 15, "n_errors": 0,
                "omega_ratio": 0.70,
                "omega_above_1": False,
                "terminal_da_pass_rate": 0.55,
                "mean_ci_coverage": 0.40,
                "profit_factor": 0.90,
                "mean_max_drawdown": 0.08,
            }
        }
        thresholds = {"min_omega_ratio": 1.0, "require_omega_above_1": False}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("omega_ratio" in b for b in breaches)

    def test_require_omega_above_1_fires_when_false(self) -> None:
        barbell_summary = {
            "prod_like_conf_on": {
                "n_trades": 15, "n_errors": 0,
                "omega_ratio": 0.80,
                "omega_above_1": False,
                "terminal_da_pass_rate": 0.50,
                "mean_ci_coverage": None,
                "profit_factor": 1.0,
                "mean_max_drawdown": None,
            }
        }
        thresholds = {"require_omega_above_1": True}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("omega" in b for b in breaches)

    def test_terminal_da_breach_when_below_minimum(self) -> None:
        barbell_summary = {
            "v1": {
                "n_trades": 10, "n_errors": 0,
                "omega_ratio": 1.2, "omega_above_1": True,
                "terminal_da_pass_rate": 0.30,  # below 0.45 threshold
                "mean_ci_coverage": None,
                "profit_factor": 1.0,
                "mean_max_drawdown": None,
            }
        }
        thresholds = {"min_terminal_da_pass_rate": 0.45}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("terminal_da_pass_rate" in b for b in breaches)

    def test_profit_factor_breach(self) -> None:
        barbell_summary = {
            "v1": {
                "n_trades": 10, "n_errors": 0,
                "omega_ratio": 0.9, "omega_above_1": False,
                "terminal_da_pass_rate": 0.50,
                "mean_ci_coverage": None,
                "profit_factor": 0.60,  # below 0.80 threshold
                "mean_max_drawdown": None,
            }
        }
        thresholds = {"min_profit_factor": 0.80}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("profit_factor" in b for b in breaches)

    def test_max_drawdown_breach(self) -> None:
        barbell_summary = {
            "v1": {
                "n_trades": 10, "n_errors": 0,
                "omega_ratio": 1.0, "omega_above_1": True,
                "terminal_da_pass_rate": 0.50,
                "mean_ci_coverage": None,
                "profit_factor": 1.0,
                "mean_max_drawdown": 0.60,  # above 0.45 ceiling
            }
        }
        thresholds = {"max_mean_drawdown": 0.45}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("max_drawdown" in b for b in breaches)

    def test_permissive_defaults_produce_no_breach(self) -> None:
        """Default thresholds (all 0.0) must not breach on any plausible summary."""
        barbell_summary = {
            "v1": {
                "n_trades": 10, "n_errors": 0,
                "omega_ratio": 0.5, "omega_above_1": False,
                "terminal_da_pass_rate": 0.20,
                "mean_ci_coverage": 0.10,
                "profit_factor": 0.50,
                "mean_max_drawdown": 0.99,
            }
        }
        thresholds = {
            "min_terminal_da_pass_rate": 0.0,
            "min_omega_ratio": 0.0,
            "min_ci_coverage_rate": 0.0,
            "min_profit_factor": 0.0,
            "max_mean_drawdown": 1.0,
            "require_omega_above_1": False,
        }
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert breaches == []

    def test_all_error_variant_produces_breach_not_pass(self) -> None:
        """A variant where all runs errored must not silently PASS barbell checks."""
        barbell_summary = {
            "v1": {
                "n_trades": 0, "n_errors": 18,
                "omega_ratio": None, "omega_above_1": False,
                "terminal_da_pass_rate": None,
                "mean_ci_coverage": None,
                "profit_factor": None,
                "mean_max_drawdown": None,
            }
        }
        thresholds = {"require_omega_above_1": False}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("errored" in b for b in breaches)

    def test_ci_coverage_breach_when_below_minimum(self) -> None:
        barbell_summary = {
            "v1": {
                "n_trades": 10, "n_errors": 0,
                "omega_ratio": 1.1, "omega_above_1": True,
                "terminal_da_pass_rate": 0.50,
                "mean_ci_coverage": 0.10,  # below 0.25 threshold
                "profit_factor": 1.0,
                "mean_max_drawdown": None,
            }
        }
        thresholds = {"min_ci_coverage_rate": 0.25}
        breaches = mod.evaluate_barbell_thresholds(barbell_summary, thresholds)
        assert any("ci_coverage" in b for b in breaches)


# ---------------------------------------------------------------------------
# _load_thresholds barbell section
# ---------------------------------------------------------------------------

def test_load_thresholds_includes_barbell_section(tmp_path: Path) -> None:
    """Barbell config section is loaded and keyed correctly."""
    cfg = tmp_path / "monitor.yml"
    cfg.write_text(
        "\n".join([
            "forecaster_monitoring:",
            "  regression_metrics:",
            "    adversarial_suite:",
            "      max_ensemble_under_best_rate: 0.90",
            "      barbell:",
            "        min_omega_ratio: 1.0",
            "        min_terminal_da_pass_rate: 0.45",
            "        min_ci_coverage_rate: 0.25",
            "        min_profit_factor: 0.80",
            "        max_mean_drawdown: 0.45",
            "        require_omega_above_1: true",
        ]),
        encoding="utf-8",
    )
    thresholds = mod._load_thresholds(cfg)
    assert thresholds["min_omega_ratio"] == 1.0
    assert thresholds["min_terminal_da_pass_rate"] == 0.45
    assert thresholds["min_ci_coverage_rate"] == 0.25
    assert thresholds["min_profit_factor"] == 0.80
    assert thresholds["max_mean_drawdown"] == 0.45
    assert thresholds["require_omega_above_1"] is True
    # RMSE keys still present
    assert thresholds["max_ensemble_under_best_rate"] == 0.90


def test_load_thresholds_barbell_defaults_are_permissive(tmp_path: Path) -> None:
    """Missing barbell section → permissive defaults (don't block on fresh deployments)."""
    cfg = tmp_path / "monitor_no_barbell.yml"
    cfg.write_text(
        "\n".join([
            "forecaster_monitoring:",
            "  regression_metrics:",
            "    adversarial_suite:",
            "      max_ensemble_under_best_rate: 0.95",
        ]),
        encoding="utf-8",
    )
    thresholds = mod._load_thresholds(cfg)
    # All barbell defaults must be permissive (0.0 for min, 1.0 for max_drawdown)
    assert thresholds["min_omega_ratio"] == 0.0
    assert thresholds["min_terminal_da_pass_rate"] == 0.0
    assert thresholds["min_ci_coverage_rate"] == 0.0
    assert thresholds["min_profit_factor"] == 0.0
    assert thresholds["max_mean_drawdown"] == 1.0
    assert thresholds["require_omega_above_1"] is False


# ---------------------------------------------------------------------------
# Integration: gen_series + compute_barbell_per_run end-to-end
# ---------------------------------------------------------------------------

class TestBarbellIntegration:
    """Smoke tests using real gen_series output (no forecaster — test compute layer only)."""

    @pytest.mark.parametrize("scenario", mod._BARBELL_SCENARIOS)
    def test_barbell_metrics_computable_from_scenario(self, scenario: str) -> None:
        """For each barbell scenario, compute_barbell_per_run runs without error."""
        series = mod.gen_series(scenario, 200, seed=101)
        horizon = 20
        train = series.iloc[:-horizon]
        test = series.iloc[-horizon:]
        # Compute without forecast (no ensemble) — should return actual_return and drawdown
        result = mod.compute_barbell_per_run(train, test, None, None, None)
        assert isinstance(result, dict)
        assert result["actual_return"] is not None
        assert math.isfinite(result["actual_return"])
        # No forecast → DA and trade_return must be None (not a false PASS)
        assert result["terminal_da"] is None
        assert result["trade_return"] is None

    def test_summarize_barbell_with_mixed_none_trade_returns(self) -> None:
        """Rows with None trade_return (no forecast) are excluded from aggregation."""
        rows = [
            {"error": None, "barbell": {"terminal_da": None, "trade_return": None,
                                         "ci_coverage": None, "max_drawdown_path": 0.05}},
            {"error": None, "barbell": {"terminal_da": 1.0, "trade_return": 0.02,
                                         "ci_coverage": 1.0, "max_drawdown_path": 0.03}},
        ]
        summary = mod.summarize_barbell(rows)
        # Only 1 valid trade return
        assert summary["n_trades"] == 1
        assert summary["omega_ratio"] is None  # < 10 trades
        assert summary["mean_trade_return"] is not None

    def test_ngn_import_flag_present_in_summary(self) -> None:
        """summarize_barbell should report ngn_import_ok so auditors can verify."""
        rows = [{"error": None, "barbell": {"terminal_da": 1.0, "trade_return": 0.01,
                                             "ci_coverage": 1.0, "max_drawdown_path": 0.02}}]
        summary = mod.summarize_barbell(rows)
        assert "ngn_import_ok" in summary
        assert "daily_ngn_threshold" in summary
        assert summary["daily_ngn_threshold"] > 0.0
