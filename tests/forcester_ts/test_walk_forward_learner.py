"""
tests/forcester_ts/test_walk_forward_learner.py
-------------------------------------------------
Integration tests for WalkForwardLearner — rolling/expanding window harness.
"""
import numpy as np
import pandas as pd
import pytest

from forcester_ts.walk_forward_learner import WalkForwardLearner, WalkForwardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_series():
    """300-bar synthetic price series."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(300) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2022-01-01", periods=300)
    return pd.Series(prices, index=idx)


@pytest.fixture
def short_series():
    """Short series for edge-case tests."""
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.standard_normal(50))
    return pd.Series(prices)


# ---------------------------------------------------------------------------
# Basic harness behavior
# ---------------------------------------------------------------------------

class TestBasicHarness:
    def test_run_returns_walk_forward_result(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        assert isinstance(result, WalkForwardResult)

    def test_has_multiple_folds(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        assert result.n_folds >= 2

    def test_fold_indices_are_sequential(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        for i, fold in enumerate(result.fold_metrics):
            assert fold.fold_idx == i

    def test_train_end_advances_by_fold_step(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=10,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        if len(result.fold_metrics) >= 2:
            delta = result.fold_metrics[1].train_end - result.fold_metrics[0].train_end
            assert delta == 10

    def test_all_folds_have_rmse(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        for fold in result.fold_metrics:
            assert fold.rmse == fold.rmse  # not NaN

    def test_dir_acc_in_valid_range(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        for fold in result.fold_metrics:
            assert 0.0 <= fold.dir_acc <= 1.0

    def test_ticker_stored_in_result(self, price_series):
        wfl = WalkForwardLearner({})
        result = wfl.run(price_series, ticker="AAPL")
        assert result.ticker == "AAPL"


# ---------------------------------------------------------------------------
# Expanding vs rolling window
# ---------------------------------------------------------------------------

class TestWindowTypes:
    def test_expanding_window_default(self, price_series):
        wfl = WalkForwardLearner({}, window_type="expanding",
                                 min_train_length=100, fold_step=20,
                                 forecast_horizon=5)
        result = wfl.run(price_series, ticker="TEST")
        assert result.n_folds > 0

    def test_rolling_window(self, price_series):
        wfl = WalkForwardLearner({}, window_type="rolling",
                                 min_train_length=100, fold_step=20,
                                 forecast_horizon=5)
        result = wfl.run(price_series, ticker="TEST")
        assert result.n_folds > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_series_too_short_returns_empty(self, short_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=10,
                                 forecast_horizon=10)
        result = wfl.run(short_series, ticker="SHORT")
        assert result.n_folds == 0
        assert result.fold_metrics == []

    def test_numpy_array_series(self):
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.standard_normal(200))
        wfl = WalkForwardLearner({}, min_train_length=60, fold_step=20,
                                 forecast_horizon=5)
        # pandas Series wrapping numpy
        result = wfl.run(pd.Series(prices), ticker="NP")
        assert result.n_folds >= 1


# ---------------------------------------------------------------------------
# VaR metrics
# ---------------------------------------------------------------------------

class TestVaRMetrics:
    def test_var_violation_rate_in_valid_range(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10, confidence_level=0.99)
        result = wfl.run(price_series, ticker="TEST")
        for fold in result.fold_metrics:
            vr = fold.var_violation_rate
            if vr == vr:  # not NaN
                assert 0.0 <= vr <= 1.0

    def test_pinball_loss_keys_present(self, price_series):
        taus = (0.1, 0.5, 0.9)
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10, taus=taus)
        result = wfl.run(price_series, ticker="TEST")
        for fold in result.fold_metrics:
            for tau in taus:
                assert tau in fold.pinball_loss, f"Missing tau={tau} in fold {fold.fold_idx}"

    def test_var_and_pinball_sources_are_explicit_and_decoupled(self, price_series):
        taus = (0.01, 0.5, 0.99)
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10, confidence_level=0.99, taus=taus)
        result = wfl.run(price_series, ticker="TEST")
        for fold in result.fold_metrics:
            assert fold.var_source == "parametric_var"
            assert fold.pinball_sources[0.01] == "empirical_quantile"
            assert fold.pinball_sources[0.5] == "empirical_quantile"
            assert fold.pinball_sources[0.99] == "empirical_quantile"


# ---------------------------------------------------------------------------
# Shapley attribution
# ---------------------------------------------------------------------------

class TestShapleyInFolds:
    def test_shapley_keys_present(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        expected_models = {"garch", "samossa", "mssa_rl"}
        for fold in result.fold_metrics:
            assert set(fold.shapley.keys()) == expected_models

    def test_aggregate_shapley_mean_in_aggregate(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        assert "shapley_mean" in result.aggregate
        mean_sv = result.aggregate["shapley_mean"]
        assert isinstance(mean_sv, dict)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_aggregate_keys_present(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        for key in ("n_folds", "rmse_mean", "rmse_std", "mae_mean",
                    "dir_acc_mean", "var_violation_rate_mean",
                    "kupiec_p_value_mean", "shapley_mean"):
            assert key in result.aggregate, f"Missing key: {key}"

    def test_aggregate_n_folds_matches(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        assert result.aggregate["n_folds"] == result.n_folds

    def test_rmse_mean_is_positive(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST")
        assert result.aggregate["rmse_mean"] > 0


# ---------------------------------------------------------------------------
# Regime sequence integration
# ---------------------------------------------------------------------------

class TestRegimeSequence:
    def test_regime_stored_in_fold(self, price_series):
        regimes = ["TRENDING"] * 150 + ["CRISIS"] * 150
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST",
                         regime_sequence=regimes)
        # At least some folds should have a regime
        regime_vals = [f.regime for f in result.fold_metrics]
        assert any(r is not None for r in regime_vals)

    def test_no_regime_sequence_gives_none_regime(self, price_series):
        wfl = WalkForwardLearner({}, min_train_length=120, fold_step=20,
                                 forecast_horizon=10)
        result = wfl.run(price_series, ticker="TEST", regime_sequence=None)
        for fold in result.fold_metrics:
            assert fold.regime is None


class TestOrderLearnerIntegration:
    def test_order_used_populated_from_order_cache(self, price_series):
        class _StubLearner:
            def suggest(self, ticker, model_type, regime):
                if model_type == "GARCH":
                    return {"p": 1, "q": 1}
                return None

        wfl = WalkForwardLearner(
            {},
            order_learner=_StubLearner(),
            min_train_length=120,
            fold_step=20,
            forecast_horizon=10,
        )
        result = wfl.run(price_series, ticker="TEST")
        assert result.fold_metrics
        assert all(fold.order_used.get("garch") == {"p": 1, "q": 1} for fold in result.fold_metrics)
