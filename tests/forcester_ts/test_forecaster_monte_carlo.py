from __future__ import annotations

import pandas as pd

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.monte_carlo_simulator import MonteCarloSimulator


def test_monte_carlo_simulator_clamps_path_count_and_returns_summary() -> None:
    simulator = MonteCarloSimulator()
    idx = pd.RangeIndex(1, 4, name="horizon")
    forecast = pd.Series([101.0, 102.0, 103.0], index=idx)
    volatility = pd.Series([0.01, 0.015, 0.02], index=idx)

    result = simulator.simulate_price_distribution(
        base_forecast=forecast,
        last_price=100.0,
        n_paths=10,
        seed=7,
        volatility=volatility,
    )

    assert result["status"] == "OK"
    assert result["alpha"] == 0.05
    assert result["lower_quantile"] == 0.025
    assert result["upper_quantile"] == 0.975
    assert result["paths_used"] == MonteCarloSimulator.MIN_PATHS
    assert result["volatility_source"] == "volatility_forecast"
    assert len(result["expected_path"]) == 3
    assert result["lower_ci"].iloc[0] <= result["median_path"].iloc[0] <= result["upper_ci"].iloc[0]


def test_monte_carlo_simulator_uses_confidence_band_when_volatility_missing() -> None:
    simulator = MonteCarloSimulator()
    idx = pd.RangeIndex(1, 4, name="horizon")
    forecast = pd.Series([101.0, 102.5, 104.0], index=idx)
    lower = pd.Series([99.0, 100.0, 101.0], index=idx)
    upper = pd.Series([103.0, 105.0, 107.0], index=idx)

    result = simulator.simulate_price_distribution(
        base_forecast=forecast,
        last_price=100.0,
        n_paths=300,
        seed=11,
        lower_ci=lower,
        upper_ci=upper,
        alpha=0.05,
    )

    assert result["status"] == "OK"
    assert result["alpha"] == 0.05
    assert result["confidence_level"] == 0.95
    assert result["volatility_source"] == "confidence_interval"
    assert result["paths_used"] == 300


def test_monte_carlo_simulator_uses_requested_alpha_for_empirical_bands() -> None:
    simulator = MonteCarloSimulator()
    idx = pd.RangeIndex(1, 5, name="horizon")
    forecast = pd.Series([101.0, 102.0, 103.0, 104.0], index=idx)
    volatility = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)

    wide = simulator.simulate_price_distribution(
        base_forecast=forecast,
        last_price=100.0,
        n_paths=2000,
        seed=17,
        volatility=volatility,
        alpha=0.10,
    )
    narrow = simulator.simulate_price_distribution(
        base_forecast=forecast,
        last_price=100.0,
        n_paths=2000,
        seed=17,
        volatility=volatility,
        alpha=0.50,
    )

    assert wide["lower_quantile"] == 0.05
    assert wide["upper_quantile"] == 0.95
    assert narrow["lower_quantile"] == 0.25
    assert narrow["upper_quantile"] == 0.75
    assert (narrow["upper_ci"] - narrow["lower_ci"]).iloc[0] < (wide["upper_ci"] - wide["lower_ci"]).iloc[0]


def test_forecaster_forecast_exposes_monte_carlo_summary_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    class _StubSamossa:
        def forecast(self, steps: int):
            idx = pd.RangeIndex(1, steps + 1, name="horizon")
            forecast = pd.Series([101.0, 102.0, 103.0][:steps], index=idx, name="samossa_forecast")
            lower = pd.Series([99.0, 100.0, 101.0][:steps], index=idx, name="samossa_lower_ci")
            upper = pd.Series([103.0, 104.0, 105.0][:steps], index=idx, name="samossa_upper_ci")
            return {"forecast": forecast, "lower_ci": lower, "upper_ci": upper}

        def get_model_summary(self):
            return {"trend_strength": 0.1, "seasonal_strength": 0.0}

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._last_price = 100.0
    forecaster._samossa = _StubSamossa()

    result = forecaster.forecast(steps=3, mc_enabled=True, mc_paths=10, mc_seed=5)

    mc_result = result["monte_carlo"]
    assert mc_result["status"] == "OK"
    assert mc_result["confidence_level"] == 0.95
    assert mc_result["volatility_source"] == "confidence_interval"
    assert mc_result["paths_used"] == MonteCarloSimulator.MIN_PATHS
    assert result["mean_forecast"] is result["samossa_forecast"]


def test_forecaster_forecast_monte_carlo_reports_skip_when_mean_forecast_missing(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._last_price = 100.0

    result = forecaster.forecast(steps=3, mc_enabled=True)

    assert result["monte_carlo"]["status"] == "SKIP"
    assert result["monte_carlo"]["reason"] == "base_forecast_missing"


def test_forecaster_forecast_without_fit_keeps_static_regime(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)

    result = forecaster.forecast(steps=3)

    assert result["mean_forecast"] is None
    assert result["regime"] == "STATIC"
    assert result["regime_confidence"] is None


def test_forecaster_forecast_monte_carlo_reports_skip_when_last_price_missing(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    class _StubSamossa:
        def forecast(self, steps: int):
            idx = pd.RangeIndex(1, steps + 1, name="horizon")
            forecast = pd.Series([101.0, 102.0, 103.0][:steps], index=idx, name="samossa_forecast")
            lower = pd.Series([99.0, 100.0, 101.0][:steps], index=idx, name="samossa_lower_ci")
            upper = pd.Series([103.0, 104.0, 105.0][:steps], index=idx, name="samossa_upper_ci")
            return {"forecast": forecast, "lower_ci": lower, "upper_ci": upper}

        def get_model_summary(self):
            return {"trend_strength": 0.1, "seasonal_strength": 0.0}

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._samossa = _StubSamossa()

    result = forecaster.forecast(steps=3, mc_enabled=True)

    assert result["monte_carlo"]["status"] == "SKIP"
    assert result["monte_carlo"]["reason"] == "last_price_missing"


def test_forecaster_forecast_uses_monte_carlo_config_defaults_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    class _StubSamossa:
        def forecast(self, steps: int):
            idx = pd.RangeIndex(1, steps + 1, name="horizon")
            forecast = pd.Series([101.0, 102.0, 103.0][:steps], index=idx, name="samossa_forecast")
            lower = pd.Series([99.0, 100.0, 101.0][:steps], index=idx, name="samossa_lower_ci")
            upper = pd.Series([103.0, 104.0, 105.0][:steps], index=idx, name="samossa_upper_ci")
            return {"forecast": forecast, "lower_ci": lower, "upper_ci": upper}

        def get_model_summary(self):
            return {"trend_strength": 0.1, "seasonal_strength": 0.0}

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        monte_carlo_config={"enabled": True, "paths": 10, "seed": 0},
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._last_price = 100.0
    forecaster._samossa = _StubSamossa()

    result = forecaster.forecast(steps=3)

    mc_result = result["monte_carlo"]
    assert mc_result["status"] == "OK"
    assert mc_result["seed"] == 0
    assert mc_result["paths_requested"] == 10
    assert mc_result["paths_used"] == MonteCarloSimulator.MIN_PATHS


def test_forecaster_forecast_explicit_mc_flag_disables_config_default(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")

    class _StubSamossa:
        def forecast(self, steps: int):
            idx = pd.RangeIndex(1, steps + 1, name="horizon")
            forecast = pd.Series([101.0, 102.0, 103.0][:steps], index=idx, name="samossa_forecast")
            lower = pd.Series([99.0, 100.0, 101.0][:steps], index=idx, name="samossa_lower_ci")
            upper = pd.Series([103.0, 104.0, 105.0][:steps], index=idx, name="samossa_upper_ci")
            return {"forecast": forecast, "lower_ci": lower, "upper_ci": upper}

        def get_model_summary(self):
            return {"trend_strength": 0.1, "seasonal_strength": 0.0}

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        monte_carlo_config={"enabled": True, "paths": 500},
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._last_price = 100.0
    forecaster._samossa = _StubSamossa()

    result = forecaster.forecast(steps=3, mc_enabled=False)

    assert "monte_carlo" not in result


def test_resolve_ticker_ignores_generic_series_names() -> None:
    generic_price = pd.Series([100.0, 101.0], name="Close")
    generic_returns = pd.Series([0.01], name="returns")
    named_price = pd.Series([100.0, 101.0], name="AAPL")

    assert TimeSeriesForecaster._resolve_ticker("", generic_price, generic_returns) == ""
    assert TimeSeriesForecaster._resolve_ticker("", named_price, None) == "AAPL"
