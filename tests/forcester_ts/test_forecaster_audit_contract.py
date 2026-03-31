from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig


def _minimal_forecaster() -> TimeSeriesForecaster:
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    return TimeSeriesForecaster(config=config)


def test_fit_records_dataset_ticker_metadata(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    forecaster = _minimal_forecaster()
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    price_series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index, name="Close")

    forecaster.fit(price_series, ticker="MSFT")

    report = forecaster.get_instrumentation_report()
    assert report["dataset"]["ticker"] == "MSFT"
    assert report["dataset"]["frequency"] == "D"


def test_forecast_records_detected_regime_in_dataset_metadata(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    forecaster = _minimal_forecaster()
    forecaster._regime_result = {
        "regime": "LOW_VOL",
        "confidence": 0.91,
        "features": {"realized_volatility": 0.1},
        "recommendations": {"default_model": "SAMOSSA"},
    }

    result = forecaster.forecast(steps=3)

    assert result["detected_regime"] == "LOW_VOL"
    assert result["instrumentation_report"]["dataset"]["detected_regime"] == "LOW_VOL"
    assert result["instrumentation_report"]["dataset"]["forecast_horizon"] == 3


def test_save_audit_report_rewrites_manifest_with_valid_jsonl_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    forecaster = _minimal_forecaster()
    forecaster._instrumentation.set_dataset_metadata(ticker="NVDA")
    forecaster._instrumentation.record_artifact("example", {"value": 1})

    audit_path = tmp_path / "forecast_audit_20260101_000000.json"
    manifest_path = audit_path.parent / "forecast_audit_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"file": "older_audit.json", "sha256": "older"}),
                "{not-json",
                json.dumps(["invalid", "record"]),
                json.dumps({"file": audit_path.name, "sha256": "stale"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    forecaster.save_audit_report(audit_path)

    raw_lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in raw_lines]

    assert len(records) == 2
    assert {record["file"] for record in records} == {"older_audit.json", audit_path.name}

    latest = next(record for record in records if record["file"] == audit_path.name)
    expected_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert latest["sha256"] == expected_digest
    assert latest["source"] == "TimeSeriesForecaster.save_audit_report"
    assert latest["bytes"] == audit_path.stat().st_size
    assert not list(tmp_path.glob(".forecast_manifest_*.tmp"))


def test_forecast_records_exog_policy_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=True,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    price_series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index, name="Close")
    macro_context = pd.DataFrame(
        {"vix_level": [18.0, 19.0, 20.0, 21.0]},
        index=pd.date_range("2024-01-03", periods=4, freq="D"),
    )

    def _fake_fit(self, series, exogenous=None, **kwargs):  # noqa: ARG001
        self.best_order = (1, 0, 1)
        self.best_seasonal_order = (0, 0, 0, 0)
        self.fitted_model = type("Dummy", (), {"aic": 1.0, "bic": 2.0, "llf": -1.0, "nobs": len(series)})()
        return self

    def _fake_forecast(self, steps, exogenous=None, alpha=0.05):  # noqa: ARG001
        idx = pd.RangeIndex(start=1, stop=steps + 1)
        values = pd.Series([105.0] * steps, index=idx)
        return {
            "forecast": values,
            "lower_ci": values - 1.0,
            "upper_ci": values + 1.0,
            "alpha": alpha,
            "diagnostics": {},
            "residual_diagnostics": {},
            "convergence": {},
        }

    def _fake_summary(self):
        return {"order": (1, 0, 1), "seasonal_order": (0, 0, 0, 0)}

    monkeypatch.setattr("forcester_ts.sarimax.SARIMAXForecaster.fit", _fake_fit)
    monkeypatch.setattr("forcester_ts.sarimax.SARIMAXForecaster.forecast", _fake_forecast)
    monkeypatch.setattr("forcester_ts.sarimax.SARIMAXForecaster.get_model_summary", _fake_summary)

    forecaster.fit(price_series, macro_context=macro_context)
    result = forecaster.forecast(steps=2)

    artifacts = result["instrumentation_report"]["artifacts"]
    assert artifacts["exog_policy"]["fit_alignment"] == "ffill_only_zero_leading"
    assert artifacts["forecast_exog_policy"]["mode"] == "last_observation_hold"
    assert result["exog_policy"]["fit_alignment"] == "ffill_only_zero_leading"
    assert result["forecast_exog_policy"]["mode"] == "last_observation_hold"


# ---------------------------------------------------------------------------
# _load_trailing_oos_metrics — ticker/horizon scoping contract
# ---------------------------------------------------------------------------

def _write_audit(audit_dir: Path, name: str, ticker: str, horizon: int) -> None:
    """Write a minimal audit file with evaluation_metrics for the given ticker/horizon."""
    payload: Dict[str, Any] = {
        "dataset": {"ticker": ticker, "forecast_horizon": horizon, "length": 100},
        "artifacts": {
            "evaluation_metrics": {
                "samossa": {"rmse": 5.0 + ord(ticker[0]) / 10, "directional_accuracy": 0.55,
                            "n_observations": 30},
                "garch": {"rmse": 8.0, "directional_accuracy": 0.50, "n_observations": 30},
            }
        },
    }
    (audit_dir / name).write_text(json.dumps(payload))


def test_load_trailing_oos_metrics_scoped_to_current_ticker(monkeypatch, tmp_path: Path) -> None:
    """_load_trailing_oos_metrics must NOT return metrics from a different ticker.

    In multi-ticker runs (run_auto_trader, run_etl_pipeline) all tickers share
    the same audit directory.  If AAPL is processed after MSFT, AAPL must not
    use MSFT's last audit to drive its weight selection.
    """
    audit_dir = tmp_path / "forecast_audits"
    audit_dir.mkdir()

    # MSFT audit — written more recently (mtime trick: write MSFT second)
    _write_audit(audit_dir, "forecast_audit_aapl.json", "AAPL", 30)
    import time; time.sleep(0.02)
    _write_audit(audit_dir, "forecast_audit_msft.json", "MSFT", 30)
    # MSFT file is now the newest — without ticker filter it would be returned first

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(audit_dir))
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=False, mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(list(range(60)), dtype=float, index=index), ticker="AAPL")
    forecaster._instrumentation.set_dataset_metadata(forecast_horizon=30)

    result = forecaster._load_trailing_oos_metrics()

    # Must return AAPL-scoped metrics, not MSFT's
    assert result, "Expected non-empty metrics from AAPL audit"
    # AAPL samossa rmse = 5.0 + ord('A')/10 ≈ 11.5
    # MSFT samossa rmse = 5.0 + ord('M')/10 ≈ 12.7
    aapl_rmse = 5.0 + ord("A") / 10
    assert abs(result.get("samossa", {}).get("rmse", 0) - aapl_rmse) < 0.1, (
        f"Expected AAPL samossa rmse ~{aapl_rmse:.1f}, got {result.get('samossa', {}).get('rmse')}"
    )


def test_load_trailing_oos_metrics_no_match_returns_empty(monkeypatch, tmp_path: Path) -> None:
    """When no audit matches the current ticker, returns empty dict (safe no-op)."""
    audit_dir = tmp_path / "forecast_audits"
    audit_dir.mkdir()
    _write_audit(audit_dir, "forecast_audit_nvda.json", "NVDA", 30)

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(audit_dir))
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=False, mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(list(range(60)), dtype=float, index=index), ticker="AAPL")
    forecaster._instrumentation.set_dataset_metadata(forecast_horizon=30)

    result = forecaster._load_trailing_oos_metrics()
    assert result == {}, f"Expected empty dict when no ticker match, got {result}"


def test_load_trailing_oos_metrics_horizon_filter(monkeypatch, tmp_path: Path) -> None:
    """When audit horizon differs from current horizon, it is skipped."""
    audit_dir = tmp_path / "forecast_audits"
    audit_dir.mkdir()
    # Write an AAPL audit with horizon=60, but current forecaster uses horizon=30
    _write_audit(audit_dir, "forecast_audit_aapl_h60.json", "AAPL", 60)

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(audit_dir))
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=False, mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(list(range(60)), dtype=float, index=index), ticker="AAPL")
    forecaster._instrumentation.set_dataset_metadata(forecast_horizon=30)

    result = forecaster._load_trailing_oos_metrics()
    assert result == {}, f"Expected empty dict when horizon mismatches, got {result}"


# ---------------------------------------------------------------------------
# ensemble_metadata["confidence"] cap contract
# ---------------------------------------------------------------------------

def test_build_ensemble_confidence_cap_applied_to_metadata(monkeypatch, tmp_path: Path) -> None:
    """ensemble_metadata['confidence'] must be capped at ENSEMBLE_SCORE_CAP (0.65).

    The signal generator reads ensemble_metadata['confidence'] for weighted position
    sizing (time_series_signal_generator.py:1673-1694).  Without the cap those values
    can exceed the accuracy ceiling and inflate position sizes.
    """
    import numpy as np
    import pandas as pd
    import forcester_ts.forecaster as _fc_mod
    from forcester_ts import ensemble as _ens_mod

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))

    # Force derive_model_confidence to return scores above the cap
    monkeypatch.setattr(
        _ens_mod,
        "derive_model_confidence",
        lambda *a, **kw: {"samossa": 0.95, "garch": 0.80, "mssa_rl": 0.90},
    )

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=True,
        forecast_horizon=5,
    )
    forecaster = TimeSeriesForecaster(config=config)

    fake_blend = {
        "forecast": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        "lower_ci": None,
        "upper_ci": None,
    }
    monkeypatch.setattr(
        _ens_mod.EnsembleCoordinator,
        "blend_forecasts",
        lambda self, *a, **kw: fake_blend,
    )
    monkeypatch.setattr(
        _ens_mod.EnsembleCoordinator,
        "select_weights",
        lambda self, conf, model_directional_accuracy=None: ({"samossa": 1.0}, 0.9),
    )

    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(np.linspace(100, 160, 60), index=index), ticker="AAPL")
    result = forecaster._build_ensemble({"samossa_forecast": {"forecast": fake_blend["forecast"]}})

    assert result is not None, "_build_ensemble returned None unexpectedly"
    for conf in (
        result.get("forecast_bundle", {}).get("confidence", {}),
        result.get("metadata", {}).get("confidence", {}),
    ):
        assert conf, "Expected capped confidence payload in forecast bundle and metadata"
        for model, val in conf.items():
            assert val <= 0.65, (
                f"ensemble_metadata confidence for {model}={val:.3f} exceeds ENSEMBLE_SCORE_CAP=0.65"
            )
    assert result.get("forecast_bundle", {}).get("selection_score", 1.0) <= 0.65
    assert result.get("metadata", {}).get("selection_score", 1.0) <= 0.65


def test_build_ensemble_uses_trailing_oos_metrics_for_confidence_and_da(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Fresh forecasters must use disk-backed trailing OOS metrics at selection time."""
    import numpy as np
    import pandas as pd
    import forcester_ts.forecaster as _fc_mod
    from forcester_ts import ensemble as _ens_mod

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))

    captured: Dict[str, Any] = {}

    def _capture_confidence(summaries, oos_metrics=None):
        captured["oos_metrics"] = oos_metrics
        return {"samossa": 0.90, "garch": 0.70}

    def _capture_select(self, conf, model_directional_accuracy=None):
        captured["model_directional_accuracy"] = model_directional_accuracy
        return {"samossa": 1.0}, 0.90

    fake_blend = {
        "forecast": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        "lower_ci": None,
        "upper_ci": None,
    }
    monkeypatch.setattr(_fc_mod, "derive_model_confidence", _capture_confidence)
    monkeypatch.setattr(_ens_mod.EnsembleCoordinator, "select_weights", _capture_select)
    monkeypatch.setattr(
        _ens_mod.EnsembleCoordinator,
        "blend_forecasts",
        lambda self, *a, **kw: fake_blend,
    )

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=True,
        forecast_horizon=5,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(np.linspace(100, 160, 60), index=index), ticker="AAPL")

    trailing_metrics = {
        "samossa": {"rmse": 9.77, "directional_accuracy": 0.63, "n_observations": 50},
        "garch": {"rmse": 12.10, "directional_accuracy": 0.51, "n_observations": 50},
        "ensemble": {"rmse": 11.00, "directional_accuracy": 0.58, "n_observations": 50},
    }
    monkeypatch.setattr(forecaster, "_load_trailing_oos_metrics", lambda: trailing_metrics)
    forecaster._latest_metrics = {}

    result = forecaster._build_ensemble(
        {
            "samossa_forecast": {"forecast": fake_blend["forecast"]},
            "garch_forecast": {"forecast": fake_blend["forecast"]},
        }
    )

    assert result is not None, "_build_ensemble returned None unexpectedly"
    assert captured["oos_metrics"] == trailing_metrics
    assert captured["model_directional_accuracy"] == {
        "samossa": 0.63,
        "garch": 0.51,
    }


def test_build_ensemble_prefers_same_instance_latest_metrics_over_disk_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """When evaluate() already populated _latest_metrics, disk fallback must not override it."""
    import numpy as np
    import pandas as pd
    import forcester_ts.forecaster as _fc_mod
    from forcester_ts import ensemble as _ens_mod

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))

    captured: Dict[str, Any] = {}

    def _capture_confidence(summaries, oos_metrics=None):
        captured["oos_metrics"] = oos_metrics
        return {"samossa": 0.88, "garch": 0.72}

    def _capture_select(self, conf, model_directional_accuracy=None):
        captured["model_directional_accuracy"] = model_directional_accuracy
        return {"garch": 1.0}, 0.88

    fake_blend = {
        "forecast": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        "lower_ci": None,
        "upper_ci": None,
    }
    monkeypatch.setattr(_fc_mod, "derive_model_confidence", _capture_confidence)
    monkeypatch.setattr(_ens_mod.EnsembleCoordinator, "select_weights", _capture_select)
    monkeypatch.setattr(
        _ens_mod.EnsembleCoordinator,
        "blend_forecasts",
        lambda self, *a, **kw: fake_blend,
    )

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=True,
        forecast_horizon=5,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    forecaster.fit(pd.Series(np.linspace(100, 160, 60), index=index), ticker="AAPL")

    latest_metrics = {
        "samossa": {"rmse": 8.50, "directional_accuracy": 0.61, "n_observations": 50},
        "garch": {"rmse": 9.20, "directional_accuracy": 0.56, "n_observations": 50},
    }
    disk_metrics = {
        "samossa": {"rmse": 50.0, "directional_accuracy": 0.10, "n_observations": 50},
        "garch": {"rmse": 40.0, "directional_accuracy": 0.20, "n_observations": 50},
    }
    forecaster._latest_metrics = latest_metrics
    monkeypatch.setattr(forecaster, "_load_trailing_oos_metrics", lambda: disk_metrics)

    result = forecaster._build_ensemble(
        {
            "samossa_forecast": {"forecast": fake_blend["forecast"]},
            "garch_forecast": {"forecast": fake_blend["forecast"]},
        }
    )

    assert result is not None, "_build_ensemble returned None unexpectedly"
    assert captured["oos_metrics"] == latest_metrics
    assert captured["model_directional_accuracy"] == {
        "samossa": 0.61,
        "garch": 0.56,
    }
