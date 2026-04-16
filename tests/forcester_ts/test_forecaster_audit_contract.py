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


def test_forecast_propagates_garch_fallback_mode_into_volatility_metadata(monkeypatch) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    forecaster = _minimal_forecaster()
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    price_series = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        index=index,
        name="Close",
    )
    forecaster.fit(price_series, ticker="AAPL")

    class _DummyGarch:
        p = 1
        q = 1

        def forecast(self, steps: int) -> Dict[str, Any]:
            idx = pd.Index(range(1, steps + 1), name="horizon")
            variance = pd.Series([0.0004] * steps, index=idx)
            mean = pd.Series([0.001] * steps, index=idx)
            return {
                "variance_forecast": variance,
                "mean_forecast": mean,
                "volatility": variance.pow(0.5),
                "steps": steps,
                "aic": None,
                "bic": None,
                "convergence_ok": True,
                "residual_diagnostics": {},
                "residual_diagnostics_status": "unavailable",
                "residual_diagnostics_reason": "ewma_fallback",
                "ewma_lambda": 0.94,
                "fallback_mode": "explicit_ewma_backend",
                "persistence": None,
                "volatility_ratio_to_realized": 1.8,
            }

        def get_model_summary(self) -> Dict[str, Any]:
            return {
                "backend": "ewma",
                "ewma_lambda": 0.94,
                "residual_diagnostics_status": "unavailable",
                "residual_diagnostics_reason": "ewma_fallback",
                "fallback_mode": "explicit_ewma_backend",
                "aic": None,
                "bic": None,
            }

    forecaster._garch = _DummyGarch()

    result = forecaster.forecast(steps=3)

    assert result["volatility_forecast"]["fallback_mode"] == "explicit_ewma_backend"
    assert result["volatility_forecast"]["ewma_lambda"] == 0.94
    assert result["volatility_forecast"]["residual_diagnostics_status"] == "unavailable"
    assert result["volatility_forecast"]["residual_diagnostics_reason"] == "ewma_fallback"
    garch_runs = [
        run
        for run in result["instrumentation_report"]["runs"]
        if run.get("model") == "garch" and run.get("phase") == "forecast"
    ]
    assert garch_runs, "expected at least one tracked garch forecast run"
    assert garch_runs[-1]["metadata"]["fallback_mode"] == "explicit_ewma_backend"
    assert garch_runs[-1]["metadata"]["ewma_lambda"] == 0.94


def test_forecaster_rmse_monitor_config_falls_closed_when_config_missing(monkeypatch, tmp_path: Path) -> None:
    missing_cfg = tmp_path / "missing.yml"
    monkeypatch.setenv("TS_FORECAST_MONITOR_CONFIG", str(missing_cfg))

    forecaster = _minimal_forecaster()

    assert forecaster._rmse_monitor_cfg["min_lift_rmse_ratio"] == 0.02
    assert forecaster._rmse_monitor_cfg["strict_preselection_max_rmse_ratio"] == 1.1
    assert forecaster._rmse_monitor_cfg["baseline_model"] == "EFFECTIVE_DEFAULT"


def test_forecaster_rmse_monitor_config_falls_closed_when_yaml_invalid(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "forecaster_monitoring.yml"
    cfg_path.write_text("forecaster_monitoring: [\n", encoding="utf-8")
    monkeypatch.setenv("TS_FORECAST_MONITOR_CONFIG", str(cfg_path))

    forecaster = _minimal_forecaster()

    assert forecaster._rmse_monitor_cfg["min_lift_rmse_ratio"] == 0.02
    assert forecaster._rmse_monitor_cfg["strict_preselection_max_rmse_ratio"] == 1.1
    assert forecaster._rmse_monitor_cfg["baseline_model"] == "EFFECTIVE_DEFAULT"


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


def test_forecast_returns_written_audit_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))
    forecaster = _minimal_forecaster()

    result = forecaster.forecast(steps=2)

    audit_path = Path(result["forecast_audit_path"])
    assert audit_path.exists()
    assert audit_path.parent == tmp_path
    assert audit_path.name.startswith("forecast_audit_")


def test_forecast_uses_unique_audit_paths_across_consecutive_writes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))
    forecaster = _minimal_forecaster()

    first = forecaster.forecast(steps=2)
    second = forecaster.forecast(steps=2)

    assert first["forecast_audit_path"] != second["forecast_audit_path"]
    assert Path(first["forecast_audit_path"]).exists()
    assert Path(second["forecast_audit_path"]).exists()


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


def test_save_audit_report_includes_configured_event_type_and_evidence_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        ensemble_kwargs={
            "audit_event_type": "FORECAST_AUDIT",
            "audit_evidence_context": "RMSE_ONLY",
        },
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._instrumentation.set_dataset_metadata(ticker="AAPL")

    audit_path = tmp_path / "forecast_audit_20260101_000001.json"
    forecaster.save_audit_report(audit_path)

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["event_type"] == "FORECAST_AUDIT"
    assert payload["evidence_context"] == "RMSE_ONLY"


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


def test_load_trailing_oos_metrics_fails_closed_when_ticker_unknown(monkeypatch, tmp_path: Path) -> None:
    """Regression: when current ticker is unknown (empty series name), must return {}
    rather than loading the newest audit file regardless of ticker.

    _resolve_ticker() returns '' when price_series.name is absent or a generic column
    name; the dataset metadata stores that as None. Without this guard, a tickerless
    forecaster (e.g. run from a bare pd.Series) would consume OOS metrics from an
    unrelated asset's audit file, silently poisoning weight selection.
    """
    audit_dir = tmp_path / "forecast_audits"
    audit_dir.mkdir()
    _write_audit(audit_dir, "forecast_audit_msft.json", "MSFT", 30)

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(audit_dir))
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=False, mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    # Fit with a series whose name is absent → ticker resolves to ""
    series = pd.Series(list(range(60)), dtype=float, index=index)
    series.name = None
    forecaster.fit(series, ticker="")  # explicitly tickerless
    forecaster._instrumentation.set_dataset_metadata(forecast_horizon=30)

    # current_ticker is None in metadata — must fail closed, not load MSFT metrics
    result = forecaster._load_trailing_oos_metrics()
    assert result == {}, (
        f"Tickerless forecaster must not load unrelated audit metrics; got {result}"
    )


# ---------------------------------------------------------------------------
# P2-B: CV OOS proxy from prior fold metrics
# ---------------------------------------------------------------------------

def test_cv_fold_metrics_used_when_no_disk_audit(monkeypatch, tmp_path: Path) -> None:
    """When no disk audit exists, _load_trailing_oos_metrics() must return
    _cv_fold_metrics (injected by RollingWindowValidator) so RMSE-rank is not
    disabled in CV context. (P2-B fix)"""
    audit_dir = tmp_path / "empty_audits"
    audit_dir.mkdir()

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(audit_dir))
    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=False, mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    series = pd.Series(list(range(60)), dtype=float, index=index, name="AAPL")
    forecaster.fit(series, ticker="AAPL")
    forecaster._instrumentation.set_dataset_metadata(ticker="AAPL", forecast_horizon=30)

    # Inject prior fold metrics (simulating what RollingWindowValidator does)
    prior_metrics = {"samossa": {"rmse": 1.2, "directional_accuracy": 0.55}}
    forecaster._cv_fold_metrics = prior_metrics

    result = forecaster._load_trailing_oos_metrics()
    assert result == prior_metrics, (
        f"_load_trailing_oos_metrics must return _cv_fold_metrics when no disk audit exists; "
        f"got {result}"
    )


def test_rolling_window_validator_injects_prior_fold_metrics() -> None:
    """RollingWindowValidator must inject prior fold metrics into each new forecaster
    so that from fold 2 onwards, _cv_fold_metrics is non-empty at forecast() time.
    (P2-B fix — RollingWindowValidator.run() prior_fold_oos_metrics injection)"""
    import numpy as np
    from forcester_ts.cross_validation import RollingWindowValidator, RollingWindowCVConfig

    np.random.seed(42)
    n = 200
    prices = pd.Series(
        100 + np.cumsum(np.random.normal(0, 1, n)),
        index=pd.date_range("2022-01-01", periods=n, freq="D"),
        name="TEST",
    )

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False, garch_enabled=False,
        samossa_enabled=True, mssa_rl_enabled=False,
        ensemble_enabled=False,  # keep simple so evaluate() always has samossa metrics
    )
    captured_cv_fold_metrics = []

    # Patch TimeSeriesForecaster to capture _cv_fold_metrics at fit time
    import forcester_ts.cross_validation as cv_mod
    _original_tsf = cv_mod.TimeSeriesForecaster

    class _CapturingForecaster(_original_tsf):
        def fit(self, *args, **kwargs):
            captured_cv_fold_metrics.append(dict(self._cv_fold_metrics))
            return super().fit(*args, **kwargs)

    cv_mod.TimeSeriesForecaster = _CapturingForecaster
    try:
        validator = RollingWindowValidator(
            forecaster_config=config,
            cv_config=RollingWindowCVConfig(min_train_size=120, horizon=10, step_size=30, max_folds=3),
        )
        validator.run(price_series=prices, ticker="TEST")
    finally:
        cv_mod.TimeSeriesForecaster = _original_tsf

    assert len(captured_cv_fold_metrics) >= 2, "At least 2 folds needed for this test"
    # Fold 1 must have empty prior metrics (no prior fold)
    assert captured_cv_fold_metrics[0] == {}, (
        f"Fold 1 must have empty _cv_fold_metrics; got {captured_cv_fold_metrics[0]}"
    )
    # Fold 2+ must have prior fold metrics injected
    assert captured_cv_fold_metrics[1] != {}, (
        "Fold 2 must have non-empty _cv_fold_metrics (injected from fold 1 evaluate)"
    )


def test_load_trailing_oos_metrics_finds_match_beyond_position_20(
    monkeypatch, tmp_path: Path
) -> None:
    """C5 fix: removing [:20] cap allows matches at position 21+ in the file list.

    Scenario: 25 non-AAPL audits are written first (newest mtime),
    then 1 AAPL audit is written last (oldest mtime, position 26 when sorted desc).
    Before the fix the scan stopped at 20 and returned {}. After the fix it must
    find and return the AAPL audit.
    """
    import time

    audit_dir = tmp_path / "many_audits"
    audit_dir.mkdir()

    # Write 25 non-AAPL audits (they will have newer mtime → sort first)
    for i in range(25):
        ticker = f"T{i:02d}"
        _write_audit(audit_dir, f"forecast_audit_{ticker.lower()}.json", ticker, 30)
        time.sleep(0.005)  # ensure distinct mtimes

    # Write AAPL audit last → it will have the OLDEST mtime → sort last (position 26)
    time.sleep(0.01)
    _write_audit(audit_dir, "forecast_audit_aapl.json", "AAPL", 30)

    # Re-touch the non-AAPL files to make them newer than AAPL
    for i in range(25):
        ticker = f"T{i:02d}"
        p = audit_dir / f"forecast_audit_{ticker.lower()}.json"
        p.touch()
        time.sleep(0.001)

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
    assert result, (
        "C5 fix: _load_trailing_oos_metrics must find AAPL audit at position 26 "
        "(beyond old [:20] cap); got empty result"
    )
    assert "samossa" in result, f"Expected samossa metrics in AAPL audit; got keys: {list(result)}"


def test_load_trailing_oos_metrics_warns_when_no_match(
    monkeypatch, tmp_path: Path, caplog: "pytest.LogCaptureFixture"
) -> None:
    """C5 fix: when no file matches the current ticker, a WARNING must be emitted.
    This makes RMSE-rank fallback visible in logs without reading source code.
    """
    import logging

    audit_dir = tmp_path / "no_match_audits"
    audit_dir.mkdir()
    for ticker in ("MSFT", "NVDA", "TSLA", "GOOGL", "META"):
        _write_audit(audit_dir, f"forecast_audit_{ticker.lower()}.json", ticker, 30)

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

    with caplog.at_level(logging.WARNING, logger="forcester_ts.forecaster"):
        result = forecaster._load_trailing_oos_metrics()

    assert result == {}, f"Expected empty result when no AAPL audit exists; got {result}"
    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no matching file with evaluation_metrics" in t for t in warning_texts), (
        f"Expected WARNING about no matching audit file; got: {warning_texts}"
    )


def test_load_trailing_oos_metrics_falls_back_to_production_eval_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Live auto_trader runs write to production/ but never write evaluation_metrics.
    OOS metrics from prior ETL/CV runs live in production_eval/.  The function must
    scan production_eval/ as a secondary source when production/ has no match.

    This is the fix for the OOS dead-code bug: RMSE-rank was permanently disabled
    in live mode because _load_trailing_oos_metrics() only scanned self._audit_dir
    (production/), which never contains evaluation_metrics during live cycles.
    """
    # Simulate production/ (primary, written by auto_trader — no evaluation_metrics)
    production_dir = tmp_path / "production"
    production_dir.mkdir()
    trade_audit = {
        "dataset": {"ticker": "AAPL", "forecast_horizon": 30, "length": 100},
        "artifacts": {},  # no evaluation_metrics — live run
        "signal_context": {"ts_signal_id": "ts_AAPL_live_0001"},
    }
    (production_dir / "forecast_audit_live.json").write_text(json.dumps(trade_audit))

    # Simulate production_eval/ (secondary, written by ETL/CV — has evaluation_metrics)
    eval_dir = tmp_path / "production_eval"
    eval_dir.mkdir()
    _write_audit(eval_dir, "forecast_audit_cv.json", "AAPL", 30)

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(production_dir))
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

    assert result, (
        "Expected non-empty OOS metrics from production_eval/ fallback; "
        "live auto_trader RMSE-rank is permanently dead without this"
    )
    assert "samossa" in result, f"Expected samossa metrics in result; got keys: {list(result)}"


def test_load_trailing_oos_metrics_eval_dir_respects_ticker_scope(
    monkeypatch, tmp_path: Path
) -> None:
    """production_eval/ fallback must apply the same ticker filter as the primary scan.
    An MSFT eval audit must NOT be returned when current ticker is AAPL.
    """
    production_dir = tmp_path / "production"
    production_dir.mkdir()

    eval_dir = tmp_path / "production_eval"
    eval_dir.mkdir()
    _write_audit(eval_dir, "forecast_audit_msft.json", "MSFT", 30)  # wrong ticker

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(production_dir))
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

    assert result == {}, (
        "production_eval/ fallback must NOT return MSFT metrics when current ticker is AAPL"
    )


def test_load_trailing_oos_metrics_falls_back_to_research_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Tertiary scan: when production/ and production_eval/ have no evaluation_metrics,
    research/ (ETL/CV audit files) must be scanned.

    Root cause of live RMSE-rank being dead: production/ and production_eval/ only
    receive auto_trader audit files which never call evaluate() → no evaluation_metrics.
    Only ETL/CV runs write evaluation_metrics, and those go to research/.
    Without the research/ scan, RMSE-rank is permanently disabled in live mode and
    confidence stays at heuristic baseline (~0.23-0.38, well below 0.55 gate threshold).
    """
    production_dir = tmp_path / "production"
    production_dir.mkdir()
    # production/ file: no evaluation_metrics (typical live auto_trader file)
    no_metrics_audit = {
        "dataset": {"ticker": "AAPL", "forecast_horizon": 30, "length": 100},
        "artifacts": {},
    }
    (production_dir / "forecast_audit_live.json").write_text(json.dumps(no_metrics_audit))

    # production_eval/ file: also no evaluation_metrics (RMSE_ONLY auto_trader sprint)
    eval_dir = tmp_path / "production_eval"
    eval_dir.mkdir()
    (eval_dir / "forecast_audit_sprint.json").write_text(json.dumps(no_metrics_audit))

    # research/ file: HAS evaluation_metrics (ETL CV run with --use-cv)
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    _write_audit(research_dir, "forecast_audit_etl_cv.json", "AAPL", 30)

    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(production_dir))
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

    assert result, (
        "Tertiary research/ scan must find ETL/CV evaluation_metrics; "
        "live RMSE-rank is permanently dead without this scan"
    )
    assert "samossa" in result, f"Expected samossa key from research/ metrics; got {list(result)}"
