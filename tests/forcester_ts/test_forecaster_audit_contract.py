from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
