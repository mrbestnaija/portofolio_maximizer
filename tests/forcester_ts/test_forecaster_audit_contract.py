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


def test_next_audit_path_is_unique_per_write(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", str(tmp_path))
    from forcester_ts import forecaster as forecaster_mod

    forecaster = _minimal_forecaster()
    uuids = iter(["aaaabbbbccccdddd", "1111222233334444"])
    monkeypatch.setattr(
        forecaster_mod.uuid,
        "uuid4",
        lambda: type("_Uuid", (), {"hex": next(uuids)})(),
    )

    path_one = forecaster._next_audit_path()
    path_two = forecaster._next_audit_path()

    assert path_one != path_two
    assert path_one.parent == tmp_path
    assert path_two.parent == tmp_path
    assert path_one.name.startswith("forecast_audit_")
    assert path_two.name.startswith("forecast_audit_")
