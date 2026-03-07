import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_auto_trader
from scripts.evaluate_sleeve_promotions import PromotionRules, evaluate_promotions


def test_compute_mid_price_prefers_bid_ask():
    frame = pd.DataFrame(
        [{"Bid": 10.0, "Ask": 12.0, "High": 15.0, "Low": 9.0, "Close": 11.0}],
        index=[pd.Timestamp("2024-01-01")],
    )
    assert run_auto_trader._compute_mid_price(frame) == pytest.approx(11.0)


def test_log_execution_event_jsonl(tmp_path: Path):
    log_path = tmp_path / "exec.jsonl"
    run_auto_trader.EXECUTION_LOG_PATH = log_path
    run_auto_trader._log_execution_event(
        "run1",
        2,
        {"ticker": "AAPL", "status": "EXECUTED", "entry_price": 10.0, "mid_price": 9.5},
    )
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["run_id"] == "run1"
    assert payload["cycle"] == 2
    assert payload["mid_price"] == 9.5


def test_evaluate_promotions_promotes_and_demotes():
    summary = [
        {"ticker": "MTN", "bucket": "speculative", "total_trades": 12, "win_rate": 0.6, "profit_factor": 1.3},
        {"ticker": "CL=F", "bucket": "core", "total_trades": 15, "win_rate": 0.4, "profit_factor": 0.8},
    ]
    rules = PromotionRules(min_trades=10, promote_win_rate=0.55, promote_profit_factor=1.2)
    plan = evaluate_promotions(summary, bucket_map={}, rules=rules)
    promoted = [p["ticker"] for p in plan["promotions"]]
    demoted = [d["ticker"] for d in plan["demotions"]]
    assert "MTN" in promoted
    assert "CL=F" in demoted



def test_save_audit_report_defaults_to_forecast_only_signal_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

    forecaster = TimeSeriesForecaster(
        config=TimeSeriesForecasterConfig(
            sarimax_enabled=False,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
            ensemble_enabled=False,
        )
    )

    def _fake_dump_json(output_path: Path) -> None:
        output_path.write_text(
            json.dumps({"dataset": {"ticker": "AAPL", "forecast_horizon": 30}}, indent=2),
            encoding="utf-8",
        )

    monkeypatch.setattr(forecaster._instrumentation, "dump_json", _fake_dump_json)
    monkeypatch.setattr(forecaster, "_append_audit_manifest_entry", lambda _path: None)

    audit_path = tmp_path / "forecast_audit_test.json"
    forecaster.save_audit_report(audit_path)

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    signal_context = payload["signal_context"]
    assert signal_context["context_type"] == "FORECAST_ONLY"
    assert signal_context["ticker"] == "AAPL"
    assert signal_context["forecast_horizon"] == 30
    assert signal_context["signal_context_missing"] is True
    assert signal_context["ts_signal_id"] is None



def test_attach_signal_context_updates_only_authoritative_audit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audit_dir = tmp_path / "logs" / "forecast_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    authoritative = audit_dir / "forecast_audit_authoritative.json"
    authoritative.write_text(
        json.dumps({"dataset": {"ticker": "AAPL", "forecast_horizon": 30}}, indent=2),
        encoding="utf-8",
    )
    unrelated = audit_dir / "forecast_audit_unrelated.json"
    unrelated.write_text(
        json.dumps(
            {
                "dataset": {"ticker": "AAPL", "forecast_horizon": 6},
                "signal_context": {
                    "context_type": "FORECAST_ONLY",
                    "ts_signal_id": None,
                    "ticker": "AAPL",
                    "run_id": "run_wrong",
                    "entry_ts": None,
                    "forecast_horizon": 2,
                    "signal_context_missing": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_auto_trader, "ROOT_PATH", tmp_path)
    run_auto_trader._attach_signal_context_to_forecast_audit(
        forecast_bundle={"horizon": 2, "forecast_audit_path": str(authoritative)},
        execution_report={
            "ts_signal_id": "ts_AAPL_1",
            "executed": True,
            "context_type": "TRADE",
            "run_id": "run_exec",
            "signal_timestamp": "2026-03-04T00:00:00+00:00",
            "forecast_horizon": 30,
        },
        ticker="AAPL",
        run_id="run_outer",
    )

    patched = json.loads(authoritative.read_text(encoding="utf-8"))
    patched_context = patched["signal_context"]
    assert patched_context["context_type"] == "TRADE"
    assert patched_context["ts_signal_id"] == "ts_AAPL_1"
    assert patched_context["run_id"] == "run_exec"
    assert patched_context["entry_ts"] == "2026-03-04T00:00:00+00:00"
    assert patched_context["forecast_horizon"] == 30
    assert patched_context["signal_context_missing"] is False

    untouched = json.loads(unrelated.read_text(encoding="utf-8"))
    untouched_context = untouched["signal_context"]
    assert untouched_context["run_id"] == "run_wrong"
    assert untouched_context["forecast_horizon"] == 2
    assert untouched_context["ts_signal_id"] is None


def test_attach_signal_context_backfills_dataset_ticker_for_forecast_only_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audit_dir = tmp_path / "logs" / "forecast_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    authoritative = audit_dir / "forecast_audit_forecast_only.json"
    authoritative.write_text(
        json.dumps({"dataset": {"forecast_horizon": 30}}, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_auto_trader, "ROOT_PATH", tmp_path)
    run_auto_trader._attach_signal_context_to_forecast_audit(
        forecast_bundle={"horizon": 30, "forecast_audit_path": str(authoritative)},
        execution_report={
            "ts_signal_id": "ts_GOOG_1",
            "executed": False,
            "run_id": "run_exec",
            "signal_timestamp": "2026-03-05T00:00:00+00:00",
            "forecast_horizon": 30,
        },
        ticker="GOOG",
        run_id="run_outer",
    )

    patched = json.loads(authoritative.read_text(encoding="utf-8"))
    assert patched["dataset"]["ticker"] == "GOOG"
    patched_context = patched["signal_context"]
    assert patched_context["context_type"] == "FORECAST_ONLY"
    assert patched_context["run_id"] == "run_exec"
    assert patched_context["forecast_horizon"] == 30
    assert patched_context["ts_signal_id"] == "ts_GOOG_1"
