import json
from pathlib import Path

from scripts.check_dashboard_health import (
    _load_json,
    _summarize_orchestration_health,
    _summarize_forecaster_health,
    _summarize_tickers,
)


def test_load_json_parses_valid_payload(tmp_path):
    path = tmp_path / "dashboard_data.json"
    payload = {"meta": {"run_id": "test_run"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_json(path)
    assert loaded["meta"]["run_id"] == "test_run"


def test_summarize_forecaster_health_handles_missing_block(capsys):
    payload = {}
    _summarize_forecaster_health(payload)
    out = capsys.readouterr().out
    assert "Forecaster Health" in out
    assert "no forecaster_health block" in out


def test_summarize_orchestration_health_handles_health_block(capsys):
    payload = {
        "orchestration_health": {
            "status": "WARN",
            "summary": {
                "snapshots": 2,
                "oos_source_counts": {"latest_metrics": 1, "heuristic_fallback": 1},
                "mssa_white_noise_failures": 1,
                "garch_unstable_runs": 1,
                "rmse_rank_active_runs": 1,
                "allow_as_default_runs": 1,
            },
            "latest": {
                "oos_source": "heuristic_fallback",
                "oos_quality": "heuristic_fallback",
                "mssa_eligible": False,
                "mssa_white_noise": False,
                "mssa_policy_status": "ready",
                "garch_fallback_mode": "exploding_variance_ratio",
                "garch_unstable": True,
            },
            "issues": ["mssa_rl_white_noise_failed"],
        }
    }
    _summarize_orchestration_health(payload)
    out = capsys.readouterr().out
    assert "Orchestration Health" in out
    assert "heuristic_fallback" in out
    assert "mssa_rl_white_noise_failed" in out


def test_summarize_preprocess_health_handles_health_block(capsys):
    from scripts.check_dashboard_health import _summarize_preprocess_health

    payload = {
        "preprocess_health": {
            "status": "WARN",
            "summary": {
                "snapshots": 2,
                "status_counts": {"PASS": 1, "WARN": 1},
                "quality_tag_counts": {"CLEAN": 1, "HIGH_IMPUTE": 1},
                "production_ok_runs": 1,
                "research_only_runs": 1,
                "sparse_runs": 1,
                "high_impute_runs": 1,
            },
            "latest": {
                "status": "WARN",
                "quality_tag": "HIGH_IMPUTE",
                "imputed_fraction": 0.35,
                "padding_fraction": 0.25,
            },
            "issues": ["preprocess_high_impute"],
        }
    }
    _summarize_preprocess_health(payload)
    out = capsys.readouterr().out
    assert "Preprocess Health" in out
    assert "HIGH_IMPUTE" in out
    assert "preprocess_high_impute" in out


def test_summarize_tickers_computes_basic_stats(capsys):
    payload = {
        "signals": [
            {
                "ticker": "AAPL",
                "status": "EXECUTED",
                "realized_pnl": 10.0,
            },
            {
                "ticker": "AAPL",
                "status": "EXECUTED",
                "realized_pnl": -5.0,
            },
            {
                "ticker": "MSFT",
                "status": "EXECUTED",
                "realized_pnl": 0.0,
            },
        ]
    }
    monitoring_cfg = {
        "quant_validation": {"min_profit_factor": 1.0, "min_win_rate": 0.0},
        "per_ticker": {},
    }

    _summarize_tickers(payload, monitoring_cfg)
    out = capsys.readouterr().out

    # AAPL should have 2 trades; MSFT 1 trade.
    assert "AAPL" in out
    assert "2" in out  # trades column for AAPL
    assert "MSFT" in out
    assert "1" in out  # trades column for MSFT
