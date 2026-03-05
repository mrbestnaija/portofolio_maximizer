"""
Tests for scripts/data_sufficiency_monitor.py
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch


def _make_l1(
    n_used: int = 60,
    coverage_ratio: float = 0.25,
    lift_recent: float = 0.10,
    lift_global: float = 0.02,
    lift_ci_low: float | None = None,
) -> dict:
    return {
        "n_used_windows": n_used,
        "coverage_ratio": coverage_ratio,
        "lift_fraction_recent": lift_recent,
        "lift_fraction_global": lift_global,
        "lift_ci_low": lift_ci_low,
    }


def _make_l3(
    n_trades: int = 55,
    win_rate: float = 0.50,
    profit_factor: float = 1.50,
) -> dict:
    return {"n_trades": n_trades, "win_rate": win_rate, "profit_factor": profit_factor}


class TestRunDataSufficiency:
    def test_sufficient_when_all_targets_met(self):
        from scripts.data_sufficiency_monitor import run_data_sufficiency

        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3", return_value=_make_l3()
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            result = run_data_sufficiency(db_path=Path("x.db"), audit_dir=Path("a"))
        assert result["sufficient"] is True
        assert result["status"] == "SUFFICIENT"
        assert result["recommendations"] == []

    def test_insufficient_when_thresholds_missed(self):
        from scripts.data_sufficiency_monitor import run_data_sufficiency

        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1(n_used=10, coverage_ratio=0.05)), patch(
            "scripts.data_sufficiency_monitor._read_layer3", return_value=_make_l3(n_trades=10, win_rate=0.30, profit_factor=0.8)
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[{"ticker": "GS", "n_trades": 5, "win_rate": 0.0, "total_pnl": -91.0, "wins": 0}]):
            result = run_data_sufficiency(db_path=Path("x.db"), audit_dir=Path("a"))
        assert result["sufficient"] is False
        assert result["status"] == "INSUFFICIENT"
        assert any("TRADE_COUNT" in rec for rec in result["recommendations"])
        assert any("WEAK_TICKERS" in rec for rec in result["recommendations"])

    def test_layer_read_failures_escalate_to_data_error(self):
        from scripts.data_sufficiency_monitor import run_data_sufficiency

        with patch(
            "scripts.data_sufficiency_monitor._read_layer1",
            return_value={"metrics": {}, "error": "layer1_read_failed: boom"},
        ), patch(
            "scripts.data_sufficiency_monitor._read_layer3",
            return_value=_make_l3(),
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            result = run_data_sufficiency(db_path=Path("x.db"), audit_dir=Path("a"))
        assert result["status"] == "DATA_ERROR"
        assert result["sufficient"] is False
        assert any("layer1_read_failed" in rec for rec in result["recommendations"])

    def test_thresholds_are_sourced_from_shared_helper(self):
        from scripts.data_sufficiency_monitor import R3_MIN_TRADES, R3_MIN_WIN_RATE
        from scripts.robustness_thresholds import threshold_map

        thresholds = threshold_map()
        assert R3_MIN_TRADES == thresholds["r3_min_trades"]
        assert R3_MIN_WIN_RATE == thresholds["r3_min_win_rate"]

    def test_profit_factor_zero_is_insufficient(self):
        from scripts.data_sufficiency_monitor import run_data_sufficiency

        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3",
            return_value=_make_l3(n_trades=55, win_rate=0.50, profit_factor=0.0),
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            result = run_data_sufficiency(db_path=Path("x.db"), audit_dir=Path("a"))

        assert result["status"] == "INSUFFICIENT"
        assert result["sufficient"] is False
        assert any("PROFIT_FACTOR" in rec for rec in result["recommendations"])

    def test_non_finite_metrics_produce_data_error(self):
        from scripts.data_sufficiency_monitor import run_data_sufficiency

        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3",
            return_value=_make_l3(n_trades=55, win_rate=float("nan"), profit_factor=float("inf")),
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            result = run_data_sufficiency(db_path=Path("x.db"), audit_dir=Path("a"))

        assert result["status"] == "DATA_ERROR"
        assert result["sufficient"] is False
        assert "non_finite_win_rate" in result["recommendations"]
        assert "non_finite_profit_factor" in result["recommendations"]


class TestCLI:
    def test_exit_code_2_when_db_missing(self, tmp_path):
        from scripts.data_sufficiency_monitor import main

        assert main(["--db", str(tmp_path / "missing.db"), "--audit-dir", str(tmp_path / "audits")]) == 2

    def test_json_output_mode(self, tmp_path, capsys):
        from scripts.data_sufficiency_monitor import main

        db = tmp_path / "test.db"
        db.write_bytes(b"")
        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3", return_value=_make_l3()
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            rc = main(["--db", str(db), "--audit-dir", str(tmp_path / "audits"), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "SUFFICIENT"
        assert "thresholds" in payload["metrics"]

    def test_exit_code_2_when_data_error_json_mode(self, tmp_path):
        from scripts.data_sufficiency_monitor import main

        db = tmp_path / "test.db"
        db.write_bytes(b"")
        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3",
            return_value=_make_l3(n_trades=55, win_rate=float("nan"), profit_factor=1.5),
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            rc = main(["--db", str(db), "--audit-dir", str(tmp_path / "audits"), "--json"])
        assert rc == 2

    def test_exit_code_2_when_data_error_text_mode(self, tmp_path):
        from scripts.data_sufficiency_monitor import main

        db = tmp_path / "test.db"
        db.write_bytes(b"")
        with patch("scripts.data_sufficiency_monitor._read_layer1", return_value=_make_l1()), patch(
            "scripts.data_sufficiency_monitor._read_layer3",
            return_value=_make_l3(n_trades=55, win_rate=0.5, profit_factor=float("inf")),
        ), patch("scripts.data_sufficiency_monitor._read_per_ticker", return_value=[]):
            rc = main(["--db", str(db), "--audit-dir", str(tmp_path / "audits")])
        assert rc == 2


class TestHardConstraints:
    def test_script_does_not_write_gate_files(self):
        from scripts import data_sufficiency_monitor

        source = inspect.getsource(data_sufficiency_monitor)
        assert "gate_status_latest" not in source
