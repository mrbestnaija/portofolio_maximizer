"""
Tests for scripts/build_training_dataset.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_db(tmp_path: Path, trades: list[dict]) -> Path:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT,
            action TEXT DEFAULT 'SELL',
            price REAL DEFAULT 100.0,
            exit_price REAL DEFAULT 110.0,
            realized_pnl REAL,
            holding_period_days REAL DEFAULT 1.0,
            exit_reason TEXT DEFAULT 'time_exit',
            is_close INTEGER DEFAULT 1,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            base_confidence REAL DEFAULT NULL,
            confidence_calibrated REAL DEFAULT NULL,
            effective_confidence REAL DEFAULT NULL,
            ts_signal_id TEXT
        )
        """
    )
    for trade in trades:
        conn.execute(
            "INSERT INTO trade_executions(ticker, trade_date, realized_pnl, is_close) VALUES(?,?,?,1)",
            (trade["ticker"], trade["date"], trade["pnl"]),
        )
    conn.execute(
        """
        CREATE VIEW production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_diagnostic, 0) = 0
          AND COALESCE(is_synthetic, 0) = 0
        """
    )
    conn.commit()
    conn.close()
    return db


def _make_eligibility(tmp_path: Path, healthy: list[str], extra_statuses: dict[str, str] | None = None) -> Path:
    payload = {"tickers": {}}
    for ticker in healthy:
        payload["tickers"][ticker] = {
            "status": "HEALTHY",
            "n_trades": 25,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "total_pnl": 500.0,
        }
    for ticker, status in (extra_statuses or {}).items():
        payload["tickers"][ticker] = {"status": status}
    path = tmp_path / "elig.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_audit_file(
    audit_dir: Path,
    name: str,
    window_end: str,
    *,
    ensemble_rmse: float | None = 1.5,
    best_single_rmse: float | None = 2.0,
) -> None:
    payload = {
        "window_id": name,
        "ticker": "NVDA",
        "window_end": window_end,
        "evaluation_metrics": {
            "ensemble": {"rmse": ensemble_rmse} if ensemble_rmse is not None else None,
            "best_single": {"rmse": best_single_rmse, "model": "samossa"},
        },
    }
    (audit_dir / f"forecast_audit_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


class TestBuildTrainingDataset:
    def test_missing_eligibility_uses_include_all_fallback(self, tmp_path):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        db = _make_db(
            tmp_path,
            [
                {"ticker": "NVDA", "date": "2025-08-01", "pnl": 100.0},
                {"ticker": "AAPL", "date": "2025-08-01", "pnl": -50.0},
            ],
        )
        result = build_training_datasets(
            db_path=db,
            audit_dir=tmp_path / "noaudits",
            eligibility_path=tmp_path / "missing.json",
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=True,
        )
        assert result["fail_closed"] is False
        assert result["eligibility_mode"] == "missing"
        assert result["trades"]["n_filtered"] == 2

    def test_unreadable_eligibility_uses_include_all_fallback(self, tmp_path):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        db = _make_db(tmp_path, [{"ticker": "NVDA", "date": "2025-08-01", "pnl": 100.0}])
        bad = tmp_path / "elig.json"
        bad.write_text("{not-json", encoding="utf-8")
        result = build_training_datasets(
            db_path=db,
            audit_dir=tmp_path / "noaudits",
            eligibility_path=bad,
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=True,
        )
        assert result["fail_closed"] is False
        assert result["eligibility_mode"] == "unreadable"
        assert result["trades"]["n_filtered"] == 1

    def test_zero_healthy_fails_closed(self, tmp_path):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        db = _make_db(tmp_path, [{"ticker": "NVDA", "date": "2025-08-01", "pnl": 100.0}])
        elig = _make_eligibility(tmp_path, healthy=[], extra_statuses={"NVDA": "WEAK"})
        result = build_training_datasets(
            db_path=db,
            audit_dir=tmp_path / "noaudits",
            eligibility_path=elig,
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=True,
        )
        assert result["fail_closed"] is True
        assert result["fail_closed_reason"] == "eligibility_exists_with_zero_healthy_tickers"
        assert result["trades"]["n_filtered"] == 0
        assert result["audits"]["n_filtered"] == 0

    def test_audit_rmse_ratio_safe_with_zero_denominator(self, tmp_path):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        audit_dir = tmp_path / "audits"
        audit_dir.mkdir()
        _make_audit_file(audit_dir, "zero", "2025-08-01", ensemble_rmse=1.5, best_single_rmse=0.0)
        result = build_training_datasets(
            db_path=tmp_path / "missing.db",
            audit_dir=audit_dir,
            eligibility_path=tmp_path / "missing.json",
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=True,
        )
        assert result["audits"]["n_filtered"] == 1

    def test_result_has_fail_closed_metadata(self, tmp_path):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        result = build_training_datasets(
            db_path=tmp_path / "missing.db",
            audit_dir=tmp_path / "missing_audits",
            eligibility_path=tmp_path / "missing.json",
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=True,
        )
        assert "fail_closed" in result
        assert "fail_closed_reason" in result
        assert "eligible_tickers_used" in result
        assert "thresholds" in result

    def test_write_failures_escalate_to_error(self, tmp_path, monkeypatch):
        pd = pytest.importorskip("pandas")
        from scripts.build_training_dataset import build_training_datasets

        db = _make_db(tmp_path, [{"ticker": "NVDA", "date": "2025-08-01", "pnl": 100.0}])
        elig = _make_eligibility(tmp_path, healthy=["NVDA"])

        def _boom(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
        result = build_training_datasets(
            db_path=db,
            audit_dir=tmp_path / "missing_audits",
            eligibility_path=elig,
            out_trades=tmp_path / "t.parquet",
            out_audits=tmp_path / "a.parquet",
            dry_run=False,
        )
        assert result["status"] == "ERROR"
        assert "trades_write_failed" in result["errors"]


class TestCLI:
    def test_fail_closed_cli_writes_summary_and_returns_1(self, tmp_path, capsys):
        pytest.importorskip("pandas")
        from scripts.build_training_dataset import main

        db = _make_db(tmp_path, [{"ticker": "NVDA", "date": "2025-08-01", "pnl": 100.0}])
        elig = _make_eligibility(tmp_path, healthy=[], extra_statuses={"NVDA": "WEAK"})
        summary = tmp_path / "summary.json"
        rc = main(
            [
                "--db",
                str(db),
                "--audit-dir",
                str(tmp_path / "noaudits"),
                "--eligibility",
                str(elig),
                "--summary-out",
                str(summary),
                "--json",
            ]
        )
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["fail_closed"] is True
        assert summary.exists()
        saved = json.loads(summary.read_text(encoding="utf-8"))
        assert saved["fail_closed"] is True
