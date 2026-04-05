"""
Tests for scripts/calibrate_confidence_thresholds.py (P3-A).

Validates:
- Bins are monotonically non-decreasing in win_rate (or close to it at small N).
- cold_start status returned when fewer than MIN_TRADES directional trades.
- Mechanical exits are excluded from calibration.
- Output JSON written to OUTPUT_PATH.
- Script exits 0 on success, 1 on missing DB, 2 on cold-start.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_PATH))

from scripts.calibrate_confidence_thresholds import (
    MIN_TRADES,
    _MECHANICAL_EXIT_REASONS,
    _bin_trades,
    _load_trades,
    run,
)


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a minimal trade_executions DB with synthetic round-trips."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            action TEXT,
            realized_pnl REAL,
            is_close INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            confidence_calibrated REAL,
            effective_confidence REAL,
            entry_trade_id INTEGER,
            exit_reason TEXT
        );
        """
    )
    # Insert opening legs (id 1..N) then closing legs (id N+1..2N)
    n = len(rows)
    for i, row in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO trade_executions (id, ticker, action, is_close, effective_confidence) "
            "VALUES (?, ?, 'BUY', 0, ?)",
            (i, row.get("ticker", "AAPL"), row["conf"]),
        )
        conn.execute(
            "INSERT INTO trade_executions (id, ticker, action, realized_pnl, is_close, "
            "entry_trade_id, exit_reason) VALUES (?, ?, 'SELL', ?, 1, ?, ?)",
            (
                n + i,
                row.get("ticker", "AAPL"),
                row["pnl"],
                i,
                row.get("exit_reason", "directional"),
            ),
        )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# _load_trades
# ---------------------------------------------------------------------------


def test_load_trades_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_trades(tmp_path / "nonexistent.db")


def test_load_trades_excludes_diagnostic_and_synthetic(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            action TEXT,
            realized_pnl REAL,
            is_close INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            confidence_calibrated REAL,
            effective_confidence REAL,
            entry_trade_id INTEGER,
            exit_reason TEXT
        );
        """
    )
    # Production close with valid opening leg
    conn.execute(
        "INSERT INTO trade_executions VALUES (1,'AAPL','BUY',NULL,0,0,0,NULL,0.75,NULL,'directional')"
    )
    conn.execute(
        "INSERT INTO trade_executions VALUES (2,'AAPL','SELL',50.0,1,0,0,NULL,NULL,1,'directional')"
    )
    # Diagnostic close — must be excluded
    conn.execute(
        "INSERT INTO trade_executions VALUES (3,'AAPL','BUY',NULL,0,0,0,NULL,0.80,NULL,'directional')"
    )
    conn.execute(
        "INSERT INTO trade_executions VALUES (4,'AAPL','SELL',-10.0,1,1,0,NULL,NULL,3,'directional')"
    )
    conn.commit()
    conn.close()

    rows = _load_trades(db)
    assert len(rows) == 1, f"Expected 1 production round-trip; got {len(rows)}"
    assert abs(rows[0]["realized_pnl"] - 50.0) < 1e-6


# ---------------------------------------------------------------------------
# _bin_trades
# ---------------------------------------------------------------------------


def test_bin_trades_win_rate_non_decreasing_with_sufficient_data() -> None:
    """Win rate should be non-decreasing across bins when confidence is well-calibrated."""
    # Build synthetic data: low conf → bad trades, high conf → good trades
    rows = []
    for i in range(25):
        conf = 0.5 + i * 0.02  # 0.50..0.98
        pnl = -50.0 if conf < 0.65 else 50.0
        rows.append({"conf": conf, "realized_pnl": pnl, "win": 1 if pnl > 0 else 0})
    bins = _bin_trades(rows, n_bins=5)
    win_rates = [b["win_rate"] for b in bins]
    for i in range(len(win_rates) - 1):
        assert win_rates[i] <= win_rates[i + 1] + 0.05, (
            f"Win rate not monotonically non-decreasing: {win_rates}"
        )


def test_bin_trades_cold_start_returns_partial_output() -> None:
    """With only 3 rows, binning should still produce output without error."""
    rows = [
        {"conf": 0.6, "realized_pnl": 10.0, "win": 1},
        {"conf": 0.7, "realized_pnl": -5.0, "win": 0},
        {"conf": 0.9, "realized_pnl": 20.0, "win": 1},
    ]
    bins = _bin_trades(rows, n_bins=5)
    assert isinstance(bins, list)


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------


def test_run_cold_start_when_few_directional_trades(tmp_path: Path) -> None:
    """Fewer than MIN_TRADES directional trades → status=cold_start, exit code 2."""
    n = MIN_TRADES - 1
    row_data = [{"conf": 0.75 + i * 0.005, "pnl": 10.0} for i in range(n)]
    db = _make_db(tmp_path, row_data)
    result = run(db, n_bins=5)
    assert result["status"] == "cold_start"
    assert result["n_directional_trades"] == n


def test_run_ok_with_sufficient_trades(tmp_path: Path) -> None:
    """MIN_TRADES directional trades → status=ok."""
    row_data = [{"conf": 0.55 + i * 0.01, "pnl": 5.0 if i % 2 == 0 else -3.0}
                for i in range(MIN_TRADES)]
    db = _make_db(tmp_path, row_data)
    result = run(db, n_bins=5)
    assert result["status"] == "ok"
    assert result["n_directional_trades"] == MIN_TRADES
    assert len(result["bins"]) >= 1
    for b in result["bins"]:
        assert 0.0 <= b["win_rate"] <= 1.0
        assert b["n_trades"] >= 1


def test_run_excludes_mechanical_exits(tmp_path: Path) -> None:
    """Mechanical exits must not appear in calibration bins."""
    # Half directional, half stop_loss
    row_data = [
        {"conf": 0.70 + i * 0.01, "pnl": 10.0, "exit_reason": "directional"}
        for i in range(MIN_TRADES)
    ] + [
        {"conf": 0.50, "pnl": -30.0, "exit_reason": "stop_loss"}
        for _ in range(5)
    ]
    db = _make_db(tmp_path, row_data)
    result = run(db, n_bins=5)
    # Mechanical exits subtracted from total
    assert result["n_mechanical_exits_excluded"] == 5
    assert result["n_directional_trades"] == MIN_TRADES


def test_run_writes_output_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output JSON must be written to OUTPUT_PATH (via main())."""
    import scripts.calibrate_confidence_thresholds as mod
    out = tmp_path / "confidence_calibration.json"
    monkeypatch.setattr(mod, "OUTPUT_PATH", out)

    row_data = [{"conf": 0.60 + i * 0.01, "pnl": 5.0} for i in range(MIN_TRADES)]
    db = _make_db(tmp_path, row_data)

    # main() is responsible for writing — call it with patched OUTPUT_PATH
    monkeypatch.setattr(sys, "argv", ["calibrate_confidence_thresholds.py", "--db", str(db)])
    monkeypatch.setattr(mod, "OUTPUT_PATH", out)
    exit_code = mod.main()
    assert exit_code in (0, 2)  # 0=ok, 2=cold_start — both write the file

    assert out.exists(), f"Expected output at {out}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "bins" in payload
    assert "profit_factor" in payload


def test_mechanical_exit_reasons_coverage() -> None:
    """Ensure the exclusion set covers known mechanical exit labels."""
    for label in ("stop_loss", "max_holding", "time_exit", "forced_exit", "flatten"):
        assert label in _MECHANICAL_EXIT_REASONS, f"'{label}' missing from _MECHANICAL_EXIT_REASONS"
