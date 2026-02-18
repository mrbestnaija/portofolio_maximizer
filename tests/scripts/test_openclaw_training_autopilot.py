from __future__ import annotations

import sqlite3
from pathlib import Path

import scripts.openclaw_training_autopilot as mod


def _mk_profitability_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        "\n".join(
            [
                "CREATE TABLE trade_executions (",
                "  id INTEGER PRIMARY KEY,",
                "  ticker TEXT,",
                "  trade_date TEXT,",
                "  realized_pnl REAL,",
                "  data_source TEXT,",
                "  action TEXT,",
                "  execution_mode TEXT,",
                "  is_close INTEGER,",
                "  entry_trade_id INTEGER",
                ")",
            ]
        )
    )

    # 30 closes across 21 trading days, net positive.
    for i in range(30):
        day = 1 + (i % 21)
        pnl = 10.0 if (i % 3) != 0 else -5.0
        cur.execute(
            "INSERT INTO trade_executions "
            "(ticker, trade_date, realized_pnl, data_source, action, execution_mode, is_close, entry_trade_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "AAPL",
                f"2026-01-{day:02d}T10:00:00Z",
                pnl,
                "yfinance",
                "SELL",
                "live",
                1,
                1,
            ),
        )
    conn.commit()
    conn.close()


def test_forecaster_eval_applies_factor_slack(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        mod.json.dumps(
            {
                "summary": {
                    "prod_like_conf_on": {
                        "errors": 0,
                        "ensemble_under_best_rate": 0.2520,
                        "avg_ensemble_ratio_vs_best": 1.10,
                        "ensemble_worse_than_rw_rate": 0.10,
                    }
                },
                "thresholds": {
                    "max_ensemble_under_best_rate": 0.25,
                    "max_avg_ensemble_ratio_vs_best": 1.2,
                    "max_ensemble_worse_than_rw_rate": 0.3,
                    "require_zero_errors": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    strict = mod._forecaster_eval(
        suite_path=suite_path,
        monitor_config=tmp_path / "unused.yml",
        factor=1.0,
        max_age_hours=999,
    )
    assert strict["status"] == "FAIL"

    tolerant = mod._forecaster_eval(
        suite_path=suite_path,
        monitor_config=tmp_path / "unused.yml",
        factor=0.989,
        max_age_hours=999,
    )
    assert tolerant["status"] == "PASS"


def test_profitability_eval_passes_minimal_db(tmp_path: Path) -> None:
    db_path = tmp_path / "pmx.db"
    _mk_profitability_db(db_path)

    req_path = tmp_path / "req.yml"
    req_path.write_text(
        "\n".join(
            [
                "profitability_proof_requirements:",
                "  data_quality:",
                "    min_data_source_coverage: 1.0",
                "    max_synthetic_ticker_pct: 0.0",
                "    allowed_execution_modes: ['live', 'paper']",
                "  statistical_significance:",
                "    min_closed_trades: 30",
                "    min_trading_days: 21",
                "    max_win_rate: 0.85",
                "    min_win_rate: 0.35",
                "  performance:",
                "    min_profit_factor: 1.1",
                "    max_drawdown: 0.30",
                "    min_sharpe_ratio: 0.3",
                "  audit_trail:",
                "    require_entry_exit_matching: true",
            ]
        ),
        encoding="utf-8",
    )

    result = mod._profitability_eval(db_path=db_path, requirements_path=req_path, factor=0.989)
    assert result["status"] == "PASS"
    assert not result["reasons"]


def test_autopilot_starts_training_when_benchmarks_fail(tmp_path: Path, monkeypatch) -> None:
    # Avoid writing into repo logs during tests.
    monkeypatch.setattr(mod, "TRAINING_PROPOSALS_DIR", tmp_path / "proposals")
    monkeypatch.setattr(mod, "TRAINING_FEEDBACK_DIR", tmp_path / "feedback")

    # Force a failing forecaster eval and a passing profitability eval.
    monkeypatch.setattr(mod, "_profitability_eval", lambda **_: {"status": "PASS", "reasons": [], "metrics": {}, "requirements": {}})
    monkeypatch.setattr(mod, "_forecaster_eval", lambda **_: {"status": "FAIL", "breaches": ["x"], "reasons": []})

    class _DummyProc:
        pid = 4242

    monkeypatch.setattr(mod, "_launch_training_detached", lambda **_: _DummyProc())

    state_file = tmp_path / "state.json"
    rc = mod.main(
        [
            "--state-file",
            str(state_file),
            "--training-output-json",
            str(tmp_path / "out.json"),
            "--training-log-file",
            str(tmp_path / "train.log"),
        ]
    )
    assert rc == 0
    state = mod._read_json(state_file)
    assert state.get("pid") == 4242

