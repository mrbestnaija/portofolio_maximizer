import json
from pathlib import Path

from scripts.capture_baseline_snapshot import capture_baseline_snapshot


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_capture_baseline_snapshot_writes_manifest_and_artifacts(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "logs" / "automation").mkdir(parents=True)
    (root / "logs" / "signals").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "visualizations").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "execution").mkdir(parents=True)

    # Minimal baseline files
    (root / "config" / "yfinance_config.yml").write_text("interval: 1d\n", encoding="utf-8")
    (root / "config" / "execution_cost_model.yml").write_text("cost: 0.0\n", encoding="utf-8")
    (root / "config" / "signal_routing_config.yml").write_text("min_confidence: 0.5\n", encoding="utf-8")
    (root / "config" / "quant_success_config.yml").write_text("min_profit_factor: 1.0\n", encoding="utf-8")
    (root / "config" / "forecaster_monitoring.yml").write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    (root / "scripts" / "run_auto_trader.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "models" / "time_series_signal_generator.py").write_text("x=1\n", encoding="utf-8")
    (root / "execution" / "paper_trading_engine.py").write_text("y=2\n", encoding="utf-8")

    _write_jsonl(
        root / "logs" / "automation" / "run_summary.jsonl",
        [{"run_id": "RUN_1", "profitability": {"pnl_dollars": 0.0}}],
    )
    _write_jsonl(
        root / "logs" / "automation" / "execution_log.jsonl",
        [{"run_id": "RUN_1", "event": "A"}, {"run_id": "RUN_1", "event": "B"}],
    )
    _write_jsonl(
        root / "logs" / "signals" / "quant_validation.jsonl",
        [
            {
                "ticker": "AAPL",
                "status": "PASS",
                "quant_validation": {"metrics": {"profit_factor": 1.2, "win_rate": 0.6, "annual_return": 0.1}},
            }
        ],
    )
    (root / "visualizations" / "dashboard_data.json").write_text(
        json.dumps({"meta": {"run_id": "RUN_1"}}),
        encoding="utf-8",
    )
    (root / "reports" / "horizon_backtest_20260101_000000.json").write_text(
        json.dumps(
            {
                "metrics": {"profit_factor": 1.1},
                "provenance": {
                    "dataset_hash": "DATA_HASH_1",
                    "db_max_ohlcv_date": "2024-01-31",
                    "config_hash": "CFG_HASH_1",
                    "git_commit": "abc123",
                    "config_paths": [str(root / "config" / "signal_routing_config.yml")],
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "logs" / "automation" / "db_provenance_RUN_1.json").write_text(
        json.dumps({"db_max_ohlcv_date": "2024-01-31"}),
        encoding="utf-8",
    )

    result = capture_baseline_snapshot(root=root, out_dir=tmp_path / "out", tag="baseline", run_id=None)
    assert result.run_id == "RUN_1"
    assert result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "RUN_1"
    assert manifest["provenance"]["dataset_hash"] == "DATA_HASH_1"
    assert manifest["provenance"]["db_max_ohlcv_date"] == "2024-01-31"
    assert manifest["provenance"]["config_hash"]
    assert "git_commit" in manifest["provenance"]
    assert manifest["provenance"]["config_paths"]
    assert (result.snapshot_dir / "artifacts" / "run_summary_last.json").exists()
    assert (result.snapshot_dir / "artifacts" / "execution_log_tail.jsonl").exists()
    assert (result.snapshot_dir / "artifacts" / "quant_validation_summary.txt").exists()
    assert (result.snapshot_dir / "artifacts" / "dashboard_data.json").exists()
    assert (result.snapshot_dir / "artifacts" / "horizon_backtest_latest.json").exists()
