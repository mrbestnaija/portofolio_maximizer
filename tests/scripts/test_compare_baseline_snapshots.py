from __future__ import annotations

import json
from pathlib import Path

from scripts import compare_baseline_snapshots


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_baseline_snapshots_detects_changed_files_and_metric_delta(tmp_path: Path) -> None:
    snap_a = tmp_path / "snap_a"
    snap_b = tmp_path / "snap_b"

    _write_json(
        snap_a / "manifest.json",
        {
            "provenance": {
                "dataset_hash": "hash_a",
                "db_max_ohlcv_date": "2024-01-31",
                "config_hash": "cfg_a",
                "git_commit": "commit_a",
                "config_paths": ["config/a.yml"],
            },
            "files": {
                "configs": [
                    {
                        "dst": str(snap_a / "configs" / "signal_routing_config.yml"),
                        "sha256": "aaa",
                    }
                ],
                "code": [],
            }
        },
    )
    _write_json(
        snap_b / "manifest.json",
        {
            "provenance": {
                "dataset_hash": "hash_b",
                "db_max_ohlcv_date": "2024-02-01",
                "config_hash": "cfg_b",
                "git_commit": "commit_b",
                "config_paths": ["config/b.yml"],
            },
            "files": {
                "configs": [
                    {
                        "dst": str(snap_b / "configs" / "signal_routing_config.yml"),
                        "sha256": "bbb",
                    }
                ],
                "code": [],
            }
        },
    )

    _write_json(snap_a / "artifacts" / "run_summary_last.json", {"profitability": {"pnl_dollars": 0.0}})
    _write_json(snap_b / "artifacts" / "run_summary_last.json", {"profitability": {"pnl_dollars": 1.0}})

    a = compare_baseline_snapshots.load_snapshot(snap_a)
    b = compare_baseline_snapshots.load_snapshot(snap_b)

    diffs = compare_baseline_snapshots.diff_files(a, b, category="configs")
    assert diffs["changed"] == ["configs/signal_routing_config.yml"]

    run_deltas = compare_baseline_snapshots.diff_metrics(
        compare_baseline_snapshots.extract_run_metrics(a.run_summary),
        compare_baseline_snapshots.extract_run_metrics(b.run_summary),
    )
    assert run_deltas["profitability.pnl_dollars"] == (0.0, 1.0, 1.0)
    markdown = compare_baseline_snapshots.render_markdown(
        snapshot_a=a,
        snapshot_b=b,
        file_diffs={"configs": diffs, "code": {"changed": [], "added": [], "removed": []}},
        run_metric_diffs=run_deltas,
        backtest_metric_diffs={},
    )
    assert "dataset_hash" in markdown
    assert "cfg_a" in markdown
    assert "cfg_b" in markdown
