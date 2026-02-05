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
