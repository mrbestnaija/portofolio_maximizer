#!/usr/bin/env python3
"""
Compare two baseline snapshots captured under reports/baselines/.

Phase 10 helper: quickly see what changed (configs/code) and whether key run/backtest
metrics moved in the right direction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Snapshot:
    path: Path
    manifest: Dict[str, Any]
    run_summary: Optional[Dict[str, Any]] = None
    horizon_backtest: Optional[Dict[str, Any]] = None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def load_snapshot(path: Path) -> Snapshot:
    snapshot_dir = path
    if snapshot_dir.is_file() and snapshot_dir.name == "manifest.json":
        snapshot_dir = snapshot_dir.parent

    manifest_path = snapshot_dir / "manifest.json"
    manifest = _load_json(manifest_path) or {}

    run_summary = _load_json(snapshot_dir / "artifacts" / "run_summary_last.json")
    horizon_backtest = _load_json(snapshot_dir / "artifacts" / "horizon_backtest_latest.json")

    return Snapshot(
        path=snapshot_dir,
        manifest=manifest,
        run_summary=run_summary,
        horizon_backtest=horizon_backtest,
    )


def _file_sha_map(snapshot: Snapshot, category: str) -> Dict[str, str]:
    files = snapshot.manifest.get("files") if isinstance(snapshot.manifest.get("files"), dict) else {}
    rows = files.get(category) if isinstance(files.get(category), list) else []

    mapping: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dst = row.get("dst")
        sha = row.get("sha256")
        if not dst or not sha:
            continue
        try:
            rel = str(Path(dst).resolve().relative_to(snapshot.path.resolve()))
        except Exception:
            rel = str(dst)
        mapping[rel] = str(sha)
    return mapping


def diff_files(snapshot_a: Snapshot, snapshot_b: Snapshot, *, category: str) -> Dict[str, list[str]]:
    a_map = _file_sha_map(snapshot_a, category)
    b_map = _file_sha_map(snapshot_b, category)

    a_keys = set(a_map)
    b_keys = set(b_map)
    added = sorted(b_keys - a_keys)
    removed = sorted(a_keys - b_keys)
    changed = sorted(k for k in (a_keys & b_keys) if a_map.get(k) != b_map.get(k))

    return {"changed": changed, "added": added, "removed": removed}


def _extract_numeric(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    return None


def extract_run_metrics(run_summary: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(run_summary, dict):
        return {}

    profitability = run_summary.get("profitability") if isinstance(run_summary.get("profitability"), dict) else {}
    liquidity = run_summary.get("liquidity") if isinstance(run_summary.get("liquidity"), dict) else {}
    forecaster = run_summary.get("forecaster") if isinstance(run_summary.get("forecaster"), dict) else {}
    quant = run_summary.get("quant_validation") if isinstance(run_summary.get("quant_validation"), dict) else {}

    forecaster_metrics = forecaster.get("metrics") if isinstance(forecaster.get("metrics"), dict) else {}
    rmse = forecaster_metrics.get("rmse") if isinstance(forecaster_metrics.get("rmse"), dict) else {}

    return {
        "profitability.pnl_dollars": _extract_numeric(profitability.get("pnl_dollars")),
        "profitability.pnl_pct": _extract_numeric(profitability.get("pnl_pct")),
        "profitability.profit_factor": _extract_numeric(profitability.get("profit_factor")),
        "profitability.win_rate": _extract_numeric(profitability.get("win_rate")),
        "profitability.trades": _extract_numeric(profitability.get("trades")),
        "profitability.realized_trades": _extract_numeric(profitability.get("realized_trades")),
        "liquidity.cash": _extract_numeric(liquidity.get("cash")),
        "liquidity.total_value": _extract_numeric(liquidity.get("total_value")),
        "liquidity.cash_ratio": _extract_numeric(liquidity.get("cash_ratio")),
        "liquidity.open_positions": _extract_numeric(liquidity.get("open_positions")),
        "forecaster.rmse.ensemble": _extract_numeric(rmse.get("ensemble")),
        "forecaster.rmse.baseline": _extract_numeric(rmse.get("baseline")),
        "forecaster.rmse.ratio": _extract_numeric(rmse.get("ratio")),
        "quant.fail_fraction": _extract_numeric(quant.get("fail_fraction")),
        "quant.negative_expected_profit_fraction": _extract_numeric(quant.get("negative_expected_profit_fraction")),
    }


def extract_backtest_metrics(backtest: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(backtest, dict):
        return {}
    metrics = backtest.get("metrics") if isinstance(backtest.get("metrics"), dict) else {}
    return {
        "backtest.total_trades": _extract_numeric(metrics.get("total_trades")),
        "backtest.total_return": _extract_numeric(metrics.get("total_return")),
        "backtest.win_rate": _extract_numeric(metrics.get("win_rate")),
        "backtest.profit_factor": _extract_numeric(metrics.get("profit_factor")),
        "backtest.max_drawdown": _extract_numeric(metrics.get("max_drawdown")),
    }


def diff_metrics(
    metrics_a: Dict[str, Optional[float]],
    metrics_b: Dict[str, Optional[float]],
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    keys = sorted(set(metrics_a) | set(metrics_b))
    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for key in keys:
        a_val = metrics_a.get(key)
        b_val = metrics_b.get(key)
        delta = None
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            delta = float(b_val) - float(a_val)
        out[key] = (a_val, b_val, delta)
    return out


def render_markdown(
    *,
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    file_diffs: Dict[str, Dict[str, list[str]]],
    run_metric_diffs: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
    backtest_metric_diffs: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
) -> str:
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        return f"{v:.6g}"

    lines: list[str] = [
        "# Baseline Snapshot Diff",
        "",
        f"- A: {snapshot_a.path}",
        f"- B: {snapshot_b.path}",
        "",
        "## File Changes",
    ]
    for category, diffs in file_diffs.items():
        lines.append(f"### {category}")
        for bucket in ("changed", "added", "removed"):
            items = diffs.get(bucket) or []
            lines.append(f"- {bucket}: {len(items)}")
            for item in items[:10]:
                lines.append(f"  - {item}")
            if len(items) > 10:
                lines.append("  - ...")
        lines.append("")

    def _metric_table(title: str, diffs: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| metric | A | B | Δ |")
        lines.append("|---|---:|---:|---:|")
        for key, (a_val, b_val, delta) in diffs.items():
            lines.append(f"| `{key}` | {_fmt(a_val)} | {_fmt(b_val)} | {_fmt(delta)} |")
        lines.append("")

    _metric_table("Run Metrics (run_summary_last.json)", run_metric_diffs)
    if backtest_metric_diffs:
        _metric_table("Backtest Metrics (horizon_backtest_latest.json)", backtest_metric_diffs)

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two baseline snapshots and render a diff report.")
    parser.add_argument("--a", type=Path, required=True, help="Snapshot directory A (or its manifest.json).")
    parser.add_argument("--b", type=Path, required=True, help="Snapshot directory B (or its manifest.json).")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    parser.add_argument("--json", dest="emit_json", action="store_true", help="Emit JSON instead of markdown.")

    args = parser.parse_args(argv)
    snap_a = load_snapshot(args.a)
    snap_b = load_snapshot(args.b)

    file_diffs = {
        "configs": diff_files(snap_a, snap_b, category="configs"),
        "code": diff_files(snap_a, snap_b, category="code"),
    }

    run_metric_diffs = diff_metrics(extract_run_metrics(snap_a.run_summary), extract_run_metrics(snap_b.run_summary))
    backtest_metric_diffs = diff_metrics(
        extract_backtest_metrics(snap_a.horizon_backtest),
        extract_backtest_metrics(snap_b.horizon_backtest),
    )

    if args.emit_json:
        payload = {
            "a": str(snap_a.path),
            "b": str(snap_b.path),
            "file_diffs": file_diffs,
            "run_metrics": run_metric_diffs,
            "backtest_metrics": backtest_metric_diffs,
        }
        output = json.dumps(payload, indent=2, sort_keys=True, default=str)
    else:
        output = render_markdown(
            snapshot_a=snap_a,
            snapshot_b=snap_b,
            file_diffs=file_diffs,
            run_metric_diffs=run_metric_diffs,
            backtest_metric_diffs=backtest_metric_diffs,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

