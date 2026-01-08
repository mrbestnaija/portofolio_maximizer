#!/usr/bin/env python3
"""
check_dashboard_health.py
-------------------------

Tiny helper to inspect the latest dashboard JSON snapshot and surface
Time Series forecaster health plus a quick per-ticker PnL summary.

It reads:
  - visualizations/dashboard_data.json by default
  - config/forecaster_monitoring.yml for PF/WR/RMSE thresholds

and prints:
  - Run metadata (run_id, timestamp, tickers, cycles)
  - Forecaster health vs thresholds (profit_factor_ok, win_rate_ok, rmse_ok)
  - Per-ticker trade counts, win-rate, and profit_factor from the last
    few signals in the dashboard payload.

This is a read-only CLI intended for brutal/manual inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
DEFAULT_DASHBOARD_PATH = ROOT_PATH / "visualizations" / "dashboard_data.json"
DEFAULT_MONITORING_CONFIG = ROOT_PATH / "config" / "forecaster_monitoring.yml"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"dashboard JSON not found at {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse dashboard JSON at {path}: {exc}") from exc


def _load_monitoring_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # lazy import; already a project dependency
    except ImportError:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("forecaster_monitoring") or {}


def _summarize_forecaster_health(
    payload: Dict[str, Any],
) -> None:
    fh = payload.get("forecaster_health") or {}
    thresholds = fh.get("thresholds") or {}
    metrics = fh.get("metrics") or {}
    status = fh.get("status") or {}

    print("=== Forecaster Health (run-level) ===")
    if not fh:
        print("  (no forecaster_health block in dashboard JSON)")
        return

    print("  Thresholds:")
    print(
        f"    min_profit_factor : {thresholds.get('profit_factor_min')!r}  "
        f"min_win_rate : {thresholds.get('win_rate_min')!r}  "
        f"max_rmse_ratio : {thresholds.get('rmse_ratio_max')!r}"
    )

    pf = (metrics.get("profit_factor"), status.get("profit_factor_ok"))
    wr = (metrics.get("win_rate"), status.get("win_rate_ok"))
    rm = metrics.get("rmse") or {}
    rm_status = status.get("rmse_ok")

    print("  Metrics:")
    print(f"    profit_factor : {pf[0]!r}  ok={pf[1]!r}")
    print(f"    win_rate      : {wr[0]!r}  ok={wr[1]!r}")
    print(
        f"    rmse          : ensemble={rm.get('ensemble')!r}, "
        f"baseline={rm.get('baseline')!r}, ratio={rm.get('ratio')!r}  ok={rm_status!r}"
    )


def _summarize_tickers(
    payload: Dict[str, Any],
    monitoring_cfg: Dict[str, Any],
) -> None:
    signals: List[Dict[str, Any]] = payload.get("signals") or []
    if not signals:
        print("\n=== Per-Ticker Summary ===")
        print("  (no signals in dashboard payload)")
        return

    qv_cfg = monitoring_cfg.get("quant_validation") or {}
    per_ticker_cfg = monitoring_cfg.get("per_ticker") or {}
    global_min_pf = qv_cfg.get("min_profit_factor")
    global_min_wr = qv_cfg.get("min_win_rate")

    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }
    )

    for sig in signals:
        ticker = str(sig.get("ticker") or "UNKNOWN")
        status = sig.get("status")
        if status != "EXECUTED":
            continue
        pnl = sig.get("realized_pnl")
        if not isinstance(pnl, (int, float)):
            continue
        b = buckets[ticker]
        b["count"] += 1
        if pnl > 0:
            b["wins"] += 1
            b["gross_profit"] += float(pnl)
        elif pnl < 0:
            b["losses"] += 1
            b["gross_loss"] += abs(float(pnl))

    print("\n=== Per-Ticker Summary (from dashboard signals) ===")
    if not buckets:
        print("  (no executed trades in recent signals)")
        return

    header = (
        f"{'Ticker':<8} {'Trades':>6} {'Wins':>5} {'Losses':>7} "
        f"{'WinRate':>8} {'PF':>8} {'Alerts':<20}"
    )
    print(header)
    print("-" * len(header))

    for ticker, b in sorted(buckets.items(), key=lambda kv: kv[0]):
        count = b["count"]
        wins = b["wins"]
        losses = b["losses"]
        win_rate = wins / count if count else 0.0
        if b["gross_loss"] > 0:
            profit_factor = b["gross_profit"] / b["gross_loss"]
        else:
            profit_factor = float("inf") if b["gross_profit"] > 0 else 0.0

        cfg = per_ticker_cfg.get(ticker, {})
        min_pf = cfg.get("min_profit_factor", global_min_pf)
        min_wr = cfg.get("min_win_rate", global_min_wr)

        alerts: List[str] = []
        if isinstance(min_pf, (int, float)) and profit_factor != float("inf"):
            if profit_factor < float(min_pf):
                alerts.append("PF<min")
        if isinstance(min_wr, (int, float)):
            if win_rate < float(min_wr):
                alerts.append("WR<min")

        wr_str = f"{win_rate:.2f}"
        pf_str = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        alerts_str = ",".join(alerts)
        print(
            f"{ticker:<8} {count:6d} {wins:5d} {losses:7d} "
            f"{wr_str:>8} {pf_str:>8} {alerts_str:<20}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect dashboard_data.json for TS forecaster health and per-ticker PnL."
    )
    parser.add_argument(
        "--dashboard-path",
        default=str(DEFAULT_DASHBOARD_PATH),
        help="Path to dashboard_data.json (default: visualizations/dashboard_data.json)",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help="Optional path to forecaster_monitoring.yml "
        "(default: config/forecaster_monitoring.yml if present)",
    )
    args = parser.parse_args()

    payload = _load_json(Path(args.dashboard_path))
    monitoring_cfg = _load_monitoring_cfg(Path(args.config_path))

    meta = payload.get("meta") or {}
    print("=== Run Metadata ===")
    print(f"  run_id  : {meta.get('run_id')}")
    print(f"  ts      : {meta.get('ts')}")
    print(f"  tickers : {meta.get('tickers')}")
    print(f"  cycles  : {meta.get('cycles')}")

    _summarize_forecaster_health(payload)
    _summarize_tickers(payload, monitoring_cfg)


if __name__ == "__main__":
    main()

