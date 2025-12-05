#!/usr/bin/env python3
"""
Summarize quant validation results for Time Series signals.

Reads logs/signals/quant_validation.jsonl and prints:
  - PASS/FAIL/SKIPPED counts
  - Per-ticker metrics aggregates (median profit_factor, win_rate, etc.)
  - Most common failed criteria

This is a read-only helper intended for brutal/regression workflows and
manual inspection; it does not modify any database or config.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_PATH = Path("logs/signals/quant_validation.jsonl")
DEFAULT_MONITORING_CONFIG = Path("config/forecaster_monitoring.yml")


@dataclass
class TickerSummary:
    ticker: str
    count: int
    pass_count: int
    fail_count: int
    median_profit_factor: Optional[float]
    median_win_rate: Optional[float]
    median_annual_return: Optional[float]


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    if n % 2 == 1:
        return xs_sorted[mid]
    return 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])


def load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"quant_validation log not found at {path}")

    entries: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        raise SystemExit("No quant validation entries found.")
    return entries


def _load_monitoring_thresholds(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml  # Local import to keep dependency optional
    except ImportError:
        return {}

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    fm = raw.get("forecaster_monitoring") or {}
    return fm


def summarize(
    entries: List[Dict[str, Any]],
    top_n: int = 20,
    monitoring_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    status_counts = Counter()
    failed_criteria = Counter()

    by_ticker: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "profit_factor": [],
            "win_rate": [],
            "annual_return": [],
            "statuses": [],
        }
    )

    for rec in entries:
        ticker = rec.get("ticker") or "UNKNOWN"
        qv = rec.get("quant_validation") or {}
        status = rec.get("status") or qv.get("status") or "UNKNOWN"
        status_counts[status] += 1

        metrics = qv.get("metrics") or {}
        pf = metrics.get("profit_factor")
        wr = metrics.get("win_rate")
        ar = metrics.get("annual_return")
        if isinstance(pf, (int, float)):
            by_ticker[ticker]["profit_factor"].append(float(pf))
        if isinstance(wr, (int, float)):
            by_ticker[ticker]["win_rate"].append(float(wr))
        if isinstance(ar, (int, float)):
            by_ticker[ticker]["annual_return"].append(float(ar))
        by_ticker[ticker]["statuses"].append(status)

        failed = rec.get("failed_criteria") or qv.get("failed_criteria") or []
        for c in failed:
            failed_criteria[c] += 1

    print("=== Quant Validation Global Status ===")
    total = sum(status_counts.values())
    for status, cnt in status_counts.items():
        pct = (cnt / total * 100.0) if total else 0.0
        print(f"  {status:<6} : {cnt:4d} ({pct:5.1f}%)")

    print("\n=== Top Failed Criteria ===")
    if failed_criteria:
        for crit, cnt in failed_criteria.most_common():
            print(f"  {crit:<18} : {cnt:4d}")
    else:
        print("  (no failed criteria recorded)")

    print("\n=== Per-Ticker Summary (median metrics) ===")
    summaries: List[TickerSummary] = []
    for ticker, buckets in by_ticker.items():
        statuses = buckets["statuses"]
        summaries.append(
            TickerSummary(
                ticker=ticker,
                count=len(statuses),
                pass_count=sum(1 for s in statuses if s == "PASS"),
                fail_count=sum(1 for s in statuses if s == "FAIL"),
                median_profit_factor=_median(buckets["profit_factor"]),
                median_win_rate=_median(buckets["win_rate"]),
                median_annual_return=_median(buckets["annual_return"]),
            )
        )

    # Order: most observations, then by median profit factor descending.
    summaries.sort(key=lambda s: (-s.count, -(s.median_profit_factor or 0.0)))

    gv = (monitoring_cfg or {}).get("quant_validation") or {}
    per_ticker_cfg = (monitoring_cfg or {}).get("per_ticker") or {}

    # Production thresholds (GREEN tier)
    global_min_pf = gv.get("min_profit_factor")
    global_min_wr = gv.get("min_win_rate")
    global_min_ar = gv.get("min_annual_return")
    global_min_pass_rate = gv.get("min_pass_rate")

    # Softer research thresholds (YELLOW tier); fall back to production
    # values when warn_* is not explicitly set.
    global_warn_pf = gv.get("warn_profit_factor", global_min_pf)
    global_warn_wr = gv.get("warn_win_rate", global_min_wr)
    global_warn_ar = gv.get("warn_annual_return", global_min_ar)
    global_warn_pass_rate = gv.get("warn_pass_rate", global_min_pass_rate)

    header = (
        f"{'Ticker':<8} {'Count':>5} {'PASS':>5} {'FAIL':>5} "
        f"{'med_PF':>8} {'med_WR':>8} {'med_AnnRet':>11} {'Tier':<7} {'Alerts':<20}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries[:top_n]:
        pf_str = (
            f"{s.median_profit_factor:.2f}"
            if s.median_profit_factor is not None
            else "n/a"
        )
        wr_str = (
            f"{s.median_win_rate:.2f}"
            if s.median_win_rate is not None
            else "n/a"
        )
        ar_str = (
            f"{s.median_annual_return:.2f}"
            if s.median_annual_return is not None
            else "n/a"
        )
        alerts: List[str] = []

        # Determine thresholds (per-ticker overrides take precedence)
        cfg = per_ticker_cfg.get(s.ticker, {})
        min_pf = cfg.get("min_profit_factor", global_min_pf)
        min_wr = cfg.get("min_win_rate", global_min_wr)
        min_ar = cfg.get("min_annual_return", global_min_ar)
        min_pass_rate = cfg.get("min_pass_rate", global_min_pass_rate)

        warn_pf = cfg.get("warn_profit_factor", global_warn_pf)
        warn_wr = cfg.get("warn_win_rate", global_warn_wr)
        warn_ar = cfg.get("warn_annual_return", global_warn_ar)
        warn_pass_rate = cfg.get("warn_pass_rate", global_warn_pass_rate)

        # Numeric values for tier classification
        pf_val = s.median_profit_factor if s.median_profit_factor is not None else 0.0
        wr_val = s.median_win_rate if s.median_win_rate is not None else 0.0
        ar_val = s.median_annual_return if s.median_annual_return is not None else 0.0
        pass_rate = s.pass_count / s.count if s.count > 0 else 0.0

        def _meets_thresholds(
            pf_threshold: Optional[float],
            wr_threshold: Optional[float],
            ar_threshold: Optional[float],
            pass_rate_threshold: Optional[float],
        ) -> bool:
            ok_pf = True if pf_threshold is None else pf_val >= float(pf_threshold)
            ok_wr = True if wr_threshold is None else wr_val >= float(wr_threshold)
            ok_ar = True if ar_threshold is None else ar_val >= float(ar_threshold)
            ok_pr = (
                True
                if pass_rate_threshold is None
                else pass_rate >= float(pass_rate_threshold)
            )
            return ok_pf and ok_wr and ok_ar and ok_pr

        production_ok = _meets_thresholds(min_pf, min_wr, min_ar, min_pass_rate)
        research_ok = _meets_thresholds(warn_pf, warn_wr, warn_ar, warn_pass_rate)

        if production_ok:
            tier = "GREEN"
        elif research_ok:
            tier = "YELLOW"
        else:
            tier = "RED"

        # Alerts are driven by production thresholds so operators
        # can quickly see which dimensions block GREEN status.
        if isinstance(min_pf, (int, float)) and s.median_profit_factor is not None:
            if s.median_profit_factor < float(min_pf):
                alerts.append("PF<min")
        if isinstance(min_wr, (int, float)) and s.median_win_rate is not None:
            if s.median_win_rate < float(min_wr):
                alerts.append("WR<min")
        if isinstance(min_ar, (int, float)) and s.median_annual_return is not None:
            if s.median_annual_return < float(min_ar):
                alerts.append("AnnRet<min")
        if isinstance(min_pass_rate, (int, float)) and s.count > 0:
            if pass_rate < float(min_pass_rate):
                alerts.append("PASS_rate<min")

        alerts_str = ",".join(alerts) if alerts else ""
        print(
            f"{s.ticker:<8} {s.count:5d} {s.pass_count:5d} {s.fail_count:5d} "
            f"{pf_str:>8} {wr_str:>8} {ar_str:>11} {tier:<7} {alerts_str:<20}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize quant validation metrics for Time Series signals."
    )
    parser.add_argument(
        "--log-path",
        default=str(LOG_PATH),
        help="Path to quant_validation.jsonl (default: logs/signals/quant_validation.jsonl)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of tickers to show in per-ticker summary (default: 20)",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help=(
            "Optional path to forecaster_monitoring.yml "
            "(default: config/forecaster_monitoring.yml if present)"
        ),
    )
    args = parser.parse_args()

    entries = load_entries(Path(args.log_path))
    cfg_path = Path(args.config_path) if args.config_path else None
    monitoring_cfg = _load_monitoring_thresholds(cfg_path)
    summarize(entries, top_n=args.top_n, monitoring_cfg=monitoring_cfg)


if __name__ == "__main__":
    main()

