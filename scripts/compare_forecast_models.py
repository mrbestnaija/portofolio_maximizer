#!/usr/bin/env python3
"""
compare_forecast_models.py
--------------------------

Small CLI to inspect time-series forecast regression metrics per ticker and
compare baseline vs ensemble performance.

By default it compares the TS ensemble (`model_type='COMBINED'`) against the
SAMOSSA baseline (`model_type='SAMOSSA'`) and prints RMSE and directional
accuracy per ticker, flagging cases where the ensemble underperforms.

Usage:
  python -m scripts.compare_forecast_models \\
      --db-path data/portfolio_maximizer.db \\
      --start-date 2025-01-01 --end-date 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import site
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
site.addsitedir(str(ROOT_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager


def _aggregate_by_ticker(
    db: DatabaseManager,
    model_type: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate regression_metrics per ticker for a given model_type.
    """
    params: list[Any] = [model_type]
    where_clauses = ["model_type = ?"]

    if start_date:
        where_clauses.append("forecast_date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("forecast_date <= ?")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT ticker, regression_metrics
        FROM time_series_forecasts
        WHERE {where_sql}
    """

    cursor = db.cursor.execute(query, params)

    aggregates: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for row in cursor.fetchall():
        ticker = row["ticker"]
        raw = row["regression_metrics"]
        if not raw:
            continue
        try:
            metrics = json.loads(raw)
        except Exception:
            continue

        agg = aggregates.setdefault(
            ticker,
            {
                "rmse": 0.0,
                "directional_accuracy": 0.0,
            },
        )
        counts[ticker] = counts.get(ticker, 0) + 1

        rmse_val = metrics.get("rmse")
        da_val = metrics.get("directional_accuracy")
        if isinstance(rmse_val, (int, float)):
            agg["rmse"] += float(rmse_val)
        if isinstance(da_val, (int, float)):
            agg["directional_accuracy"] += float(da_val)

    for ticker, agg in aggregates.items():
        count = counts.get(ticker, 0) or 1
        agg["rmse"] = agg["rmse"] / count
        agg["directional_accuracy"] = agg["directional_accuracy"] / count

    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TS ensemble vs baseline regression metrics per ticker."
    )
    parser.add_argument(
        "--db-path",
        default="data/portfolio_maximizer.db",
        help="Path to SQLite database (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument(
        "--start-date",
        help="Optional ISO start date (YYYY-MM-DD) for filtering forecasts",
    )
    parser.add_argument(
        "--end-date",
        help="Optional ISO end date (YYYY-MM-DD) for filtering forecasts",
    )
    parser.add_argument(
        "--baseline-model",
        default="SAMOSSA",
        choices=["SAMOSSA", "SARIMAX"],
        help="Baseline model_type to compare against (default: SAMOSSA)",
    )
    parser.add_argument(
        "--max-rmse-ratio",
        type=float,
        default=1.0,
        help=(
            "Flag ticker when ensemble_rmse / baseline_rmse exceeds this value "
            "(default: 1.0, i.e. ensemble no worse than baseline)."
        ),
    )
    parser.add_argument(
        "--min-da-delta",
        type=float,
        default=0.0,
        help=(
            "Flag ticker when (ensemble_DA - baseline_DA) is below this delta "
            "(default: 0.0, i.e. ensemble directional accuracy must be at least "
            "as good as baseline)."
        ),
    )
    parser.add_argument(
        "--max-underperform-fraction",
        type=float,
        default=0.5,
        help=(
            "Exit with code 1 when the fraction of evaluated tickers that "
            "underperform (via RMSE and/or DA) exceeds this threshold "
            "(default: 0.5)."
        ),
    )
    args = parser.parse_args()

    db = DatabaseManager(db_path=args.db_path)

    ensemble = _aggregate_by_ticker(
        db=db,
        model_type="COMBINED",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    baseline = _aggregate_by_ticker(
        db=db,
        model_type=args.baseline_model,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    tickers = sorted(set(ensemble.keys()) | set(baseline.keys()))

    header = (
        f"{'Ticker':<8} "
        f"{'Ens_RMSE':>10} {'Base_RMSE':>10} {'RMSE_Ratio':>11} "
        f"{'Ens_DA':>8} {'Base_DA':>8} {'DA_Delta':>9} {'Flag':<10}"
    )
    print(header)
    print("-" * len(header))

    underperform_count = 0
    evaluated_count = 0

    for ticker in tickers:
        ens = ensemble.get(ticker, {})
        base = baseline.get(ticker, {})
        ens_rmse = ens.get("rmse")
        base_rmse = base.get("rmse")
        ens_da = ens.get("directional_accuracy")
        base_da = base.get("directional_accuracy")

        rmse_ratio = None
        if isinstance(ens_rmse, (int, float)) and isinstance(base_rmse, (int, float)) and base_rmse > 0:
            rmse_ratio = float(ens_rmse) / float(base_rmse)

        da_delta = None
        if isinstance(ens_da, (int, float)) and isinstance(base_da, (int, float)):
            da_delta = float(ens_da) - float(base_da)

        flag = ""
        # Underperformance rules: RMSE worse than allowed, or DA below delta.
        if rmse_ratio is not None and rmse_ratio > args.max_rmse_ratio:
            flag = "RMSE_worse"
        if da_delta is not None and da_delta < args.min_da_delta:
            flag = (flag + "|DA_worse") if flag else "DA_worse"

        # Track evaluated/underperform ticks where we have at least one metric.
        if rmse_ratio is not None or da_delta is not None:
            evaluated_count += 1
            if flag:
                underperform_count += 1

        def _fmt(x: Optional[float], width: int) -> str:
            if not isinstance(x, (int, float)):
                return "n/a".rjust(width)
            return f"{x:.4f}".rjust(width)

        print(
            f"{ticker:<8} "
            f"{_fmt(ens_rmse, 10)} {_fmt(base_rmse, 10)} {_fmt(rmse_ratio, 11)} "
            f"{_fmt(ens_da, 8)} {_fmt(base_da, 8)} {_fmt(da_delta, 9)} {flag:<10}"
        )

    if evaluated_count > 0:
        frac = underperform_count / evaluated_count
        summary = (
            f"Underperforming tickers: {underperform_count}/{evaluated_count} "
            f"({frac:.1%}) vs max_underperform_fraction={args.max_underperform_fraction:.1%}"
        )
        print(summary)
        if frac > args.max_underperform_fraction:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
