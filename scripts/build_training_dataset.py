"""
Read-only curation of filtered trade and audit datasets for training.

Key hardening rule:
If eligibility data exists and explicitly yields zero HEALTHY tickers, the
curation path fails closed. It does not silently fall back to "include all".
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import time
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality_pipeline_common import (
    append_threshold_hash_change_warning,
    coalesce_expr,
    connect_ro,
    first_existing_columns,
    has_production_closed_trades_view,
    load_json_dict,
    table_columns,
)
from scripts.robustness_thresholds import threshold_map

log = logging.getLogger(__name__)

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits"
DEFAULT_ELIGIBILITY = ROOT / "logs" / "ticker_eligibility.json"
DEFAULT_OUT_TRADES = ROOT / "data" / "training" / "trades_filtered.parquet"
DEFAULT_OUT_AUDITS = ROOT / "data" / "training" / "audits_filtered.parquet"
DEFAULT_SUMMARY_OUT = ROOT / "logs" / "training_dataset_latest.json"

PHASE_715F_CUTOFF = "2025-07-01"
MIN_TICKERS_FOR_WARN = 2


def _load_eligibility_state(eligibility_path: Path) -> dict[str, Any]:
    payload, error = load_json_dict(eligibility_path)
    if error == "missing":
        return {"healthy_tickers": None, "mode": "missing", "warnings": ["eligibility_missing"]}
    if error:
        log.warning("Could not load eligibility file %s; include-all fallback.", eligibility_path)
        return {"healthy_tickers": None, "mode": "unreadable", "warnings": ["eligibility_unreadable"]}

    tickers = payload.get("tickers", {}) if isinstance(payload, dict) else {}
    if not isinstance(tickers, dict):
        return {"healthy_tickers": None, "mode": "unreadable", "warnings": ["eligibility_invalid"]}

    healthy = sorted(
        str(ticker).upper()
        for ticker, info in tickers.items()
        if isinstance(info, dict) and info.get("status") == "HEALTHY"
    )
    if not healthy:
        return {
            "healthy_tickers": set(),
            "mode": "explicit_zero_healthy",
            "warnings": ["eligibility_zero_healthy_fail_closed"],
        }
    return {"healthy_tickers": set(healthy), "mode": "healthy_only", "warnings": []}


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        num = float(numerator)
        den = float(denominator)
    except Exception:
        return None
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return None
    return num / den


def _build_trades_df(
    db_path: Path,
    healthy_tickers: Optional[set[str]],
    min_date: str,
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    if not db_path.exists():
        log.warning("DB not found: %s", db_path)
        return pd.DataFrame(), {
            "n_total": 0,
            "n_filtered": 0,
            "n_excluded_date": 0,
            "n_excluded_ticker": 0,
            "tickers_included": [],
        }

    try:
        conn = connect_ro(db_path)
    except Exception as exc:
        log.warning("Could not open DB %s: %s", db_path, exc)
        return pd.DataFrame(), {
            "n_total": 0,
            "n_filtered": 0,
            "n_excluded_date": 0,
            "n_excluded_ticker": 0,
            "tickers_included": [],
        }

    try:
        te_cols = table_columns(conn, "trade_executions")
        conf_expr = coalesce_expr(
            "src",
            first_existing_columns(te_cols, ("base_confidence", "confidence_calibrated", "effective_confidence")),
        )
        if has_production_closed_trades_view(conn):
            query = f"""
                SELECT src.id, src.ticker, src.trade_date, src.action, src.price, src.exit_price, src.realized_pnl,
                       src.holding_period_days, src.exit_reason, {conf_expr} AS confidence, src.ts_signal_id
                FROM production_closed_trades src
                ORDER BY src.trade_date ASC
            """
        else:
            query = f"""
                SELECT src.id, src.ticker, src.trade_date, src.action, src.price, src.exit_price, src.realized_pnl,
                       src.holding_period_days, src.exit_reason, {conf_expr} AS confidence, src.ts_signal_id
                FROM trade_executions src
                WHERE src.is_close = 1
                  AND src.realized_pnl IS NOT NULL
                  AND COALESCE(src.is_diagnostic, 0) = 0
                  AND COALESCE(src.is_synthetic, 0) = 0
                ORDER BY src.trade_date ASC
            """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    n_total = len(df)
    df_dated = df[df["trade_date"] >= min_date].copy() if "trade_date" in df.columns else df.copy()
    n_excluded_date = n_total - len(df_dated)

    if healthy_tickers is not None:
        df_filtered = df_dated[df_dated["ticker"].isin(healthy_tickers)].copy()
        n_excluded_ticker = len(df_dated) - len(df_filtered)
    else:
        df_filtered = df_dated
        n_excluded_ticker = 0

    tickers_included = sorted(str(t).upper() for t in df_filtered["ticker"].dropna().unique().tolist())
    stats = {
        "n_total": n_total,
        "n_excluded_date": n_excluded_date,
        "n_excluded_ticker": n_excluded_ticker,
        "n_filtered": len(df_filtered),
        "tickers_included": tickers_included,
    }
    return df_filtered, stats


def _build_audits_df(
    audit_dir: Path,
    min_date: str,
    healthy_tickers: Optional[set[str]],
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    if not audit_dir.exists():
        log.warning("Audit dir not found: %s", audit_dir)
        return pd.DataFrame(), {
            "n_total": 0,
            "n_filtered": 0,
            "n_excluded_format": 0,
            "n_excluded_date": 0,
            "n_excluded_ticker": 0,
        }

    files = sorted(audit_dir.glob("forecast_audit_*.json"))
    n_excluded_format = 0
    n_excluded_date = 0
    n_excluded_ticker = 0
    records: list[dict[str, Any]] = []

    for file_path in files:
        payload, error = load_json_dict(file_path)
        if error:
            n_excluded_format += 1
            continue

        evaluation_metrics = payload.get("evaluation_metrics")
        if not isinstance(evaluation_metrics, dict):
            n_excluded_format += 1
            continue
        ensemble = evaluation_metrics.get("ensemble")
        best_single = evaluation_metrics.get("best_single")
        if not isinstance(ensemble, dict) or not isinstance(best_single, dict):
            n_excluded_format += 1
            continue

        window_end = (
            str(payload.get("window_end") or "")[:10]
            or str(payload.get("generation_ts") or "")[:10]
        )
        if window_end and window_end < min_date:
            n_excluded_date += 1
            continue

        ticker = str(payload.get("ticker") or "").upper()
        if healthy_tickers is not None and ticker and ticker not in healthy_tickers:
            n_excluded_ticker += 1
            continue

        records.append(
            {
                "file": file_path.name,
                "ticker": ticker,
                "window_end": window_end,
                "window_id": payload.get("window_id"),
                "ensemble_rmse": ensemble.get("rmse"),
                "best_single_rmse": best_single.get("rmse"),
                "best_single_model": best_single.get("model"),
                "rmse_ratio": _safe_ratio(ensemble.get("rmse"), best_single.get("rmse")),
                "lift_fraction_global": payload.get("lift_fraction_global"),
            }
        )

    df = pd.DataFrame(records)
    stats = {
        "n_total": len(files),
        "n_excluded_format": n_excluded_format,
        "n_excluded_date": n_excluded_date,
        "n_excluded_ticker": n_excluded_ticker,
        "n_filtered": len(df),
    }
    return df, stats


def build_training_datasets(
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    eligibility_path: Path = DEFAULT_ELIGIBILITY,
    out_trades: Path = DEFAULT_OUT_TRADES,
    out_audits: Path = DEFAULT_OUT_AUDITS,
    min_date: str = PHASE_715F_CUTOFF,
    dry_run: bool = False,
) -> dict[str, Any]:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return {
            "error": "pandas not available; install with: pip install pandas pyarrow",
            "status": "ERROR",
            "trades": {"n_filtered": 0},
            "audits": {"n_filtered": 0},
            "thresholds": threshold_map(),
        }

    eligibility_state = _load_eligibility_state(eligibility_path)
    healthy_tickers = eligibility_state["healthy_tickers"]
    fail_closed = eligibility_state["mode"] == "explicit_zero_healthy"
    fail_closed_reason = None
    warnings = list(eligibility_state["warnings"])

    if fail_closed:
        fail_closed_reason = "eligibility_exists_with_zero_healthy_tickers"

    errors: list[str] = []

    if fail_closed:
        import pandas as pd

        trades_df = pd.DataFrame()
        audits_df = pd.DataFrame()
        trades_stats = {
            "n_total": 0,
            "n_excluded_date": 0,
            "n_excluded_ticker": 0,
            "n_filtered": 0,
            "tickers_included": [],
            "output": "skipped (fail-closed)",
        }
        audits_stats = {
            "n_total": 0,
            "n_excluded_format": 0,
            "n_excluded_date": 0,
            "n_excluded_ticker": 0,
            "n_filtered": 0,
            "output": "skipped (fail-closed)",
        }
    else:
        trades_df, trades_stats = _build_trades_df(db_path, healthy_tickers, min_date)
        audits_df, audits_stats = _build_audits_df(audit_dir, min_date, healthy_tickers)

    if len(trades_df) > 0 and not dry_run and not fail_closed:
        out_trades.parent.mkdir(parents=True, exist_ok=True)
        try:
            trades_df.to_parquet(str(out_trades), index=False, engine="pyarrow")
            trades_stats["output"] = str(out_trades)
        except Exception as exc:
            log.error("Failed to write trades parquet: %s", exc)
            trades_stats["error"] = str(exc)
            errors.append("trades_write_failed")
    elif dry_run:
        trades_stats["dry_run"] = True
    elif not fail_closed and len(trades_df) == 0:
        trades_stats["output"] = "skipped (empty dataset)"

    if len(audits_df) > 0 and not dry_run and not fail_closed:
        out_audits.parent.mkdir(parents=True, exist_ok=True)
        try:
            audits_df.to_parquet(str(out_audits), index=False, engine="pyarrow")
            audits_stats["output"] = str(out_audits)
        except Exception as exc:
            log.error("Failed to write audits parquet: %s", exc)
            audits_stats["error"] = str(exc)
            errors.append("audits_write_failed")
    elif dry_run:
        audits_stats["dry_run"] = True
    elif not fail_closed and len(audits_df) == 0:
        audits_stats["output"] = "skipped (empty dataset)"

    if (
        not fail_closed
        and trades_stats["n_filtered"] > 0
        and len(trades_stats.get("tickers_included", [])) < MIN_TICKERS_FOR_WARN
    ):
        warnings.append("low_ticker_diversity")
        log.warning("Trades dataset contains fewer than %s tickers.", MIN_TICKERS_FOR_WARN)

    eligible_used = sorted(healthy_tickers) if isinstance(healthy_tickers, set) else None

    status = "ERROR" if errors else ("WARN" if (fail_closed or warnings) else "PASS")
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "min_date_filter": min_date,
        "healthy_tickers_from_eligibility": eligible_used,
        "eligible_tickers_used": eligible_used,
        "eligibility_mode": eligibility_state["mode"],
        "fail_closed": fail_closed,
        "fail_closed_reason": fail_closed_reason,
        "thresholds": threshold_map(),
        "warnings": warnings,
        "errors": errors,
        "trades": trades_stats,
        "audits": audits_stats,
        "note": (
            "Filtered dataset excludes pre-7.15-F audit files and non-HEALTHY tickers when "
            "eligibility data is available. This workflow is read-only and reproducible."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build curated training datasets from production trade and audit data. "
            "Read-only: never modifies gate logic or thresholds."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--out-trades", type=Path, default=DEFAULT_OUT_TRADES)
    parser.add_argument("--out-audits", type=Path, default=DEFAULT_OUT_AUDITS)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--min-date", type=str, default=PHASE_715F_CUTOFF)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = build_training_datasets(
        db_path=args.db,
        audit_dir=args.audit_dir,
        eligibility_path=args.eligibility,
        out_trades=args.out_trades,
        out_audits=args.out_audits,
        min_date=args.min_date,
        dry_run=args.dry_run,
    )
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["started_at"] = started_at
    result["finished_at"] = finished_at
    result["duration_seconds"] = round(max(0.0, time.perf_counter() - started), 4)
    append_threshold_hash_change_warning(args.summary_out, result)
    if result.get("warnings") and result.get("status") == "PASS":
        result["status"] = "WARN"
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
        return 1 if (result.get("status") == "ERROR" or result.get("fail_closed")) else 0

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return 1
    if result.get("status") == "ERROR":
        print("[ERROR] Training dataset build encountered write failures.")
        for error in result.get("errors", []):
            print(f"  - {error}")
        return 1

    print(f"Training dataset build {'(DRY RUN) ' if args.dry_run else ''}complete.")
    print(
        f"  Trades: {result['trades']['n_filtered']} kept"
        f"{' [FAIL-CLOSED]' if result['fail_closed'] else ''}"
    )
    print(
        f"  Audits: {result['audits']['n_filtered']} kept"
        f"{' [FAIL-CLOSED]' if result['fail_closed'] else ''}"
    )
    print(f"  Summary: {args.summary_out}")
    return 1 if result.get("fail_closed") else 0


if __name__ == "__main__":
    sys.exit(main())
