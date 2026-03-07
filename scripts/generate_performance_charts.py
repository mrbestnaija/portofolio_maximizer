"""
Generate read-only performance charts plus a machine-readable metrics summary.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_DIR = str(ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from scripts.data_sufficiency_monitor import run_data_sufficiency
from scripts.compute_ticker_eligibility import load_per_ticker_aggregates
from scripts.quality_pipeline_common import append_threshold_hash_change_warning
from scripts.robustness_thresholds import (
    BREAK_EVEN_PROFIT_FACTOR,
    MIN_LIFT_FRACTION,
    POLICY_WR_FLOOR,
    R3_MIN_PROFIT_FACTOR,
    R3_MIN_WIN_RATE,
    threshold_map,
)

log = logging.getLogger(__name__)

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits"
DEFAULT_OUT_DIR = ROOT / "visualizations" / "performance"
DEFAULT_ELIGIBILITY = ROOT / "logs" / "ticker_eligibility.json"
DEFAULT_CONTEXT_QUALITY = ROOT / "logs" / "context_quality_latest.json"
DEFAULT_METRICS_PATH = DEFAULT_OUT_DIR / "metrics_summary.json"

R3_WIN_RATE = R3_MIN_WIN_RATE


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _lift_axis_bounds(
    *,
    lift_global_pct: float,
    lift_recent_pct: float,
    threshold_pct: float,
    ci_low_pct: float | None = None,
    ci_high_pct: float | None = None,
) -> tuple[float, float]:
    values = [0.0, lift_global_pct, lift_recent_pct, threshold_pct]
    if ci_low_pct is not None and math.isfinite(ci_low_pct):
        values.append(ci_low_pct)
    if ci_high_pct is not None and math.isfinite(ci_high_pct):
        values.append(ci_high_pct)
    lower = min(values)
    upper = max(values)
    span = max(upper - lower, 10.0)
    pad = max(2.0, span * 0.12)
    return lower - pad, upper + pad


def _load_per_ticker(db_path: Path) -> list[dict[str, Any]]:
    rows, errors = load_per_ticker_aggregates(db_path)
    if errors:
        log.warning("Per-ticker load warnings: %s", ",".join(errors))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "ticker": str(row.get("ticker") or "").upper(),
                "n": int(row.get("n_trades") or 0),
                "win_rate": _finite_float(row.get("win_rate")),
                "profit_factor": _finite_float(row.get("profit_factor")),
                "total_pnl": _finite_float(row.get("total_pnl")),
            }
        )
    return normalized


def _load_wr_over_time(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            raw = conn.execute(
                """
                SELECT trade_date, realized_pnl
                FROM production_closed_trades
                ORDER BY trade_date ASC
                """
            ).fetchall()
        finally:
            conn.close()
        wins = 0
        total = 0
        for row in raw:
            total += 1
            if _finite_float(row["realized_pnl"]) > 0:
                wins += 1
            rows.append({"date": row["trade_date"], "cumulative_wr": wins / total, "n": total})
    except Exception as exc:
        log.warning("WR-over-time load failed: %s", exc)
    return rows


def _load_lift_metrics(audit_dir: Path) -> dict[str, Any]:
    try:
        from check_model_improvement import run_layer1_forecast_quality

        result = run_layer1_forecast_quality(audit_dir=audit_dir)
        return result.metrics or {}
    except Exception as exc:
        log.warning("Layer 1 load failed: %s", exc)
        return {}


def _load_json(path: Path, label: str, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        warnings.append(f"{label}_unreadable")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"{label}_invalid")
        return {}
    return payload


def _load_calibration_metrics(db_path: Path) -> dict[str, Any]:
    try:
        from check_model_improvement import run_layer4_calibration

        result = run_layer4_calibration(db_path, ROOT / "logs" / "signals" / "quant_validation.jsonl")
        return result.metrics or {}
    except Exception:
        return {}


def _build_metrics_summary(
    per_ticker: list[dict[str, Any]],
    l1_metrics: dict[str, Any],
    eligibility: dict[str, Any],
    context: dict[str, Any],
    sufficiency: dict[str, Any],
    warnings: list[str],
    chart_paths: dict[str, str],
    db_path: Path,
) -> dict[str, Any]:
    n_trades = sum(int(row["n"]) for row in per_ticker)
    wins = sum(int(row["n"]) * float(row["win_rate"]) for row in per_ticker)
    overall_wr = round((wins / n_trades), 4) if n_trades else 0.0
    total_pnl = round(sum(_finite_float(row["total_pnl"]) for row in per_ticker), 2)
    eligibility_counts = eligibility.get("summary", {}) if isinstance(eligibility, dict) else {}
    context_summary = {
        "n_total_trades": context.get("n_total_trades", 0),
        "n_trades_no_confidence": context.get("n_trades_no_confidence", 0),
        "partial_data": bool(context.get("partial_data", False)),
        "regime_count": len(context.get("regime_quality", {}) or {}),
        "confidence_bin_count": len(context.get("confidence_bin_quality", {}) or {}),
    }
    calibration = _load_calibration_metrics(db_path)
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # NOTE: "status" is intentionally absent here — caller (generate_performance_artifacts)
        # sets it authoritatively after evaluating both warnings and errors.
        "overall_trade_metrics": {
            "overall_win_rate": overall_wr,
            "n_trades": n_trades,
            "n_tickers": len(per_ticker),
            "total_pnl": total_pnl,
        },
        "coverage_ratio": l1_metrics.get("coverage_ratio"),
        "lift_metrics": {
            "lift_fraction_global": l1_metrics.get("lift_fraction_global"),
            "lift_fraction_recent": l1_metrics.get("lift_fraction_recent"),
            "lift_ci_low": l1_metrics.get("lift_ci_low"),
            "lift_ci_high": l1_metrics.get("lift_ci_high"),
            "lift_ci_insufficient_data": l1_metrics.get("lift_ci_insufficient_data", True),
            "n_used_audit_windows": l1_metrics.get("n_used_windows") or l1_metrics.get("n_used"),
        },
        "calibration_metrics": {
            "calibration_tier": calibration.get("calibration_active_tier"),
            "brier_score": calibration.get("brier_score"),
            "ece": calibration.get("ece"),
        },
        "thresholds": threshold_map(),
        "eligibility_counts": eligibility_counts,
        "context_summary": context_summary,
        "sufficiency_status": sufficiency.get("status"),
        "sufficiency": sufficiency,
        "per_ticker": per_ticker,
        "chart_paths": chart_paths,
        "warnings": sorted(set(warnings)),
    }


def chart_per_ticker_wr_pf(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping per-ticker chart")
        return
    if not rows:
        return
    tickers = [row["ticker"] for row in rows]
    wrs = [_finite_float(row["win_rate"]) * 100 for row in rows]
    pfs = [min(max(_finite_float(row["profit_factor"]), 0.0), 5.0) for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    colors_wr = [
        "#d32f2f" if wr < POLICY_WR_FLOOR * 100 else "#ff8f00" if wr < R3_WIN_RATE * 100 else "#388e3c"
        for wr in wrs
    ]
    ax1.bar(tickers, wrs, color=colors_wr, edgecolor="black", linewidth=0.6)
    ax1.axhline(R3_WIN_RATE * 100, color="#ff8f00", linewidth=2, linestyle="--")
    ax1.axhline(POLICY_WR_FLOOR * 100, color="#d32f2f", linewidth=1.5, linestyle=":")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate by Ticker")

    colors_pf = [
        "#d32f2f" if pf < BREAK_EVEN_PROFIT_FACTOR else "#ff8f00" if pf < R3_MIN_PROFIT_FACTOR else "#388e3c"
        for pf in pfs
    ]
    ax2.bar(tickers, pfs, color=colors_pf, edgecolor="black", linewidth=0.6)
    ax2.axhline(R3_MIN_PROFIT_FACTOR, color="#ff8f00", linewidth=2, linestyle="--")
    ax2.axhline(BREAK_EVEN_PROFIT_FACTOR, color="#d32f2f", linewidth=1.5, linestyle=":")
    ax2.set_ylabel("Profit Factor (capped at 5)")
    ax2.set_title("Profit Factor by Ticker")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def chart_global_wr_over_time(time_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping WR-over-time chart")
        return
    if not time_rows:
        return
    wrs = [max(0.0, min(100.0, _finite_float(row["cumulative_wr"]) * 100)) for row in time_rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(len(wrs)), wrs, color="#1565c0", linewidth=2)
    ax.axhline(R3_WIN_RATE * 100, color="#ff8f00", linewidth=2, linestyle="--")
    ax.axhline(POLICY_WR_FLOOR * 100, color="#d32f2f", linewidth=1.2, linestyle=":")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Cumulative Win Rate (%)")
    ax.set_xlabel("Trade index")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def chart_lift_global_vs_recent(l1_metrics: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping lift chart")
        return

    lift_global = _finite_float(l1_metrics.get("lift_fraction_global"))
    lift_recent = _finite_float(l1_metrics.get("lift_fraction_recent"))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Global lift", "Recent lift"],
        [lift_global * 100, lift_recent * 100],
        color=["#e57373", "#81c784"],
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax.axhline(MIN_LIFT_FRACTION * 100, color="#ff8f00", linewidth=2, linestyle="--")
    ci_low = l1_metrics.get("lift_ci_low")
    ci_high = l1_metrics.get("lift_ci_high")
    ci_low_pct = None
    ci_high_pct = None
    if ci_low is not None and ci_high is not None and not l1_metrics.get("lift_ci_insufficient_data", True):
        low = _finite_float(ci_low) * 100
        high = _finite_float(ci_high) * 100
        if high < low:
            low, high = high, low
        ci_low_pct = low
        ci_high_pct = high
        mid = lift_global * 100
        ax.errorbar(
            [0],
            [mid],
            yerr=[[max(0.0, mid - low)], [max(0.0, high - mid)]],
            fmt="none",
            capsize=8,
            color="#333",
        )
    ymin, ymax = _lift_axis_bounds(
        lift_global_pct=lift_global * 100,
        lift_recent_pct=lift_recent * 100,
        threshold_pct=MIN_LIFT_FRACTION * 100,
        ci_low_pct=ci_low_pct,
        ci_high_pct=ci_high_pct,
    )
    ax.set_ylim(ymin, ymax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}%", ha="center")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def chart_ticker_eligibility_grid(eligibility: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping eligibility grid chart")
        return
    tickers_data = eligibility.get("tickers", {}) if isinstance(eligibility, dict) else {}
    if not tickers_data:
        return

    status_colors = {"HEALTHY": "#388e3c", "WEAK": "#e53935", "LAB_ONLY": "#9e9e9e"}
    status_order = {"HEALTHY": 0, "WEAK": 1, "LAB_ONLY": 2}
    rows = sorted(
        tickers_data.items(),
        key=lambda kv: (status_order.get(kv[1].get("status"), 9), -_finite_float(kv[1].get("win_rate"))),
    )
    tickers = [ticker for ticker, _ in rows]
    wrs = [_finite_float(info.get("win_rate")) * 100 for _, info in rows]
    colors = [status_colors.get(info.get("status"), "#9e9e9e") for _, info in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.5 + 2)))
    y_pos = list(range(len(rows)))
    ax.barh(y_pos, wrs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(R3_WIN_RATE * 100, color="#ff8f00", linewidth=2, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Win Rate (%)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def chart_context_quality_heatmap(context: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping context chart")
        return
    regime_q = context.get("regime_quality", {}) if isinstance(context, dict) else {}
    conf_q = context.get("confidence_bin_quality", {}) if isinstance(context, dict) else {}
    if not regime_q and not conf_q:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    if regime_q:
        regimes = list(regime_q.keys())
        wrs = [max(0.0, min(100.0, _finite_float(regime_q[r]["win_rate"]) * 100)) for r in regimes]
        ax1.bar(range(len(regimes)), wrs, color="#1565c0")
        ax1.axhline(R3_WIN_RATE * 100, color="#ff8f00", linewidth=2, linestyle="--")
        ax1.set_xticks(range(len(regimes)))
        ax1.set_xticklabels(regimes, rotation=30, ha="right", fontsize=7)
        ax1.set_ylim(0, 105)
    else:
        ax1.text(0.5, 0.5, "No regime data", ha="center", va="center", transform=ax1.transAxes)
    if conf_q:
        bins = list(conf_q.keys())
        wrs = [max(0.0, min(100.0, _finite_float(conf_q[b]["win_rate"]) * 100)) for b in bins]
        ax2.bar(range(len(bins)), wrs, color="#25c2a0")
        ax2.axhline(R3_WIN_RATE * 100, color="#ff8f00", linewidth=2, linestyle="--")
        ax2.set_xticks(range(len(bins)))
        ax2.set_xticklabels(bins, rotation=15, ha="right", fontsize=8)
        ax2.set_ylim(0, 105)
    else:
        ax2.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax2.transAxes)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def generate_performance_artifacts(
    *,
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    eligibility_path: Path = DEFAULT_ELIGIBILITY,
    context_quality_path: Path = DEFAULT_CONTEXT_QUALITY,
    json_metrics_path: Path = DEFAULT_METRICS_PATH,
    sufficiency: dict[str, Any] | None = None,
    strict_mode: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []

    per_ticker = _load_per_ticker(db_path)
    time_rows = _load_wr_over_time(db_path)
    l1_metrics = _load_lift_metrics(audit_dir)
    eligibility = _load_json(eligibility_path, "eligibility", warnings)
    context = _load_json(context_quality_path, "context_quality", warnings)
    if sufficiency is None:
        sufficiency = run_data_sufficiency(db_path=db_path, audit_dir=audit_dir) if db_path.exists() else {
            "status": "DATA_ERROR",
            "sufficient": False,
            "recommendations": ["DB missing"],
        }
    if context.get("partial_data"):
        warnings.append("context_partial_data")
    if sufficiency.get("status") != "SUFFICIENT":
        warnings.append("sufficiency_not_green")

    chart_paths = {
        "per_ticker_wr_pf": str(out_dir / "per_ticker_wr_pf.png"),
        "global_wr_over_time": str(out_dir / "global_wr_over_time.png"),
        "lift_global_vs_recent": str(out_dir / "lift_global_vs_recent.png"),
        "ticker_eligibility_grid": str(out_dir / "ticker_eligibility_grid.png"),
        "context_quality_heatmap": str(out_dir / "context_quality_heatmap.png"),
    }

    chart_per_ticker_wr_pf(per_ticker, Path(chart_paths["per_ticker_wr_pf"]))
    chart_global_wr_over_time(time_rows, Path(chart_paths["global_wr_over_time"]))
    chart_lift_global_vs_recent(l1_metrics, Path(chart_paths["lift_global_vs_recent"]))
    chart_ticker_eligibility_grid(eligibility, Path(chart_paths["ticker_eligibility_grid"]))
    chart_context_quality_heatmap(context, Path(chart_paths["context_quality_heatmap"]))

    for name, raw_path in chart_paths.items():
        path = Path(raw_path)
        if path.exists():
            continue
        missing_code = f"chart_missing:{name}"
        warnings.append(missing_code)
        if strict_mode:
            errors.append(missing_code)

    metrics = _build_metrics_summary(
        per_ticker=per_ticker,
        l1_metrics=l1_metrics,
        eligibility=eligibility,
        context=context,
        sufficiency=sufficiency,
        warnings=warnings,
        chart_paths=chart_paths,
        db_path=db_path,
    )
    append_threshold_hash_change_warning(json_metrics_path, metrics)
    metrics["warnings"] = sorted(set(metrics.get("warnings", [])))
    metrics["errors"] = sorted(set(errors))
    if metrics["errors"]:
        metrics["status"] = "ERROR"
    else:
        metrics["status"] = "WARN" if metrics.get("warnings") else "PASS"
    json_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    json_metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    return {
        "metrics_path": str(json_metrics_path),
        "chart_paths": chart_paths,
        "metrics": metrics,
        "status": metrics["status"],
        "warnings": metrics["warnings"],
        "errors": metrics["errors"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate performance charts and JSON metrics for capital-readiness monitoring."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--context-quality", type=Path, default=DEFAULT_CONTEXT_QUALITY)
    parser.add_argument("--json-metrics", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = generate_performance_artifacts(
        db_path=args.db,
        audit_dir=args.audit_dir,
        out_dir=args.out_dir,
        eligibility_path=args.eligibility,
        context_quality_path=args.context_quality,
        json_metrics_path=(args.json_metrics or (args.out_dir / "metrics_summary.json")),
    )

    print(f"Charts written to {args.out_dir}/")
    for name, path in result["chart_paths"].items():
        status = "[OK]" if Path(path).exists() else "[SKIP]"
        print(f"  {status} {Path(path).name}")
    if Path(result["metrics_path"]).exists():
        print(f"  [OK] metrics_summary.json -> {result['metrics_path']}")
    return 1 if result.get("status") == "ERROR" else 0


if __name__ == "__main__":
    sys.exit(main())
