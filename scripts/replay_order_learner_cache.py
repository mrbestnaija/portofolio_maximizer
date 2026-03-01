#!/usr/bin/env python3
"""
replay_order_learner_cache.py
----------------------------
Run a bounded synthetic replay that performs real TimeSeriesForecaster fits over
multiple shifted windows for selected tickers.

Purpose:
- build legitimate OrderLearner evidence without lowering thresholds,
- avoid the bar-aware same-bar short-circuit in run_auto_trader,
- make snapshot restores visible instead of counting them as fresh fits.

This script never increments cache evidence directly. It only calls the normal
fit path and reports whether each model was actually fitted or restored from a
snapshot based on forecaster events.

Usage:
    python scripts/replay_order_learner_cache.py --tickers AAPL,MSFT
    python scripts/replay_order_learner_cache.py --tickers AAPL --replays 2 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.synthetic_extractor import SyntheticExtractor
from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.order_learner import OrderLearner
from integrity.sqlite_guardrails import guarded_sqlite_connect

DEFAULT_DB_PATH = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_FORECASTING_CONFIG_PATH = ROOT / "config" / "forecasting_config.yml"
ORDER_CACHE_MODELS = {"SARIMAX", "SAMOSSA", "GARCH"}
EVENT_TO_CACHE_MODEL_TYPE = {
    "SARIMAX": "SARIMAX",
    "SAMOSSA": "SAMOSSA_ARIMA",
    "GARCH": "GARCH",
}
CACHE_MODEL_TYPES = tuple(sorted(set(EVENT_TO_CACHE_MODEL_TYPE.values())))


@dataclass(frozen=True)
class ReplayWindow:
    replay_index: int
    start_date: str
    end_date: str
    train_points: int


def _parse_tickers(raw: str) -> list[str]:
    tickers = [token.strip().upper() for token in str(raw or "").split(",") if token.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")
    return list(dict.fromkeys(tickers))


def _load_forecasting_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict) and "forecasting" in payload:
        nested = payload["forecasting"]
        return nested if isinstance(nested, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _cfg_section(root_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    section = root_cfg.get(key, {})
    return section if isinstance(section, dict) else {}


def _require_order_learning_enabled(root_cfg: dict[str, Any]) -> dict[str, Any]:
    order_learning_cfg = _cfg_section(root_cfg, "order_learning")
    if not bool(order_learning_cfg.get("enabled", False)):
        raise ValueError("Order learning is disabled in forecasting config; replay would not build cache evidence.")
    return order_learning_cfg


def _build_forecaster_config(
    root_cfg: dict[str, Any],
    *,
    forecast_horizon: int,
    db_path: Path,
) -> TimeSeriesForecasterConfig:
    ensemble_cfg = _cfg_section(root_cfg, "ensemble")
    regime_cfg = _cfg_section(root_cfg, "regime_detection")
    order_learning_cfg = _require_order_learning_enabled(root_cfg)
    monte_carlo_cfg = _cfg_section(root_cfg, "monte_carlo")
    sarimax_cfg = _cfg_section(root_cfg, "sarimax")
    garch_cfg = _cfg_section(root_cfg, "garch")
    samossa_cfg = _cfg_section(root_cfg, "samossa")
    mssa_cfg = _cfg_section(root_cfg, "mssa_rl")

    samossa_kwargs = {k: v for k, v in samossa_cfg.items() if k != "enabled"}
    if bool(samossa_cfg.get("enabled", True)):
        samossa_kwargs["forecast_horizon"] = int(forecast_horizon)

    mssa_kwargs = {k: v for k, v in mssa_cfg.items() if k != "enabled"}
    if bool(mssa_cfg.get("enabled", True)):
        mssa_kwargs["forecast_horizon"] = int(forecast_horizon)

    return TimeSeriesForecasterConfig(
        forecast_horizon=int(forecast_horizon),
        sarimax_enabled=bool(sarimax_cfg.get("enabled", False)),
        garch_enabled=bool(garch_cfg.get("enabled", True)),
        samossa_enabled=bool(samossa_cfg.get("enabled", True)),
        mssa_rl_enabled=bool(mssa_cfg.get("enabled", True)),
        ensemble_enabled=bool(ensemble_cfg.get("enabled", True)),
        sarimax_kwargs={k: v for k, v in sarimax_cfg.items() if k != "enabled"},
        garch_kwargs={k: v for k, v in garch_cfg.items() if k != "enabled"},
        samossa_kwargs=samossa_kwargs,
        mssa_rl_kwargs=mssa_kwargs,
        ensemble_kwargs={k: v for k, v in ensemble_cfg.items() if k != "enabled"},
        regime_detection_enabled=bool(regime_cfg.get("enabled", False)),
        regime_detection_kwargs={k: v for k, v in regime_cfg.items() if k != "enabled"},
        order_learning_config=order_learning_cfg,
        order_learning_db_path=str(db_path),
        monte_carlo_config=monte_carlo_cfg,
    )


def _build_order_learner(db_path: Path, root_cfg: dict[str, Any]) -> OrderLearner:
    return OrderLearner(
        db_path=str(db_path),
        config=_cfg_section(root_cfg, "order_learning"),
    )


def _read_coverage_stats(db_path: Path, root_cfg: dict[str, Any]) -> dict[str, int]:
    learner = _build_order_learner(db_path, root_cfg)
    stats = learner.coverage_stats()
    return {
        "total_entries": int(stats.get("total_entries", 0) or 0),
        "qualified_entries": int(stats.get("qualified_entries", 0) or 0),
    }


def _connect_ro(db_path: Path):
    conn = guarded_sqlite_connect(str(db_path), timeout=5.0, enable_guardrails=False)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _read_cache_snapshot(db_path: Path, tickers: list[str]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not tickers:
        return {}

    placeholders = ",".join("?" for _ in tickers)
    model_placeholders = ",".join("?" for _ in CACHE_MODEL_TYPES)
    sql = f"""
        SELECT ticker, model_type, regime, order_params, n_fits, best_aic, last_used
        FROM model_order_stats
        WHERE ticker IN ({placeholders})
          AND model_type IN ({model_placeholders})
        ORDER BY ticker, model_type, regime, order_params
    """
    params = tuple(tickers) + CACHE_MODEL_TYPES
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    snapshot: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row[0] or ""),
            str(row[1] or ""),
            str(row[2] or ""),
            str(row[3] or ""),
        )
        snapshot[key] = {
            "ticker": key[0],
            "model_type": key[1],
            "regime": key[2],
            "order_params": key[3],
            "n_fits": int(row[4] or 0),
            "best_aic": row[5],
            "last_used": str(row[6] or ""),
        }
    return snapshot


def _is_qualified_row(row: dict[str, Any], min_fits: int) -> bool:
    return int(row.get("n_fits", 0) or 0) >= int(min_fits) and row.get("best_aic") is not None


def _summarize_cache_evidence(
    before: dict[tuple[str, str, str, str], dict[str, Any]],
    after: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    min_fits: int,
    actual_fit_events_by_cache_model: dict[str, int],
) -> dict[str, Any]:
    model_types = sorted(
        set(CACHE_MODEL_TYPES)
        | {row["model_type"] for row in before.values()}
        | {row["model_type"] for row in after.values()}
    )
    all_keys = set(before) | set(after)

    before_count = len(before)
    after_count = len(after)
    before_qualified = sum(1 for row in before.values() if _is_qualified_row(row, min_fits))
    after_qualified = sum(1 for row in after.values() if _is_qualified_row(row, min_fits))

    total_n_fits_delta = 0
    new_rows = 0
    touched_rows = 0
    by_model_type: dict[str, dict[str, int]] = {}

    for model_type in model_types:
        before_rows = [row for row in before.values() if row["model_type"] == model_type]
        after_rows = [row for row in after.values() if row["model_type"] == model_type]
        model_n_fits_delta = 0
        model_new_rows = 0
        model_touched_rows = 0

        for key in all_keys:
            after_row = after.get(key)
            if not after_row or after_row["model_type"] != model_type:
                continue
            before_row = before.get(key)
            if before_row is None:
                delta = int(after_row.get("n_fits", 0) or 0)
                if delta > 0:
                    model_new_rows += 1
            else:
                delta = int(after_row.get("n_fits", 0) or 0) - int(before_row.get("n_fits", 0) or 0)
                if delta > 0:
                    model_touched_rows += 1
            if delta > 0:
                model_n_fits_delta += delta

        actual_fit_events = int(actual_fit_events_by_cache_model.get(model_type, 0) or 0)
        actual_without_cache_write = max(0, actual_fit_events - model_n_fits_delta)
        by_model_type[model_type] = {
            "rows_before": len(before_rows),
            "rows_after": len(after_rows),
            "row_count_delta": len(after_rows) - len(before_rows),
            "qualified_before": sum(1 for row in before_rows if _is_qualified_row(row, min_fits)),
            "qualified_after": sum(1 for row in after_rows if _is_qualified_row(row, min_fits)),
            "qualified_delta": sum(1 for row in after_rows if _is_qualified_row(row, min_fits))
            - sum(1 for row in before_rows if _is_qualified_row(row, min_fits)),
            "n_fits_delta": model_n_fits_delta,
            "new_rows": model_new_rows,
            "touched_rows": model_touched_rows,
            "actual_fit_events": actual_fit_events,
            "actual_without_cache_write": actual_without_cache_write,
        }
        total_n_fits_delta += model_n_fits_delta
        new_rows += model_new_rows
        touched_rows += model_touched_rows

    return {
        "rows_before": before_count,
        "rows_after": after_count,
        "row_count_delta": after_count - before_count,
        "qualified_before": before_qualified,
        "qualified_after": after_qualified,
        "qualified_row_delta": after_qualified - before_qualified,
        "n_fits_delta": total_n_fits_delta,
        "new_rows": new_rows,
        "touched_rows": touched_rows,
        "by_model_type": by_model_type,
    }


def _extract_close_series(frame: pd.DataFrame, ticker: str) -> pd.Series:
    subset = frame
    if "ticker" in frame.columns:
        subset = frame.loc[frame["ticker"].astype(str).str.upper() == ticker]
    if subset.empty:
        raise ValueError(f"No rows returned for ticker {ticker}")
    if "Close" not in subset.columns:
        raise ValueError("Synthetic replay requires a Close column.")
    series = subset["Close"].astype(float).dropna().sort_index()
    if series.empty:
        raise ValueError(f"Close series for {ticker} is empty after cleaning.")
    series = series.copy()
    series.name = ticker
    return series


def _iter_train_lengths(total_points: int, min_train_size: int, train_step: int, max_windows: int) -> list[int]:
    if total_points < min_train_size:
        raise ValueError(
            f"Insufficient data for replay (need >= {min_train_size}, received {total_points})."
        )
    step = max(1, int(train_step))
    windows = max(1, int(max_windows))
    start = max(min_train_size, total_points - (step * (windows - 1)))
    lengths = list(range(start, total_points + 1, step))
    if not lengths:
        lengths = [total_points]
    if lengths[-1] != total_points:
        lengths.append(total_points)
    if len(lengths) > windows:
        lengths = lengths[-windows:]
    return sorted(dict.fromkeys(int(value) for value in lengths))


def _build_replay_windows(
    *,
    total_points: int,
    lookback_days: int,
    replays: int,
    end_offset_step_days: int,
    min_train_size: int,
    train_step: int,
    max_train_windows: int,
    as_of_date: date | None = None,
) -> list[ReplayWindow]:
    anchor = as_of_date or date.today()
    train_lengths = _iter_train_lengths(total_points, min_train_size, train_step, max_train_windows)
    windows: list[ReplayWindow] = []
    for replay_index in range(int(replays)):
        offset_days = replay_index * max(0, int(end_offset_step_days))
        end_dt = anchor - timedelta(days=offset_days)
        start_dt = end_dt - timedelta(days=max(1, int(lookback_days)))
        for train_points in train_lengths:
            windows.append(
                ReplayWindow(
                    replay_index=replay_index + 1,
                    start_date=start_dt.isoformat(),
                    end_date=end_dt.isoformat(),
                    train_points=int(train_points),
                )
            )
    return windows


def _summarize_fit_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, int]] = {}
    actual_fits_by_cache_model: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    for event in events:
        phase = str(event.get("phase") or "")
        model = str(event.get("model") or "").upper()
        if phase == "fit_complete" and model in ORDER_CACHE_MODELS:
            bucket = by_model.setdefault(model, {"actual_fits": 0, "restored_fits": 0})
            if bool(event.get("restored", False)):
                bucket["restored_fits"] += 1
            else:
                bucket["actual_fits"] += 1
                mapped_model_type = EVENT_TO_CACHE_MODEL_TYPE.get(model)
                if mapped_model_type:
                    actual_fits_by_cache_model[mapped_model_type] = (
                        int(actual_fits_by_cache_model.get(mapped_model_type, 0) or 0) + 1
                    )
        elif phase.endswith("_failed") and model in ORDER_CACHE_MODELS:
            failures.append({"model": model, "phase": phase, "error": str(event.get("error") or "")})
    actual_fit_count = sum(v["actual_fits"] for v in by_model.values())
    restored_fit_count = sum(v["restored_fits"] for v in by_model.values())
    return {
        "actual_fit_count": int(actual_fit_count),
        "restored_fit_count": int(restored_fit_count),
        "by_model": by_model,
        "actual_fits_by_cache_model": actual_fits_by_cache_model,
        "failures": failures,
    }


def replay_order_learner_cache(
    *,
    tickers: list[str],
    db_path: Path,
    forecasting_config_path: Path,
    lookback_days: int = 365,
    replays: int = 3,
    end_offset_step_days: int = 21,
    min_train_size: int = 180,
    train_step: int = 30,
    max_train_windows: int = 3,
    forecast_horizon: int = 30,
) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if replays <= 0:
        raise ValueError("replays must be >= 1")
    if max_train_windows <= 0:
        raise ValueError("max_train_windows must be >= 1")

    root_cfg = _load_forecasting_config(forecasting_config_path)
    _require_order_learning_enabled(root_cfg)
    learner = _build_order_learner(db_path, root_cfg)
    min_fits = int(getattr(learner, "_min_fits", 3) or 3)
    before = _read_coverage_stats(db_path, root_cfg)
    before_snapshot = _read_cache_snapshot(db_path, tickers)
    extractor = SyntheticExtractor()

    total_actual = 0
    total_restored = 0
    actual_fit_events_by_cache_model: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    replay_details: list[dict[str, Any]] = []

    for replay_index in range(int(replays)):
        offset_days = replay_index * max(0, int(end_offset_step_days))
        end_dt = date.today() - timedelta(days=offset_days)
        start_dt = end_dt - timedelta(days=max(1, int(lookback_days)))
        market_data = extractor.extract_ohlcv(
            tickers=tickers,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
        )

        for ticker in tickers:
            close_series = _extract_close_series(market_data, ticker)
            replay_windows = _build_replay_windows(
                total_points=len(close_series),
                lookback_days=lookback_days,
                replays=1,
                end_offset_step_days=end_offset_step_days,
                min_train_size=min_train_size,
                train_step=train_step,
                max_train_windows=max_train_windows,
                as_of_date=end_dt,
            )
            for window in replay_windows:
                train = close_series.iloc[: window.train_points].copy()
                returns = train.pct_change().dropna()
                if not returns.empty:
                    returns = returns.copy()
                    returns.name = ticker

                config = _build_forecaster_config(
                    root_cfg,
                    forecast_horizon=forecast_horizon,
                    db_path=db_path,
                )
                forecaster = TimeSeriesForecaster(config=config)
                forecaster.fit(
                    price_series=train,
                    returns_series=returns,
                    ticker=ticker,
                )
                event_summary = _summarize_fit_events(
                    list(forecaster.get_component_summaries().get("events", []))
                )
                total_actual += event_summary["actual_fit_count"]
                total_restored += event_summary["restored_fit_count"]
                for model_type, count in event_summary["actual_fits_by_cache_model"].items():
                    actual_fit_events_by_cache_model[model_type] = (
                        int(actual_fit_events_by_cache_model.get(model_type, 0) or 0) + int(count or 0)
                    )
                failures.extend(event_summary["failures"])
                replay_details.append(
                    {
                        "ticker": ticker,
                        "replay_index": int(window.replay_index + replay_index),
                        "start_date": window.start_date,
                        "end_date": window.end_date,
                        "train_points": int(window.train_points),
                        "actual_fit_count": int(event_summary["actual_fit_count"]),
                        "restored_fit_count": int(event_summary["restored_fit_count"]),
                        "models": event_summary["by_model"],
                    }
                )

    after = _read_coverage_stats(db_path, root_cfg)
    after_snapshot = _read_cache_snapshot(db_path, tickers)
    cache_evidence = _summarize_cache_evidence(
        before_snapshot,
        after_snapshot,
        min_fits=min_fits,
        actual_fit_events_by_cache_model=actual_fit_events_by_cache_model,
    )
    qualified_gain = int(after["qualified_entries"] - before["qualified_entries"])
    total_gain = int(after["total_entries"] - before["total_entries"])
    warnings: list[str] = []
    for model_type, summary in cache_evidence["by_model_type"].items():
        if int(summary.get("actual_without_cache_write", 0) or 0) > 0:
            warnings.append(
                f"{model_type}: {summary['actual_without_cache_write']} actual fit(s) produced no cache write"
            )

    status = "PASS"
    if failures:
        status = "WARN"
    if total_actual <= 0:
        status = "WARN"
    if total_actual > 0 and int(cache_evidence.get("n_fits_delta", 0) or 0) <= 0:
        status = "WARN"

    return {
        "status": status,
        "db_path": str(db_path),
        "forecasting_config_path": str(forecasting_config_path),
        "tickers": tickers,
        "replays": int(replays),
        "lookback_days": int(lookback_days),
        "end_offset_step_days": int(end_offset_step_days),
        "min_train_size": int(min_train_size),
        "train_step": int(train_step),
        "max_train_windows": int(max_train_windows),
        "forecast_horizon": int(forecast_horizon),
        "before_coverage": before,
        "after_coverage": after,
        "coverage_delta": {
            "total_entries": total_gain,
            "qualified_entries": qualified_gain,
        },
        "cache_evidence": cache_evidence,
        "actual_fit_count": int(total_actual),
        "restored_fit_count": int(total_restored),
        "failure_count": int(len(failures)),
        "failures": failures,
        "warnings": warnings,
        "windows": replay_details,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay bounded synthetic forecast fits to build OrderLearner cache evidence."
    )
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list, e.g. AAPL,MSFT")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to portfolio_maximizer.db")
    parser.add_argument(
        "--forecasting-config",
        default=str(DEFAULT_FORECASTING_CONFIG_PATH),
        help="Path to forecasting_config.yml",
    )
    parser.add_argument("--lookback-days", type=int, default=365, help="Synthetic lookback window per replay.")
    parser.add_argument("--replays", type=int, default=3, help="Number of shifted synthetic replays to run.")
    parser.add_argument(
        "--end-offset-step-days",
        type=int,
        default=21,
        help="How far to shift the replay end date between replays.",
    )
    parser.add_argument("--min-train-size", type=int, default=180, help="Minimum train points per replay window.")
    parser.add_argument("--train-step", type=int, default=30, help="Point spacing between train windows.")
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=3,
        help="Maximum train windows per ticker per replay.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=30,
        help="Forecast horizon used when constructing the forecaster config.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = replay_order_learner_cache(
            tickers=_parse_tickers(args.tickers),
            db_path=Path(args.db),
            forecasting_config_path=Path(args.forecasting_config),
            lookback_days=int(args.lookback_days),
            replays=int(args.replays),
            end_offset_step_days=int(args.end_offset_step_days),
            min_train_size=int(args.min_train_size),
            train_step=int(args.train_step),
            max_train_windows=int(args.max_train_windows),
            forecast_horizon=int(args.forecast_horizon),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"[replay_order_learner_cache] status=ERROR")
            print(f"[replay_order_learner_cache] error={exc}")
        return 1

    if args.json:
        print(json.dumps({"ok": True, **result}, default=str))
        return 0

    print(f"[replay_order_learner_cache] status={result['status']}")
    print(f"[replay_order_learner_cache] DB: {result['db_path']}")
    print(f"  tickers: {', '.join(result['tickers'])}")
    print(f"  replays: {result['replays']}")
    print(f"  actual_fit_count: {result['actual_fit_count']}")
    print(f"  restored_fit_count: {result['restored_fit_count']}")
    print(f"  failure_count: {result['failure_count']}")
    print(f"  cache_n_fits_delta: {result['cache_evidence']['n_fits_delta']}")
    print(
        "  coverage_delta: total_entries={total_entries} qualified_entries={qualified_entries}".format(
            **result["coverage_delta"]
        )
    )
    print(f"  before_coverage: {json.dumps(result['before_coverage'], sort_keys=True)}")
    print(f"  after_coverage: {json.dumps(result['after_coverage'], sort_keys=True)}")
    if result["warnings"]:
        print("  warnings:")
        for item in result["warnings"][:10]:
            print(f"    {item}")
    if result["failures"]:
        print("  failures:")
        for item in result["failures"][:10]:
            print(
                "    model={model} phase={phase} error={error}".format(
                    **item
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
