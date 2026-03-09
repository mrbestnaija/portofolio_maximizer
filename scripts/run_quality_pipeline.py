"""
Run the read-only quality pipeline end-to-end and emit one JSON status contract.
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
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_context_quality import DEFAULT_OUTPUT as DEFAULT_CONTEXT_OUT
from scripts.compute_context_quality import compute_context_quality
from scripts.compute_ticker_eligibility import DEFAULT_OUTPUT as DEFAULT_ELIGIBILITY_OUT
from scripts.compute_ticker_eligibility import compute_eligibility
from scripts.apply_ticker_eligibility_gates import (
    DEFAULT_OUTPUT as DEFAULT_ELIGIBILITY_GATES_OUT,
    apply_eligibility_gates,
)
from scripts.data_sufficiency_monitor import DEFAULT_AUDIT_DIR, DEFAULT_DB, run_data_sufficiency
from scripts.generate_performance_charts import (
    DEFAULT_METRICS_PATH,
    DEFAULT_OUT_DIR,
    generate_performance_artifacts,
)
from scripts.quality_pipeline_common import (
    RESIDUAL_EXPERIMENT_CANONICAL_FIELDS,
    RESIDUAL_EXPERIMENT_CONTRACT_PATH,
    RESIDUAL_EXPERIMENT_CONTRACT_VERSION,
    append_threshold_hash_change_warning,
    connect_ro,
    extract_residual_experiment_diagnostics,
    first_existing_columns,
    load_forecast_audit_windows,
    residual_experiment_metrics_present,
    residual_experiment_window_active,
    sqlite_master_names,
    table_columns,
)
from scripts.robustness_thresholds import threshold_map

log = logging.getLogger(__name__)
PIPELINE_VERSION = "2026.03.08.1"
DEFAULT_RESIDUAL_EXPERIMENT_OUT = DEFAULT_OUT_DIR / "residual_experiment_summary.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _step_status_from_flags(*, warnings: list[str], error: str | None = None) -> str:
    if error:
        return "ERROR"
    if warnings:
        return "WARN"
    return "PASS"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _finite_float(value: Any) -> float | None:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _rmse(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) != len(predicted) or len(actual) == 0:
        return None
    err_sq = [(predicted[i] - actual[i]) ** 2 for i in range(len(actual))]
    return _finite_float(math.sqrt(sum(err_sq) / len(err_sq)))


def _directional_accuracy(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) != len(predicted) or len(actual) < 2:
        return None
    actual_diff = [actual[i] - actual[i - 1] for i in range(1, len(actual))]
    pred_diff = [predicted[i] - predicted[i - 1] for i in range(1, len(predicted))]
    def _sign(v: float) -> int:
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    correct = [_sign(a) == _sign(p) for a, p in zip(actual_diff, pred_diff)]
    if not correct:
        return None
    return _finite_float(sum(1.0 for c in correct if c) / len(correct))


def _quote_sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _load_realized_close_window(
    *,
    db_path: Path,
    ticker: str,
    dataset_end: str,
    max_points: int,
) -> list[float]:
    if max_points <= 0:
        return []
    try:
        conn = connect_ro(db_path)
    except Exception:
        return []
    try:
        if "ohlcv_data" not in sqlite_master_names(conn, "table"):
            return []
        cols = table_columns(conn, "ohlcv_data")
        date_cols = first_existing_columns(
            cols,
            ("date", "Date", "datetime", "timestamp", "trade_date"),
        )
        close_cols = first_existing_columns(
            cols,
            ("close", "Close", "adj_close", "Adj_Close", "adjclose"),
        )
        if not date_cols or not close_cols:
            return []
        date_col = date_cols[0]
        close_col = close_cols[0]
        sql = (
            "SELECT "
            f"CAST({_quote_sql_identifier(date_col)} AS TEXT) AS d, "
            f"CAST({_quote_sql_identifier(close_col)} AS REAL) AS c "
            "FROM ohlcv_data "
            f"WHERE ticker = ? AND {_quote_sql_identifier(date_col)} > ? "
            f"ORDER BY {_quote_sql_identifier(date_col)} ASC"
        )
        rows = conn.execute(sql, (ticker, dataset_end)).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    closes: list[float] = []
    seen_dates: set[str] = set()
    for row in rows:
        d_raw = row["d"] if hasattr(row, "__getitem__") else None
        c_raw = row["c"] if hasattr(row, "__getitem__") else None
        if d_raw is None:
            continue
        day = str(d_raw)[:10]
        if day in seen_dates:
            continue
        value = _finite_float(c_raw)
        if value is None:
            continue
        seen_dates.add(day)
        closes.append(value)
        if len(closes) >= max_points:
            break
    return closes


def _populate_realized_residual_metrics_from_db(
    *,
    fields: dict[str, Any],
    payload: dict[str, Any],
    db_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Backfill residual realized metrics from OHLCV when artifact scalars are absent."""
    out = dict(fields)
    anchor = out.get("y_hat_anchor")
    residual = out.get("y_hat_residual_ensemble")
    if not isinstance(anchor, list) or not isinstance(residual, list):
        return out, False
    if len(anchor) < 2 or len(residual) < 2:
        return out, False

    ds = payload.get("dataset")
    if not isinstance(ds, dict):
        return out, False
    ticker = ds.get("ticker")
    dataset_end = ds.get("end")
    if not isinstance(ticker, str) or not ticker.strip():
        return out, False
    if not isinstance(dataset_end, str) or not dataset_end.strip():
        return out, False

    max_points = min(len(anchor), len(residual))
    realized = _load_realized_close_window(
        db_path=db_path,
        ticker=ticker.strip(),
        dataset_end=dataset_end.strip(),
        max_points=max_points,
    )
    n = min(len(realized), len(anchor), len(residual))
    if n < 2:
        return out, False

    actual_vals = [float(v) for v in realized[:n]]
    anchor_vals = [float(v) for v in anchor[:n]]
    residual_vals = [float(v) for v in residual[:n]]

    rmse_anchor = _rmse(actual_vals, anchor_vals)
    rmse_residual = _rmse(actual_vals, residual_vals)
    da_anchor = _directional_accuracy(actual_vals, anchor_vals)
    da_residual = _directional_accuracy(actual_vals, residual_vals)

    # Populate only missing fields; never overwrite values already emitted by Agent A.
    if out.get("rmse_anchor") is None and rmse_anchor is not None:
        out["rmse_anchor"] = rmse_anchor
    if out.get("rmse_residual_ensemble") is None and rmse_residual is not None:
        out["rmse_residual_ensemble"] = rmse_residual
    if out.get("da_anchor") is None and da_anchor is not None:
        out["da_anchor"] = da_anchor
    if out.get("da_residual_ensemble") is None and da_residual is not None:
        out["da_residual_ensemble"] = da_residual

    if out.get("rmse_ratio") is None:
        r_a = _finite_float(out.get("rmse_anchor"))
        r_r = _finite_float(out.get("rmse_residual_ensemble"))
        if r_a is not None and r_r is not None and r_a > 0:
            out["rmse_ratio"] = _finite_float(r_r / r_a)

    return out, True


def _probe_residual_artifact_contract_compatibility() -> dict[str, Any]:
    """Call Agent A builder and verify extractor compatibility against canonical fields."""
    try:
        import pandas as pd
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"residual_experiment_contract_probe_import_failed:{exc}"],
            "warnings": [],
            "expected_not_fitted": None,
        }

    try:
        forecaster = TimeSeriesForecaster(
            TimeSeriesForecasterConfig(
                sarimax_enabled=False,
                garch_enabled=False,
                samossa_enabled=False,
                mssa_rl_enabled=False,
                ensemble_enabled=False,
                forecast_horizon=3,
                residual_experiment_enabled=True,
            )
        )
        anchor_idx = pd.date_range("2026-01-05", periods=3, freq="B")
        anchor_series = pd.Series([100.0, 101.0, 102.0], index=anchor_idx, dtype=float)
        artifact = forecaster._build_residual_experiment_artifact(  # noqa: SLF001
            {
                "mssa_rl_forecast": {
                    "forecast": anchor_series,
                    "lower_ci": None,
                    "upper_ci": None,
                }
            }
        )
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"residual_experiment_contract_probe_call_failed:{exc}"],
            "warnings": [],
            "expected_not_fitted": None,
        }

    errors: list[str] = []
    warnings: list[str] = []
    artifact_keys = set(artifact.keys()) if isinstance(artifact, dict) else set()
    missing = sorted(set(RESIDUAL_EXPERIMENT_CANONICAL_FIELDS) - artifact_keys)
    if missing:
        errors.append("residual_experiment_contract_missing_fields:" + ",".join(missing))

    diagnostics = extract_residual_experiment_diagnostics({"artifacts": {"residual_experiment": artifact}})
    if bool(diagnostics.get("malformed")):
        diag_codes = diagnostics.get("errors", [])
        errors.append("residual_experiment_contract_extractor_malformed:" + ",".join(map(str, diag_codes)))

    expected_not_fitted = bool(
        artifact.get("residual_status") == "inactive"
        and artifact.get("residual_active") is False
    )
    if not expected_not_fitted:
        warnings.append("residual_experiment_contract_probe_phase_state_changed")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "expected_not_fitted": expected_not_fitted,
    }


def _collect_residual_experiment_summary(audit_dir: Path, db_path: Path) -> dict[str, Any]:
    audit_windows, audit_stats = load_forecast_audit_windows(Path(audit_dir), dedupe=True)
    warnings: list[str] = []
    errors: list[str] = []
    windows: list[dict[str, Any]] = []
    malformed_windows = 0
    not_fitted_windows = 0
    non_active_signal_windows = 0
    active_signal_windows = 0
    active_without_metrics_windows = 0
    realized_metrics_backfilled = 0
    parse_error_count = int(audit_stats.get("parse_error_count", 0))
    parse_error_samples = list(audit_stats.get("parse_error_samples", []))
    contract_probe = _probe_residual_artifact_contract_compatibility()
    if not contract_probe.get("ok", False):
        errors.extend(list(contract_probe.get("errors", [])))
    warnings.extend(list(contract_probe.get("warnings", [])))

    def _has_realized_residual_scalars(row: dict[str, Any]) -> bool:
        return any(
            isinstance(row.get(key), (int, float))
            for key in ("rmse_residual_ensemble", "da_residual_ensemble", "rmse_ratio")
        )

    for payload in audit_windows:
        diagnostics = extract_residual_experiment_diagnostics(payload)
        fields = diagnostics.get("fields", {})
        if bool(diagnostics.get("malformed")):
            malformed_windows += 1
            for code in diagnostics.get("errors", []):
                errors.append(f"residual_experiment_payload_malformed:{code}")
            continue
        residual_status = fields.get("residual_status")
        residual_active = fields.get("residual_active")
        if residual_status == "inactive" and residual_active is False:
            not_fitted_windows += 1
            continue
        if not residual_experiment_window_active(fields):
            if diagnostics.get("has_experiment_signal"):
                non_active_signal_windows += 1
            continue
        active_signal_windows += 1
        has_structural_forecasts = (
            isinstance(fields.get("y_hat_anchor"), list)
            and isinstance(fields.get("y_hat_residual_ensemble"), list)
        )
        has_realized_scalars = _has_realized_residual_scalars(fields)
        if not has_realized_scalars:
            fields, backfilled = _populate_realized_residual_metrics_from_db(
                fields=fields,
                payload=payload,
                db_path=db_path,
            )
            if backfilled:
                realized_metrics_backfilled += 1
        has_phase3_metrics = residual_experiment_metrics_present(fields)
        if not has_phase3_metrics and not has_structural_forecasts:
            active_without_metrics_windows += 1
            continue
        windows.append({"window_id": str(payload.get("_path", "")), **fields})

    if parse_error_count:
        warnings.append(f"residual_experiment_parse_errors:{parse_error_count}")
    if parse_error_samples:
        warnings.append("residual_experiment_parse_error_samples:" + ",".join(parse_error_samples))

    if malformed_windows:
        warnings.append(f"residual_experiment_malformed_windows:{malformed_windows}")
        errors.append("residual_experiment_payload_malformed")

    if not_fitted_windows > 0:
        warnings.append(f"residual_experiment_not_fitted_windows:{not_fitted_windows}")
        warnings.append("residual_experiment_not_fitted")
    if non_active_signal_windows > 0:
        warnings.append(f"residual_experiment_non_active_signal_windows:{non_active_signal_windows}")
    if active_without_metrics_windows > 0:
        warnings.append(f"residual_experiment_active_without_metrics_windows:{active_without_metrics_windows}")
    if (
        not_fitted_windows == 0
        and active_without_metrics_windows == 0
        and not windows
        and not malformed_windows
    ):
        warnings.append("residual_experiment_not_available")

    realized_metric_windows = [row for row in windows if _has_realized_residual_scalars(row)]
    structural_only_windows = [row for row in windows if not _has_realized_residual_scalars(row)]
    missing_realized_metric_windows = max(0, active_signal_windows - len(realized_metric_windows))
    if windows and not realized_metric_windows:
        warnings.append("residual_experiment_realized_metrics_unavailable")
    if missing_realized_metric_windows > 0:
        warnings.append(
            f"residual_experiment_missing_realized_metrics_windows:{missing_realized_metric_windows}"
        )
    def _mean(key: str, source_windows: list[dict[str, Any]]) -> float | None:
        values = [float(row[key]) for row in source_windows if isinstance(row.get(key), (int, float))]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    # EXP-R5-001 Early abort guard (item 5, Agent C plan 2026-03-08).
    # Fires when >=5 consecutive Phase-3 windows all have rmse_ratio > 1.02,
    # meaning the residual correction is actively hurting — recommend redesign.
    _EARLY_ABORT_RMSE_THRESHOLD = 1.02
    _EARLY_ABORT_MIN_CONSECUTIVE = 5
    sorted_windows = sorted(windows, key=lambda w: w.get("window_id", ""))
    _max_consecutive = 0
    _streak = 0
    for _w in sorted_windows:
        _rr = _w.get("rmse_ratio")
        if isinstance(_rr, (int, float)) and _rr > _EARLY_ABORT_RMSE_THRESHOLD:
            _streak += 1
            _max_consecutive = max(_max_consecutive, _streak)
        else:
            _streak = 0
    early_abort_signal = _max_consecutive >= _EARLY_ABORT_MIN_CONSECUTIVE
    if early_abort_signal:
        warnings.append(
            f"EARLY_ABORT_SIGNAL:rmse_ratio>{_EARLY_ABORT_RMSE_THRESHOLD}"
            f"_for_{_max_consecutive}_consecutive_windows"
        )

    status = "PASS"
    reason_code = "RESIDUAL_EXPERIMENT_AVAILABLE"
    if errors:
        status = "ERROR"
        reason_code = "RESIDUAL_EXPERIMENT_PAYLOAD_MALFORMED"
    elif not_fitted_windows > 0 and not windows:
        status = "SKIP"
        reason_code = "RESIDUAL_EXPERIMENT_NOT_FITTED"
    elif not windows:
        status = "SKIP"
        reason_code = "RESIDUAL_EXPERIMENT_NOT_AVAILABLE"

    return {
        "contract_version": RESIDUAL_EXPERIMENT_CONTRACT_VERSION,
        "contract_path": RESIDUAL_EXPERIMENT_CONTRACT_PATH,
        "status": status,
        "reason_code": reason_code,
        "n_audits_scanned": int(audit_stats.get("audit_files_scanned", 0)),
        "n_audits_loaded": int(audit_stats.get("audit_files_loaded", 0)),
        "n_deduped_windows": int(audit_stats.get("deduped_windows", 0)),
        "n_malformed_windows": malformed_windows,
        "n_not_fitted_windows": not_fitted_windows,
        "n_non_active_signal_windows": non_active_signal_windows,
        "n_active_signal_windows": active_signal_windows,
        "n_active_without_metrics_windows": active_without_metrics_windows,
        "n_windows_with_residual_metrics": len(windows),
        "n_windows_with_realized_residual_metrics": len(realized_metric_windows),
        "n_windows_structural_only_metrics": len(structural_only_windows),
        "n_active_windows_missing_realized_metrics": missing_realized_metric_windows,
        "n_windows_with_db_backfilled_realized_metrics": realized_metrics_backfilled,
        "m2_review_ready": len(realized_metric_windows) >= 5,
        "rmse_anchor_mean": _mean("rmse_anchor", windows),
        "rmse_residual_ensemble_mean": _mean("rmse_residual_ensemble", realized_metric_windows),
        "rmse_ratio_mean": _mean("rmse_ratio", realized_metric_windows),
        "da_anchor_mean": _mean("da_anchor", windows),
        "da_residual_ensemble_mean": _mean("da_residual_ensemble", realized_metric_windows),
        "corr_anchor_residual_mean": _mean("corr_anchor_residual", realized_metric_windows),
        "early_abort_signal": early_abort_signal,
        "early_abort_consecutive_rmse_above_threshold": _max_consecutive,
        "windows": windows,
        "errors": _dedupe_preserve_order(errors),
        "warnings": _dedupe_preserve_order(warnings),
        "audit_parse_stats": audit_stats,
        "contract_probe": contract_probe,
    }


def run_quality_pipeline(
    *,
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    eligibility_out: Path = DEFAULT_ELIGIBILITY_OUT,
    eligibility_gates_out: Path = DEFAULT_ELIGIBILITY_GATES_OUT,
    context_out: Path = DEFAULT_CONTEXT_OUT,
    charts_out_dir: Path = DEFAULT_OUT_DIR,
    metrics_out: Path = DEFAULT_METRICS_PATH,
    enable_residual_experiment: bool = False,
    residual_experiment_out: Path = DEFAULT_RESIDUAL_EXPERIMENT_OUT,
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    eligibility = compute_eligibility(db_path=db_path)
    append_threshold_hash_change_warning(eligibility_out, eligibility)
    _write_json(eligibility_out, eligibility)
    eligibility_warnings = list(eligibility.get("warnings", []))
    eligibility_errors = list(eligibility.get("errors", []))
    if eligibility.get("n_tickers", 0) == 0:
        eligibility_warnings.append("no_tickers_found")
    if int((eligibility.get("summary") or {}).get("HEALTHY", 0)) == 0:
        eligibility_warnings.append("zero_healthy_tickers")
    eligibility_warnings = _dedupe_preserve_order(eligibility_warnings)
    eligibility_errors = _dedupe_preserve_order(eligibility_errors)
    eligibility_error_flag = "eligibility_error" if eligibility_errors else None
    steps.append(
        {
            "name": "compute_ticker_eligibility",
            "status": _step_status_from_flags(warnings=eligibility_warnings, error=eligibility_error_flag),
            "warnings": eligibility_warnings,
            "errors": eligibility_errors,
            "output": str(eligibility_out),
        }
    )
    warnings.extend(eligibility_warnings)
    errors.extend(eligibility_errors)

    eligibility_gate = apply_eligibility_gates(
        eligibility_path=eligibility_out,
        output_path=eligibility_gates_out,
    )
    eligibility_gate_warnings = _dedupe_preserve_order(list(eligibility_gate.get("warnings", [])))
    eligibility_gate_errors = _dedupe_preserve_order(list(eligibility_gate.get("errors", [])))
    eligibility_gate_error_flag = "eligibility_gate_error" if eligibility_gate_errors else None
    steps.append(
        {
            "name": "apply_ticker_eligibility_gates",
            "status": _step_status_from_flags(
                warnings=eligibility_gate_warnings,
                error=eligibility_gate_error_flag,
            ),
            "warnings": eligibility_gate_warnings,
            "errors": eligibility_gate_errors,
            "lab_only_tickers": list(eligibility_gate.get("lab_only_tickers", [])),
            "gate_written": bool(eligibility_gate.get("gate_written", False)),
            "output": str(eligibility_gates_out),
        }
    )
    warnings.extend(eligibility_gate_warnings)
    errors.extend(eligibility_gate_errors)

    context = compute_context_quality(db_path=db_path)
    append_threshold_hash_change_warning(context_out, context)
    _write_json(context_out, context)
    context_warnings = list(context.get("warnings", []))
    context_errors: list[str] = []
    if isinstance(context.get("schema_used"), dict) and context.get("schema_used", {}).get("db_error"):
        context_errors.append(str(context["schema_used"]["db_error"]))
    for code in ("db_open_failed", "query_failed"):
        if code in context_warnings:
            context_errors.append(code)
    if context.get("partial_data"):
        context_warnings.append("partial_data")
    context_warnings = _dedupe_preserve_order(context_warnings)
    context_errors = _dedupe_preserve_order(context_errors)
    context_error_flag = "context_quality_error" if context_errors else None
    steps.append(
        {
            "name": "compute_context_quality",
            "status": _step_status_from_flags(warnings=context_warnings, error=context_error_flag),
            "warnings": context_warnings,
            "errors": context_errors,
            "output": str(context_out),
        }
    )
    warnings.extend(context_warnings)
    errors.extend(context_errors)

    sufficiency = run_data_sufficiency(db_path=db_path, audit_dir=audit_dir) if db_path.exists() else {
        "status": "DATA_ERROR",
        "sufficient": False,
        "recommendations": [f"DB not found: {db_path}"],
    }
    sufficiency_warnings = []
    sufficiency_error = None
    if sufficiency.get("status") == "DATA_ERROR":
        sufficiency_error = "data_error"
        errors.extend(sufficiency.get("recommendations", []) or [sufficiency_error])
    elif sufficiency.get("status") != "SUFFICIENT":
        sufficiency_warnings.append("insufficient_data")
    sufficiency_warnings = _dedupe_preserve_order(sufficiency_warnings)
    steps.append(
        {
            "name": "data_sufficiency_monitor",
            "status": _step_status_from_flags(warnings=sufficiency_warnings, error=sufficiency_error),
            "warnings": sufficiency_warnings,
            "output": None,
        }
    )
    warnings.extend(sufficiency_warnings)

    chart_result = generate_performance_artifacts(
        db_path=db_path,
        audit_dir=audit_dir,
        out_dir=charts_out_dir,
        eligibility_path=eligibility_out,
        context_quality_path=context_out,
        json_metrics_path=metrics_out,
        sufficiency=sufficiency,
        strict_mode=True,
    )
    chart_warnings = list(chart_result.get("warnings", []))
    chart_errors = list(chart_result.get("errors", []))
    chart_warnings = _dedupe_preserve_order(chart_warnings)
    chart_errors = _dedupe_preserve_order(chart_errors)
    chart_error_flag = "chart_generation_error" if chart_errors else None
    steps.append(
        {
            "name": "generate_performance_charts",
            "status": _step_status_from_flags(warnings=chart_warnings, error=chart_error_flag),
            "warnings": chart_warnings,
            "errors": chart_errors,
            "output": str(metrics_out),
        }
    )
    warnings.extend(chart_warnings)
    errors.extend(chart_errors)

    if enable_residual_experiment:
        residual_summary = _collect_residual_experiment_summary(audit_dir, db_path)
        _write_json(residual_experiment_out, residual_summary)
        residual_warnings = _dedupe_preserve_order(list(residual_summary.get("warnings", [])))
        residual_errors = _dedupe_preserve_order(list(residual_summary.get("errors", [])))
        residual_error_flag = "residual_experiment_error" if residual_errors else None
        steps.append(
            {
                "name": "residual_experiment_audit",
                "status": _step_status_from_flags(warnings=residual_warnings, error=residual_error_flag),
                "warnings": residual_warnings,
                "errors": residual_errors,
                "output": str(residual_experiment_out),
                "n_windows_with_residual_metrics": residual_summary.get("n_windows_with_residual_metrics", 0),
            }
        )
        warnings.extend(residual_warnings)
        errors.extend(residual_errors)

    if errors:
        status = "ERROR"
    elif any(step["status"] == "WARN" for step in steps):
        status = "WARN"
    else:
        status = "PASS"

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "ok": status != "ERROR",
        "status": status,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(max(0.0, time.perf_counter() - started_perf), 4),
        "db_path": str(db_path),
        "audit_dir": str(audit_dir),
        "thresholds": threshold_map(),
        "steps": steps,
        "artifacts": {
            "eligibility": str(eligibility_out),
            "eligibility_gates": str(eligibility_gates_out),
            "context_quality": str(context_out),
            "charts_out_dir": str(charts_out_dir),
            "metrics_summary": str(metrics_out),
            **({"residual_experiment": str(residual_experiment_out)} if enable_residual_experiment else {}),
        },
        "residual_experiment_enabled": bool(enable_residual_experiment),
        "eligibility_summary": eligibility.get("summary", {}),
        "sufficiency": sufficiency,
        "warnings": sorted(set(warnings)),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only quality pipeline.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--eligibility-out", type=Path, default=DEFAULT_ELIGIBILITY_OUT)
    parser.add_argument("--eligibility-gates-out", type=Path, default=DEFAULT_ELIGIBILITY_GATES_OUT)
    parser.add_argument("--context-out", type=Path, default=DEFAULT_CONTEXT_OUT)
    parser.add_argument("--charts-out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--enable-residual-experiment", action="store_true")
    parser.add_argument("--residual-experiment-out", type=Path, default=DEFAULT_RESIDUAL_EXPERIMENT_OUT)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = run_quality_pipeline(
        db_path=args.db,
        audit_dir=args.audit_dir,
        eligibility_out=args.eligibility_out,
        eligibility_gates_out=args.eligibility_gates_out,
        context_out=args.context_out,
        charts_out_dir=args.charts_out_dir,
        metrics_out=args.metrics_out,
        enable_residual_experiment=bool(args.enable_residual_experiment),
        residual_experiment_out=args.residual_experiment_out,
    )

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Quality pipeline status: {result['status']}")
        for step in result["steps"]:
            print(f"  {step['name']}: {step['status']}")
    return 0 if result["status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    sys.exit(main())
