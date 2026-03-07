#!/usr/bin/env python3
"""
check_forecast_audits.py
------------------------

Brutal-style sanity check for Time Series forecaster performance.

Reads the most recent forecast audit JSON files emitted by
forcester_ts/forecaster.py (via ModelInstrumentation) from
logs/forecast_audits/, compares ensemble regression metrics to a
baseline model, and exits non-zero if the ensemble underperforms
systematically.

This script is read-only and safe to call from brutal/dry-run
or CI workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scripts.audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT
except Exception:  # pragma: no cover - script execution path fallback
    from audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT

try:
    from scripts.telemetry_adapter import normalize_telemetry_payload, sha256_file, telemetry_now_utc
except Exception:  # pragma: no cover - script execution path fallback
    from telemetry_adapter import normalize_telemetry_payload, sha256_file, telemetry_now_utc

try:
    from scripts.quality_pipeline_common import (
        resolve_forecast_audit_dir,
        resolve_forecast_audit_roots,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from quality_pipeline_common import (
        resolve_forecast_audit_dir,
        resolve_forecast_audit_roots,
    )

DEFAULT_AUDIT_ROOT = Path("logs/forecast_audits")
DEFAULT_AUDIT_PRODUCTION_DIR = DEFAULT_AUDIT_ROOT / "production"
DEFAULT_AUDIT_DIR = DEFAULT_AUDIT_PRODUCTION_DIR
DEFAULT_MONITORING_CONFIG = Path("config/forecaster_monitoring.yml")
DEFAULT_BASELINE_MODEL = "BEST_SINGLE"
DEFAULT_DECISION_KEEP = "KEEP"
DEFAULT_DECISION_RESEARCH = "RESEARCH_ONLY"
DEFAULT_DECISION_DISABLE = "DISABLE_DEFAULT"
DEFAULT_MANIFEST_FILENAME = "forecast_audit_manifest.jsonl"
DEFAULT_MANIFEST_MODE = "off"
MANIFEST_MODES = {"off", "warn", "fail"}
TELEMETRY_SCHEMA_VERSION = 3
OUTCOME_ELIGIBILITY_BUFFER = timedelta(minutes=5)


@dataclass
class AuditCheckResult:
    path: Path
    ensemble_rmse: Optional[float]
    baseline_rmse: Optional[float]
    rmse_ratio: Optional[float]
    violation: bool
    baseline_model: Optional[str] = None
    ensemble_missing: bool = False


def _load_audit_with_error(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, str(exc)


def _load_audit(path: Path) -> Optional[Dict[str, Any]]:
    payload, _ = _load_audit_with_error(path)
    return payload


def _sha256_file(path: Path) -> Optional[str]:
    digest, _ = sha256_file(path)
    return digest


def _parse_window_day(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except Exception:
        return text[:10] if len(text) >= 10 else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            # Date-only windows are interpreted as end-of-day UTC to avoid
            # prematurely treating same-day windows as outcome-eligible.
            day_start = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            return day_start + timedelta(days=1) - timedelta(seconds=1)
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _expected_close_ts(end_raw: Any, horizon_raw: Any) -> Optional[datetime]:
    end_ts = _parse_utc_datetime(end_raw)
    if end_ts is None:
        return None
    try:
        horizon = int(horizon_raw)
    except (TypeError, ValueError):
        return None
    if horizon < 0:
        return None
    return end_ts + timedelta(days=horizon)


def _parse_non_negative_int(raw: Any) -> Optional[int]:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def compute_expected_close(
    signal_context: Dict[str, Any],
    dataset: Dict[str, Any],
) -> Tuple[Optional[datetime], str]:
    """Prefer signal-context timing for outcome eligibility.

    Returns (expected_close_ts, source) where source is one of:
      - signal_context
      - dataset_fallback
      - signal_context_invalid
      - unavailable
    """
    ctx = signal_context if isinstance(signal_context, dict) else {}
    has_signal_context = bool(ctx) and not bool(ctx.get("signal_context_missing"))
    if has_signal_context:
        entry_ts = _parse_utc_datetime(ctx.get("entry_ts"))
        horizon = _parse_non_negative_int(ctx.get("forecast_horizon"))
        if entry_ts is None or horizon is None:
            return None, "signal_context_invalid"
        return entry_ts + timedelta(days=horizon), "signal_context"

    fallback = _expected_close_ts(dataset.get("end"), dataset.get("forecast_horizon"))
    if fallback is None:
        return None, "unavailable"
    return fallback, "dataset_fallback"


def _load_closed_trade_match_counts(
    db_path: Optional[Path],
) -> Tuple[bool, Dict[str, int], Optional[str]]:
    if db_path is None:
        return False, {}, "db_not_configured"
    if not db_path.exists():
        return False, {}, f"db_not_found:{db_path}"
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts_signal_id, COUNT(*) AS n
            FROM production_closed_trades
            WHERE ts_signal_id IS NOT NULL AND TRIM(ts_signal_id) <> ''
            GROUP BY ts_signal_id
            """
        )
        mapping = {}
        for ts_signal_id, count in cur.fetchall():
            sid = str(ts_signal_id or "").strip()
            if sid:
                mapping[sid] = int(count or 0)
        return True, mapping, None
    except Exception as exc:
        return False, {}, str(exc)
    finally:
        if conn is not None:
            conn.close()


def _extract_window_metadata(audit: Dict[str, Any]) -> Dict[str, Optional[str]]:
    dataset = audit.get("dataset") or {}
    ticker = str(dataset.get("ticker") or dataset.get("symbol") or "").strip().upper() or None
    regime = str(dataset.get("detected_regime") or dataset.get("regime") or "").strip().upper() or None
    return {
        "ticker": ticker,
        "detected_regime": regime,
        "end_day": _parse_window_day(dataset.get("end")),
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for _attempt in range(2):
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            with tmp_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            os.replace(tmp_path, path)
            return
        except Exception as exc:
            last_error = exc
            try:
                tmp_path.unlink()
            except OSError:
                pass
    if last_error is not None:
        raise last_error


def _resolve_audit_roots(audit_dir: Path, include_research: bool) -> List[Path]:
    return resolve_forecast_audit_roots(
        audit_dir,
        include_research=include_research,
        default_audit_root=DEFAULT_AUDIT_ROOT,
    )


def _collect_audit_files(
    *,
    audit_roots: List[Path],
    max_files: int,
) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for root in audit_roots:
        if not root.exists():
            continue
        for path in root.glob("forecast_audit_*.json"):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:max(0, int(max_files))]


def _load_manifest_index(manifest_path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    index: Dict[str, str] = {}
    stats = {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "invalid_records": 0,
    }
    if not manifest_path.exists():
        return index, stats

    for raw_line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            stats["invalid_records"] += 1
            continue
        file_name = str(payload.get("file") or "").strip()
        digest = str(payload.get("sha256") or "").strip().lower()
        if not file_name or len(digest) != 64:
            stats["invalid_records"] += 1
            continue
        index[file_name] = digest
    return index, stats


def _verify_manifest_entry(path: Path, manifest_index: Dict[str, str]) -> str:
    """
    Return manifest verification status for an audit artifact.

    Status values:
    - ok
    - missing
    - hash_failed
    - mismatch
    """
    expected = manifest_index.get(path.name)
    if not expected:
        return "missing"
    actual = _sha256_file(path)
    if not actual:
        return "hash_failed"
    if actual.lower() != expected.lower():
        return "mismatch"
    return "ok"


def _extract_metrics(
    audit: Dict[str, Any], *, baseline_model: str
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[str]]:
    """
    Return (ensemble_metrics, baseline_metrics, resolved_baseline_model) from an audit payload.

    Ensemble metrics are taken from artifacts['evaluation_metrics']['ensemble']
    when available.

    Baseline selection:
    - BEST_SINGLE: choose the available single-model entry with the smallest RMSE.
      Candidate set includes SARIMAX, GARCH, SAMOSSA, and MSSA_RL.
    - SAMOSSA: use samossa when present; else fall back to BEST_SINGLE.
    - GARCH: use garch when present; else fall back to BEST_SINGLE.
    - SARIMAX: use sarimax when present; else fall back to BEST_SINGLE.
    """
    artifacts = audit.get("artifacts") or {}
    eval_metrics = artifacts.get("evaluation_metrics") or {}
    if not isinstance(eval_metrics, dict):
        return None, None, None

    ensemble = eval_metrics.get("ensemble")
    sarimax = eval_metrics.get("sarimax")
    garch = eval_metrics.get("garch")
    samossa = eval_metrics.get("samossa")

    if ensemble is None and sarimax is None and garch is None and samossa is None:
        return None, None, None

    ensemble_metrics = ensemble if isinstance(ensemble, dict) else None

    baseline_model = (baseline_model or DEFAULT_BASELINE_MODEL).strip().upper()

    def _best_single() -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for name in ("sarimax", "garch", "samossa", "mssa_rl"):
            payload = eval_metrics.get(name)
            if isinstance(payload, dict):
                candidates.append((name, payload))
        best_payload: Optional[Dict[str, Any]] = None
        best_rmse: Optional[float] = None
        best_name: Optional[str] = None
        for name, payload in candidates:
            rmse = _rmse_from(payload)
            if rmse is None:
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_payload = payload
                best_name = name.upper()
        if best_payload is not None:
            return best_payload, best_name
        if isinstance(sarimax, dict):
            return sarimax, "SARIMAX"
        if isinstance(garch, dict):
            return garch, "GARCH"
        if isinstance(samossa, dict):
            return samossa, "SAMOSSA"
        return None, None

    if baseline_model == "SAMOSSA":
        if isinstance(samossa, dict):
            baseline_metrics = samossa
            resolved_baseline = "SAMOSSA"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    elif baseline_model == "GARCH":
        if isinstance(garch, dict):
            baseline_metrics = garch
            resolved_baseline = "GARCH"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    elif baseline_model == "SARIMAX":
        if isinstance(sarimax, dict):
            baseline_metrics = sarimax
            resolved_baseline = "SARIMAX"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    else:
        baseline_metrics, resolved_baseline = _best_single()

    return ensemble_metrics, baseline_metrics if isinstance(baseline_metrics, dict) else None, resolved_baseline


def _rmse_from(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("rmse")
    return float(val) if isinstance(val, (int, float)) else None


def check_audit_file(
    path: Path,
    tolerance: float,
    *,
    baseline_model: str,
) -> Optional[AuditCheckResult]:
    audit = _load_audit(path)
    if not audit:
        return None

    ensemble_metrics, baseline_metrics, resolved_baseline = _extract_metrics(
        audit, baseline_model=baseline_model
    )
    ensemble_rmse = _rmse_from(ensemble_metrics)
    baseline_rmse = _rmse_from(baseline_metrics)
    ensemble_missing = (
        ensemble_metrics is None
        and baseline_rmse is not None
        and baseline_rmse > 0
    )

    if ensemble_rmse is None or baseline_rmse is None or baseline_rmse <= 0:
        return AuditCheckResult(
            path=path,
            ensemble_rmse=ensemble_rmse,
            baseline_rmse=baseline_rmse,
            rmse_ratio=None,
            violation=False,
            baseline_model=resolved_baseline,
            ensemble_missing=ensemble_missing,
        )

    rmse_ratio = ensemble_rmse / baseline_rmse
    violation = rmse_ratio > (1.0 + tolerance)

    return AuditCheckResult(
        path=path,
        ensemble_rmse=ensemble_rmse,
        baseline_rmse=baseline_rmse,
        rmse_ratio=rmse_ratio,
        violation=violation,
        baseline_model=resolved_baseline,
        ensemble_missing=ensemble_missing,
    )


def _load_monitoring_thresholds(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml  # Local import to keep dependency optional
    except ImportError:
        return {}

    raw = yaml.safe_load(config_path.read_text()) or {}
    fm = raw.get("forecaster_monitoring") or {}
    return fm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check TS forecast audit files for ensemble underperformance."
    )
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help="Directory containing forecast_audit_*.json files "
        "(default: logs/forecast_audits/production when present, else logs/forecast_audits)",
    )
    parser.add_argument(
        "--include-research",
        action="store_true",
        help=(
            "Include logs/forecast_audits/research alongside the selected audit directory. "
            "Default mode inspects production audits only."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=FORECAST_AUDIT_MAX_FILES_DEFAULT,
        help=(
            "Maximum number of most recent audit files to inspect "
            f"(default: {FORECAST_AUDIT_MAX_FILES_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Allowed RMSE degradation vs baseline before flagging a violation. "
        "If omitted, will fall back to config/forecaster_monitoring.yml or 0.10.",
    )
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        default=None,
        help="Maximum fraction of checked audits allowed to violate the RMSE tolerance "
        "before exiting non-zero. If omitted, will fall back to config/forecaster_monitoring.yml or 0.25.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help="Optional path to forecaster_monitoring.yml "
        "(default: config/forecaster_monitoring.yml if present)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Optional path to SQLite DB for deterministic ts_signal_id outcome joins. "
            "If omitted, outcome join telemetry stays disabled."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Baseline model for the RMSE gate: BEST_SINGLE, SAMOSSA, GARCH, or SARIMAX. "
        "If omitted, uses forecaster_monitoring.regression_metrics.baseline_model "
        f"or {DEFAULT_BASELINE_MODEL}.",
    )
    parser.add_argument(
        "--require-effective-audits",
        type=int,
        default=None,
        help="If set, exit non-zero when effective audits with RMSE metrics are below this count.",
    )
    parser.add_argument(
        "--require-holding-period",
        action="store_true",
        help="If set, require effective audits to meet holding_period_audits from the monitoring config.",
    )
    parser.add_argument(
        "--manifest-integrity-mode",
        default=None,
        choices=sorted(MANIFEST_MODES),
        help=(
            "Audit provenance enforcement mode: off|warn|fail. "
            "If omitted, uses forecaster_monitoring.regression_metrics.manifest_integrity_mode."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Path to forecast audit manifest JSONL. "
            "If omitted, uses <audit-dir>/forecast_audit_manifest.jsonl "
            "or regression_metrics.manifest_filename."
        ),
    )
    parser.add_argument(
        "--max-missing-ensemble-rate",
        type=float,
        default=None,
        help=(
            "Maximum fraction of unique audits allowed to miss ensemble metrics "
            "before exiting non-zero. If omitted, uses config value or 1.0."
        ),
    )
    parser.add_argument(
        "--min-forecast-horizon",
        type=int,
        default=None,
        help=(
            "Minimum dataset.forecast_horizon required for an audit artifact "
            "to participate in gate statistics. If omitted, uses "
            "regression_metrics.min_forecast_horizon when present."
        ),
    )
    return parser


def _resolve_main_inputs(
    args: argparse.Namespace,
) -> Tuple[Path, Optional[Path], List[Path], List[Path]]:
    requested_audit_dir = Path(args.audit_dir)
    audit_dir = resolve_forecast_audit_dir(
        requested_audit_dir,
        default_audit_root=DEFAULT_AUDIT_ROOT,
        default_audit_production_dir=DEFAULT_AUDIT_PRODUCTION_DIR,
    )
    if (
        audit_dir == DEFAULT_AUDIT_ROOT
        and not DEFAULT_AUDIT_ROOT.exists()
        and requested_audit_dir == DEFAULT_AUDIT_PRODUCTION_DIR
    ):
        # Keep legacy behavior when neither production nor root audit directories exist.
        audit_dir = requested_audit_dir
    db_path = Path(args.db) if args.db else None
    audit_roots = _resolve_audit_roots(audit_dir, bool(args.include_research))
    files = _collect_audit_files(audit_roots=audit_roots, max_files=int(args.max_files))
    if not files:
        roots_text = ", ".join(str(root) for root in audit_roots)
        raise SystemExit(f"No forecast_audit_*.json files found in: {roots_text}")
    return audit_dir, db_path, audit_roots, files


def _resolve_monitoring_settings(
    args: argparse.Namespace,
    *,
    audit_dir: Path,
) -> Dict[str, Any]:
    monitoring_cfg = _load_monitoring_thresholds(
        Path(args.config_path) if args.config_path else None
    )
    rmse_cfg = monitoring_cfg.get("regression_metrics") if monitoring_cfg else {}

    tolerance = (
        float(args.tolerance)
        if args.tolerance is not None
        else float(rmse_cfg.get("max_rmse_ratio_vs_baseline", 1.10)) - 1.0
    )
    max_violation_rate = (
        float(args.max_violation_rate)
        if args.max_violation_rate is not None
        else float(rmse_cfg.get("max_violation_rate", 0.25))
    )
    min_effective_audits = int(rmse_cfg.get("min_effective_audits", 0) or 0)
    baseline_model = (
        str(args.baseline_model)
        if args.baseline_model
        else str(rmse_cfg.get("baseline_model", DEFAULT_BASELINE_MODEL))
    )
    holding_period = int(rmse_cfg.get("holding_period_audits", 0) or 0)
    fail_on_violation_during_holding_period = bool(
        rmse_cfg.get("fail_on_violation_during_holding_period", False)
    )
    disable_if_no_lift = bool(rmse_cfg.get("disable_ensemble_if_no_lift", False))
    min_lift_rmse_ratio = float(rmse_cfg.get("min_lift_rmse_ratio", 0.0) or 0.0)
    min_lift_fraction = float(rmse_cfg.get("min_lift_fraction", 0.0) or 0.0)
    promotion_margin = float(rmse_cfg.get("promotion_margin", 0.0) or 0.0)
    recent_window_audits = max(int(rmse_cfg.get("recent_window_audits", 0) or 0), 0)
    recent_window_max_violation_rate = float(
        rmse_cfg.get("recent_window_max_violation_rate", max_violation_rate)
    )
    raw_recent_p90 = rmse_cfg.get("recent_window_max_p90_rmse_ratio")
    recent_window_max_p90_rmse_ratio = (
        float(raw_recent_p90)
        if isinstance(raw_recent_p90, (int, float))
        else None
    )
    manifest_mode = (
        str(args.manifest_integrity_mode).strip().lower()
        if args.manifest_integrity_mode
        else str(rmse_cfg.get("manifest_integrity_mode", DEFAULT_MANIFEST_MODE)).strip().lower()
    )
    if manifest_mode not in MANIFEST_MODES:
        manifest_mode = DEFAULT_MANIFEST_MODE
    manifest_filename = str(rmse_cfg.get("manifest_filename", DEFAULT_MANIFEST_FILENAME))
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else (audit_dir / manifest_filename)
    )
    max_missing_ensemble_rate = (
        float(args.max_missing_ensemble_rate)
        if args.max_missing_ensemble_rate is not None
        else float(rmse_cfg.get("max_missing_ensemble_rate", 1.0))
    )
    min_forecast_horizon = (
        int(args.min_forecast_horizon)
        if args.min_forecast_horizon is not None
        else (
            int(rmse_cfg.get("min_forecast_horizon"))
            if rmse_cfg.get("min_forecast_horizon") is not None
            else None
        )
    )
    if min_forecast_horizon is not None and min_forecast_horizon < 0:
        min_forecast_horizon = 0

    return {
        "monitoring_cfg": monitoring_cfg,
        "rmse_cfg": rmse_cfg,
        "tolerance": tolerance,
        "max_violation_rate": max_violation_rate,
        "min_effective_audits": min_effective_audits,
        "baseline_model": baseline_model,
        "holding_period": holding_period,
        "fail_on_violation_during_holding_period": fail_on_violation_during_holding_period,
        "disable_if_no_lift": disable_if_no_lift,
        "min_lift_rmse_ratio": min_lift_rmse_ratio,
        "min_lift_fraction": min_lift_fraction,
        "promotion_margin": promotion_margin,
        "recent_window_audits": recent_window_audits,
        "recent_window_max_violation_rate": recent_window_max_violation_rate,
        "recent_window_max_p90_rmse_ratio": recent_window_max_p90_rmse_ratio,
        "manifest_mode": manifest_mode,
        "manifest_path": manifest_path,
        "max_missing_ensemble_rate": max_missing_ensemble_rate,
        "min_forecast_horizon": min_forecast_horizon,
    }


def main() -> None:
    args = _build_parser().parse_args()
    audit_dir, db_path, audit_roots, files = _resolve_main_inputs(args)
    settings = _resolve_monitoring_settings(args, audit_dir=audit_dir)
    monitoring_cfg = settings["monitoring_cfg"]
    rmse_cfg = settings["rmse_cfg"]
    tolerance = settings["tolerance"]
    max_violation_rate = settings["max_violation_rate"]
    min_effective_audits = settings["min_effective_audits"]
    baseline_model = settings["baseline_model"]
    holding_period = settings["holding_period"]
    fail_on_violation_during_holding_period = settings["fail_on_violation_during_holding_period"]
    disable_if_no_lift = settings["disable_if_no_lift"]
    min_lift_rmse_ratio = settings["min_lift_rmse_ratio"]
    min_lift_fraction = settings["min_lift_fraction"]
    promotion_margin = settings["promotion_margin"]
    recent_window_audits = settings["recent_window_audits"]
    recent_window_max_violation_rate = settings["recent_window_max_violation_rate"]
    recent_window_max_p90_rmse_ratio = settings["recent_window_max_p90_rmse_ratio"]
    manifest_mode = settings["manifest_mode"]
    manifest_path = settings["manifest_path"]
    max_missing_ensemble_rate = settings["max_missing_ensemble_rate"]
    min_forecast_horizon = settings["min_forecast_horizon"]

    def _rmse_dedupe_key_from_audit(audit: Dict[str, Any]) -> Tuple[Any, ...]:
        dataset = audit.get("dataset") or {}
        ds_key = (
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )
        # Deduplicate by data window (start/end/length/horizon) only and keep the
        # most recent file as the authoritative result. This prevents stale
        # earlier runs (often with different ensemble weights) from inflating
        # the violation rate and aligns with "latest evidence wins" monitoring.
        return ds_key

    def _outcome_dedupe_key_from_audit(audit: Dict[str, Any]) -> Tuple[Any, ...]:
        dataset = audit.get("dataset") or {}
        ticker = str(dataset.get("ticker") or dataset.get("symbol") or "").strip().upper() or None
        return (
            ticker,
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )

    def _forecast_horizon_from_audit(audit: Dict[str, Any]) -> Optional[int]:
        dataset = audit.get("dataset") or {}
        raw = dataset.get("forecast_horizon")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    unique_map: dict[Tuple[Any, ...], Path] = {}
    outcome_unique_map: dict[Tuple[Any, ...], Path] = {}
    horizon_filtered_count = 0
    parseable_count = 0
    parse_error_count = 0

    manifest_index: Dict[str, str] = {}
    manifest_stats: Dict[str, Any] = {
        "mode": manifest_mode,
        "manifest_path": str(manifest_path),
        "manifest_exists": None,
        "invalid_records": 0,
        "verified": 0,
        "missing": 0,
        "hash_failed": 0,
        "mismatch": 0,
        "excluded_unverified": 0,
    }
    if manifest_mode != "off":
        manifest_index, loaded_stats = _load_manifest_index(manifest_path)
        manifest_stats.update(loaded_stats)

    for f in files:
        audit, load_error = _load_audit_with_error(f)
        if not audit:
            if load_error:
                parse_error_count += 1
            continue
        parseable_count += 1

        if min_forecast_horizon is not None:
            horizon = _forecast_horizon_from_audit(audit)
            if horizon is None or horizon < min_forecast_horizon:
                horizon_filtered_count += 1
                continue

        if manifest_mode != "off":
            status = _verify_manifest_entry(f, manifest_index)
            if status == "ok":
                manifest_stats["verified"] += 1
            else:
                manifest_stats[status] += 1
                if manifest_mode == "fail":
                    manifest_stats["excluded_unverified"] += 1
                    continue
        key = _rmse_dedupe_key_from_audit(audit)
        # Files are sorted newest-first, so keep the first (newest) entry we see
        # for each dataset window and ignore older duplicates.
        if key in unique_map:
            pass
        else:
            unique_map[key] = f
        outcome_key = _outcome_dedupe_key_from_audit(audit)
        if outcome_key not in outcome_unique_map:
            outcome_unique_map[outcome_key] = f

    unique_files: List[Path] = list(unique_map.values())
    outcome_unique_files: List[Path] = list(outcome_unique_map.values())

    results: List[AuditCheckResult] = []
    for f in unique_files:
        res = check_audit_file(f, tolerance, baseline_model=baseline_model)
        if res is not None:
            results.append(res)

    print("=== Forecast Audit Regression Check ===")
    print(f"Audit directory : {audit_dir}")
    print(
        "Audit roots    : "
        + ", ".join(str(root) for root in audit_roots)
        + f" (include_research={int(bool(args.include_research))})"
    )
    print(
        f"Files inspected : {len(results)} unique (raw={len(files)}, max_files={args.max_files})"
    )
    print(
        "Parse stats     : "
        f"parseable={parseable_count} "
        f"parse_errors={parse_error_count}"
    )
    print(
        "Window stats    : "
        f"deduped={len(unique_map)} "
        f"checked={len(results)}"
    )
    print(f"Baseline model  : {baseline_model}")
    print(f"RMSE tolerance  : ensemble_rmse <= (1 + {tolerance:.2f}) * baseline_rmse")
    if min_effective_audits > 0:
        print(f"Min effective   : {min_effective_audits} audit(s) before hard gating")
    if holding_period > 0:
        print(f"Holding period  : {holding_period} effective audit(s)")
        if fail_on_violation_during_holding_period:
            print("Warmup behavior : fail on violations during holding period")
    if disable_if_no_lift:
        print(
            "No-lift gate    : enabled "
            f"(min_lift_rmse_ratio={min_lift_rmse_ratio:.2%}, "
            f"min_lift_fraction={min_lift_fraction:.2%})"
        )
    if promotion_margin > 0:
        print(f"Promotion margin: requires >= {promotion_margin:.2%} lift to keep ensemble as default")
    if recent_window_audits > 0:
        print(
            f"Recent window  : {recent_window_audits} effective audit(s) "
            f"(max violation rate {recent_window_max_violation_rate:.2%})"
        )
    if min_forecast_horizon is not None:
        print(
            "Horizon filter : "
            f"forecast_horizon >= {min_forecast_horizon} "
            f"(excluded={horizon_filtered_count})"
        )
    if recent_window_max_p90_rmse_ratio is not None:
        print(
            "Recent p90 gate: "
            f"p90(rmse_ratio) <= {recent_window_max_p90_rmse_ratio:.3f}"
        )
    if manifest_mode != "off":
        print(
            "Manifest check : "
            f"mode={manifest_mode} path={manifest_stats.get('manifest_path')} "
            f"exists={manifest_stats.get('manifest_exists')}"
        )
        print(
            "Manifest stats : "
            f"verified={manifest_stats.get('verified', 0)} "
            f"missing={manifest_stats.get('missing', 0)} "
            f"mismatch={manifest_stats.get('mismatch', 0)} "
            f"hash_failed={manifest_stats.get('hash_failed', 0)} "
            f"invalid_records={manifest_stats.get('invalid_records', 0)}"
        )
        if manifest_mode == "fail":
            unverified = (
                int(manifest_stats.get("missing", 0))
                + int(manifest_stats.get("mismatch", 0))
                + int(manifest_stats.get("hash_failed", 0))
                + int(manifest_stats.get("invalid_records", 0))
            )
            if not bool(manifest_stats.get("manifest_exists", False)):
                raise SystemExit(
                    f"Manifest integrity mode=fail but manifest file is missing: {manifest_path}"
                )
            if unverified > 0:
                raise SystemExit(
                    "Manifest integrity failed: "
                    f"missing={manifest_stats.get('missing', 0)} "
                    f"mismatch={manifest_stats.get('mismatch', 0)} "
                    f"hash_failed={manifest_stats.get('hash_failed', 0)} "
                    f"invalid_records={manifest_stats.get('invalid_records', 0)}"
                )

    violation_count = sum(1 for r in results if r.violation)
    rmse_windows_processed = len(results)
    effective_n = sum(
        1
        for r in results
        if (
            r.ensemble_rmse is not None
            and r.baseline_rmse is not None
            and r.baseline_rmse > 0
        )
    )
    violation_rate = (violation_count / effective_n) if effective_n else 0.0
    ensemble_missing_count = sum(1 for r in results if r.ensemble_missing)
    ensemble_missing_rate = (ensemble_missing_count / len(results)) if results else 0.0

    def _percentiles(values: list[float], percents: list[float]) -> Dict[float, float]:
        if not values:
            return {}
        vals = sorted(values)
        out: Dict[float, float] = {}
        for p in percents:
            if p <= 0:
                out[p] = vals[0]
                continue
            if p >= 1:
                out[p] = vals[-1]
                continue
            idx = (len(vals) - 1) * p
            lower = int(idx)
            upper = min(lower + 1, len(vals) - 1)
            weight = idx - lower
            out[p] = vals[lower] * (1 - weight) + vals[upper] * weight
        return out

    ratios = [
        r.rmse_ratio
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    pct = _percentiles(ratios, [0.1, 0.5, 0.9]) if ratios else {}
    recent_results: List[AuditCheckResult] = (
        results[:recent_window_audits] if recent_window_audits > 0 else []
    )
    recent_effective_n = sum(
        1
        for r in recent_results
        if (
            r.ensemble_rmse is not None
            and r.baseline_rmse is not None
            and r.baseline_rmse > 0
        )
    )
    recent_violation_count = sum(1 for r in recent_results if r.violation)
    recent_violation_rate = (
        (recent_violation_count / recent_effective_n) if recent_effective_n else 0.0
    )
    recent_ratios = [
        r.rmse_ratio
        for r in recent_results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    recent_pct = _percentiles(recent_ratios, [0.5, 0.9]) if recent_ratios else {}
    rmse_windows_usable = effective_n
    outcomes_loaded = False
    outcome_join_attempted = False
    outcome_join_error: Optional[str] = None
    outcome_windows_eligible = 0
    outcome_windows_matched = 0
    outcome_windows_missing = 0
    outcome_windows_ambiguous = 0
    outcome_windows_not_due = 0
    outcome_windows_not_yet_eligible = 0  # legacy alias for telemetry schema v2
    outcome_windows_invalid_context = 0
    outcome_windows_outcomes_not_loaded = 0
    outcome_windows_no_signal_id = 0
    outcome_windows_non_trade_context = 0
    outcome_windows_missing_execution_metadata = 0

    print(
        "RMSE coverage  : "
        f"raw={len(files)} "
        f"parseable={parseable_count} "
        f"deduped={len(unique_map)} "
        f"processed={rmse_windows_processed} "
        f"usable={rmse_windows_usable}"
    )
    print(
        "Outcome dedupe : "
        f"raw={len(files)} "
        f"deduped={len(outcome_unique_map)}"
    )

    print(f"\nEffective audits with RMSE: {effective_n}")
    print(f"Violations (ensemble worse than baseline beyond tolerance): {violation_count}")
    print(f"Violation rate: {violation_rate:.2%} (max allowed {max_violation_rate:.2%})")
    print(
        "Missing ensemble metrics      : "
        f"{ensemble_missing_count}/{len(results)} ({ensemble_missing_rate:.2%}) "
        f"(max allowed {max_missing_ensemble_rate:.2%})"
    )
    if (
        len(results) > 0
        and max_missing_ensemble_rate >= 0
    ):
        if ensemble_missing_rate > max_missing_ensemble_rate:
            raise SystemExit(
                "Missing ensemble metric rate "
                f"{ensemble_missing_rate:.2%} exceeds max "
                f"{max_missing_ensemble_rate:.2%}"
            )
    if pct:
        print(
            "RMSE ratio percentiles: "
            f"p10={pct.get(0.1):.3f}, median={pct.get(0.5):.3f}, p90={pct.get(0.9):.3f}"
        )
    if recent_window_audits > 0:
        print(
            "Recent window stats: "
            f"effective={recent_effective_n}/{recent_window_audits}, "
            f"violations={recent_violation_count}, "
            f"violation_rate={recent_violation_rate:.2%}"
        )
        if recent_pct:
            print(
                "Recent RMSE ratio percentiles: "
                f"median={recent_pct.get(0.5):.3f}, p90={recent_pct.get(0.9):.3f}"
            )

    if recent_window_audits > 0:
        if recent_effective_n < recent_window_audits:
            print(
                "Recent-window gate inconclusive: "
                f"effective_audits={recent_effective_n} < required_audits={recent_window_audits}"
            )
        else:
            if recent_violation_rate > recent_window_max_violation_rate:
                raise SystemExit(
                    f"Recent-window violation rate {recent_violation_rate:.2%} exceeds "
                    f"max-violation-rate {recent_window_max_violation_rate:.2%} "
                    f"(window={recent_window_audits})"
                )
            if (
                recent_window_max_p90_rmse_ratio is not None
                and recent_pct
                and recent_pct.get(0.9) is not None
                and float(recent_pct.get(0.9)) > float(recent_window_max_p90_rmse_ratio)
            ):
                raise SystemExit(
                    f"Recent-window p90 RMSE ratio {float(recent_pct.get(0.9)):.3f} exceeds "
                    f"{float(recent_window_max_p90_rmse_ratio):.3f} "
                    f"(window={recent_window_audits})"
                )

    warmup_required = max(min_effective_audits, holding_period, 0)
    rmse_inconclusive = False
    if warmup_required > 0 and effective_n < warmup_required:
        explicit_required: Optional[int] = None
        if args.require_holding_period and holding_period > 0:
            explicit_required = holding_period
        if args.require_effective_audits is not None:
            explicit_required = int(args.require_effective_audits)
        if explicit_required is not None and effective_n < explicit_required:
            raise SystemExit(
                f"Insufficient effective audits for RMSE gating: effective_audits={effective_n} "
                f"< required_audits={explicit_required}"
            )
        if (
            fail_on_violation_during_holding_period
            and effective_n > 0
            and violation_rate > max_violation_rate
        ):
            raise SystemExit(
                f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
                f"max-violation-rate {max_violation_rate:.2%} during holding period "
                f"(effective_audits={effective_n} < required_audits={warmup_required})"
            )
        print(
            f"\nRMSE gate inconclusive: effective_audits={effective_n} "
            f"< required_audits={warmup_required}.",
        )
        rmse_inconclusive = True

    if effective_n == 0:
        print("\nRMSE gate inconclusive: no usable RMSE metrics.")
        rmse_inconclusive = True

    if not rmse_inconclusive:
        print("\nSample details (most recent first):")
        header = f"{'File':<32} {'ens_rmse':>10} {'base_rmse':>10} {'ratio':>8} {'VIOL':>6}"
        print(header)
        print("-" * len(header))
        for r in results[:10]:
            ratio_str = f"{r.rmse_ratio:.3f}" if r.rmse_ratio is not None else "n/a"
            ens_str = f"{r.ensemble_rmse:.4f}" if r.ensemble_rmse is not None else "n/a"
            base_str = f"{r.baseline_rmse:.4f}" if r.baseline_rmse is not None else "n/a"
            viol_flag = "YES" if r.violation else ""
            display_name = r.path.name
            if r.baseline_model:
                display_name = f"{display_name} ({r.baseline_model})"
            display_name = display_name[:32]
            print(
                f"{display_name:<32} {ens_str:>10} {base_str:>10} {ratio_str:>8} {viol_flag:>6}"
            )

    decision = DEFAULT_DECISION_KEEP
    decision_reason = "ensemble within tolerance"

    lift_fraction = 0.0
    if effective_n:
        lift_threshold = 1.0 - min_lift_rmse_ratio
        lift_count = sum(
            1
            for r in results
            if (
                r.rmse_ratio is not None
                and isinstance(r.rmse_ratio, (int, float))
                and float(r.rmse_ratio) < float(lift_threshold)
            )
        )
        lift_fraction = lift_count / effective_n

    if rmse_inconclusive:
        required = warmup_required if warmup_required > 0 else 1
        decision = "INCONCLUSIVE"
        decision_reason = (
            f"effective_audits={effective_n} < required_audits={required}"
        )
    else:
        if holding_period > 0 and effective_n >= holding_period:
            print(
                f"\nEnsemble lift fraction: {lift_fraction:.2%} "
                f"(required >= {min_lift_fraction:.2%})"
            )
            if lift_fraction < min_lift_fraction:
                decision = DEFAULT_DECISION_DISABLE
                decision_reason = "insufficient lift vs baseline"
                if disable_if_no_lift:
                    raise SystemExit(
                        "Ensemble shows insufficient lift over baseline during holding period; "
                        "disable ensemble as default source of truth (reward-to-effort)."
                    )
                print(
                    "No-lift hard fail disabled by config; keeping ensemble in "
                    "non-default/research-only posture."
                )
            else:
                decision_reason = "lift demonstrated during holding period"

        if violation_rate > max_violation_rate:
            decision = DEFAULT_DECISION_RESEARCH
            decision_reason = (
                f"violation rate {violation_rate:.2%} exceeds {max_violation_rate:.2%}"
            )
            raise SystemExit(
                f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
                f"max-violation-rate {max_violation_rate:.2%}"
            )

        if promotion_margin > 0 and effective_n > 0 and decision == DEFAULT_DECISION_KEEP:
            margin_threshold = 1.0 - promotion_margin
            margin_lift = sum(
                1
                for r in results
                if r.rmse_ratio is not None
                and isinstance(r.rmse_ratio, (int, float))
                and float(r.rmse_ratio) < float(margin_threshold)
            )
            margin_lift_fraction = (margin_lift / effective_n) if effective_n else 0.0
            if margin_lift_fraction <= 0.0:
                decision = DEFAULT_DECISION_RESEARCH
                decision_reason = (
                    f"no ensemble lift >= {promotion_margin:.2%} across recent audits"
                )

    print(f"\nDecision: {decision} ({decision_reason})")

    cache_dir = Path("logs/forecast_audits_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ratios_filtered = [
        (r.path.name, float(r.rmse_ratio))
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    ratio_values = [val for _, val in ratios_filtered]
    ratio_stats = {
        "count": len(ratio_values),
        "min": min(ratio_values) if ratio_values else None,
        "max": max(ratio_values) if ratio_values else None,
        "mean": (sum(ratio_values) / len(ratio_values)) if ratio_values else None,
        "p10": pct.get(0.1) if pct else None,
        "p50": pct.get(0.5) if pct else None,
        "p90": pct.get(0.9) if pct else None,
        "best": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1])[:5]
        ],
        "worst": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1], reverse=True)[
                :5
            ]
        ],
    }
    results_by_path = {r.path: r for r in results}
    dataset_entries = []
    rmse_usable_entries = []
    for f in outcome_unique_files:
        audit = _load_audit(f)
        ds = (audit or {}).get("dataset") or {}
        signal_context = (audit or {}).get("signal_context") or {}
        if not isinstance(signal_context, dict):
            signal_context = {}
        raw_ts_signal_id = signal_context.get("ts_signal_id")
        ts_signal_id = str(raw_ts_signal_id).strip() if raw_ts_signal_id is not None else ""
        ts_signal_id = ts_signal_id or None
        signal_context_missing = bool(signal_context.get("signal_context_missing"))
        expected_close, expected_close_source = compute_expected_close(signal_context, ds)
        context_type = str(signal_context.get("context_type") or "").strip().upper()
        if not context_type:
            context_type = "TRADE"
        meta = _extract_window_metadata(audit or {})
        entry = {
            "file": f.name,
            "start": ds.get("start"),
            "end": ds.get("end"),
            "length": ds.get("length"),
            "forecast_horizon": ds.get("forecast_horizon"),
            "ticker": meta.get("ticker"),
            "detected_regime": meta.get("detected_regime"),
            "end_day": meta.get("end_day"),
            "context_type": context_type,
            "ts_signal_id": ts_signal_id,
            "run_id": signal_context.get("run_id"),
            "entry_ts": signal_context.get("entry_ts"),
            "signal_context_missing": signal_context_missing,
            "dataset_forecast_horizon": ds.get("forecast_horizon"),
            "signal_forecast_horizon": signal_context.get("forecast_horizon"),
            "expected_close_ts": expected_close.isoformat() if expected_close else None,
            "expected_close_source": expected_close_source,
            "outcome_status": None,
            "outcome_reason": None,
        }
        matching = results_by_path.get(f)
        if matching:
            entry["rmse_ratio"] = matching.rmse_ratio
            entry["ensemble_rmse"] = matching.ensemble_rmse
            entry["baseline_rmse"] = matching.baseline_rmse
            entry["ensemble_missing"] = bool(matching.ensemble_missing)
            if (
                matching.ensemble_rmse is not None
                and matching.baseline_rmse is not None
                and matching.baseline_rmse > 0
            ):
                rmse_usable_entries.append(entry)
        dataset_entries.append(entry)

    if db_path is not None:
        outcome_join_attempted = True
        closed_trade_counts: Dict[str, int] = {}
        outcomes_loaded, closed_trade_counts, outcome_join_error = _load_closed_trade_match_counts(
            db_path
        )
        if outcomes_loaded:
            now = now_utc()
            for entry in dataset_entries:
                context_type = str(entry.get("context_type") or "").strip().upper()
                if context_type and context_type != "TRADE":
                    entry["outcome_status"] = "NON_TRADE_CONTEXT"
                    entry["outcome_reason"] = "NON_TRADE_CONTEXT"
                    outcome_windows_non_trade_context += 1
                    continue

                ticker = str(entry.get("ticker") or "").strip().upper()
                if not ticker:
                    entry["outcome_status"] = "NON_TRADE_CONTEXT"
                    entry["outcome_reason"] = "MISSING_TICKER"
                    outcome_windows_non_trade_context += 1
                    continue

                run_id = str(entry.get("run_id") or "").strip()
                entry_ts_raw = str(entry.get("entry_ts") or "").strip()
                if not run_id or not entry_ts_raw:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "MISSING_EXECUTION_METADATA"
                    outcome_windows_missing_execution_metadata += 1
                    outcome_windows_invalid_context += 1
                    continue

                ts_signal_id = str(entry.get("ts_signal_id") or "").strip()
                if not ts_signal_id:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "MISSING_SIGNAL_ID"
                    outcome_windows_no_signal_id += 1
                    outcome_windows_invalid_context += 1
                    continue

                signal_horizon = _parse_non_negative_int(entry.get("signal_forecast_horizon"))
                dataset_horizon = _parse_non_negative_int(entry.get("dataset_forecast_horizon"))
                if (
                    signal_horizon is not None
                    and dataset_horizon is not None
                    and signal_horizon != dataset_horizon
                ):
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "HORIZON_MISMATCH"
                    outcome_windows_invalid_context += 1
                    continue

                expected_close_ts = _parse_utc_datetime(entry.get("expected_close_ts"))
                entry_ts = _parse_utc_datetime(entry.get("entry_ts"))
                if expected_close_ts is None:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "EXPECTED_CLOSE_UNAVAILABLE"
                    outcome_windows_invalid_context += 1
                    continue
                if entry_ts is not None and expected_close_ts < entry_ts:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "CAUSALITY_VIOLATION"
                    outcome_windows_invalid_context += 1
                    continue
                if (
                    (expected_close_ts + OUTCOME_ELIGIBILITY_BUFFER) > now
                ):
                    entry["outcome_status"] = "NOT_DUE"
                    entry["outcome_reason"] = "OUTCOME_WINDOW_OPEN"
                    outcome_windows_not_due += 1
                    outcome_windows_not_yet_eligible += 1
                    continue

                match_count = int(closed_trade_counts.get(ts_signal_id, 0))
                entry["outcome_match_count"] = match_count
                if match_count == 1:
                    entry["outcome_status"] = "MATCHED"
                    entry["outcome_reason"] = "ONE_TO_ONE_MATCH"
                    outcome_windows_eligible += 1
                    outcome_windows_matched += 1
                elif match_count == 0:
                    entry["outcome_status"] = "OUTCOME_MISSING"
                    entry["outcome_reason"] = "DUE_BUT_MISSING_CLOSE"
                    outcome_windows_eligible += 1
                    outcome_windows_missing += 1
                else:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "AMBIGUOUS_MATCH"
                    outcome_windows_invalid_context += 1
                    outcome_windows_ambiguous += 1
        else:
            for entry in dataset_entries:
                entry["outcome_status"] = "OUTCOMES_NOT_LOADED"
                entry["outcome_reason"] = "OUTCOME_JOIN_UNAVAILABLE"
                outcome_windows_outcomes_not_loaded += 1

    generated_utc = telemetry_now_utc()

    def _legacy_to_status(outcome_status: str) -> str:
        mapping = {
            "MATCHED": "PASS",
            "OUTCOME_MISSING": "FAIL",
            "NOT_DUE": "INCONCLUSIVE_ALLOWED",
            "INVALID_CONTEXT": "INVALID_CONTEXT",
            "NON_TRADE_CONTEXT": "WARN",
            "OUTCOMES_NOT_LOADED": "WARN",
        }
        return mapping.get(outcome_status, "WARN")

    for entry in dataset_entries:
        outcome_status = str(entry.get("outcome_status") or "OUTCOMES_NOT_LOADED").strip().upper()
        outcome_reason = str(entry.get("outcome_reason") or "OUTCOME_JOIN_UNAVAILABLE").strip().upper()
        context_type = str(entry.get("context_type") or "TRADE").strip().upper() or "TRADE"
        counts_toward_linkage = outcome_status in {"MATCHED", "OUTCOME_MISSING"}
        counts_toward_readiness = (
            context_type == "TRADE"
            and outcome_status not in {"INVALID_CONTEXT", "NON_TRADE_CONTEXT", "OUTCOMES_NOT_LOADED"}
        )
        severity = "LOW"
        blocking = False
        if outcome_status == "INVALID_CONTEXT":
            severity = "HIGH"
            blocking = True
        elif outcome_status == "OUTCOME_MISSING":
            severity = "MEDIUM"
        elif outcome_status == "OUTCOMES_NOT_LOADED":
            severity = "MEDIUM"
        normalized = normalize_telemetry_payload(
            {
                "status": _legacy_to_status(outcome_status),
                "reason_code": outcome_reason,
                "context_type": context_type,
                "severity": severity,
                "blocking": blocking,
                "counts_toward_readiness_denominator": counts_toward_readiness,
                "counts_toward_linkage_denominator": counts_toward_linkage,
                "generated_utc": generated_utc,
                "source_script": "scripts/check_forecast_audits.py",
            },
            source_script="scripts/check_forecast_audits.py",
            generated_utc=generated_utc,
        )
        entry.update(normalized)

    readiness_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_readiness_denominator"))
    )
    linkage_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_linkage_denominator"))
    )
    readiness_excluded_non_trade = sum(
        1 for entry in dataset_entries if str(entry.get("outcome_status") or "").upper() == "NON_TRADE_CONTEXT"
    )
    readiness_excluded_invalid = sum(
        1 for entry in dataset_entries if str(entry.get("outcome_status") or "").upper() == "INVALID_CONTEXT"
    )

    if (
        (outcome_windows_eligible > 0 or outcome_windows_matched > 0)
        and (not outcomes_loaded or not outcome_join_attempted)
    ):
        raise SystemExit(
            "Telemetry contract violation: outcome window counts > 0 without "
            "outcomes_loaded=true and outcome_join_attempted=true."
        )

    if outcome_join_attempted and outcome_join_error:
        print(f"[WARN] outcome_join_unavailable db={db_path} error={outcome_join_error}")
    print(
        "Outcome join   : "
        f"outcomes_loaded={int(outcomes_loaded)} "
        f"join_attempted={int(outcome_join_attempted)} "
        f"eligible={outcome_windows_eligible} "
        f"matched={outcome_windows_matched} "
        f"missing={outcome_windows_missing} "
        f"ambiguous={outcome_windows_ambiguous} "
        f"not_due={outcome_windows_not_due} "
        f"invalid_context={outcome_windows_invalid_context} "
        f"not_yet_eligible={outcome_windows_not_yet_eligible} "
        f"no_signal_id={outcome_windows_no_signal_id} "
        f"non_trade_context={outcome_windows_non_trade_context} "
        f"missing_exec_meta={outcome_windows_missing_execution_metadata}"
    )
    print(
        "Denominators   : "
        f"readiness_included={readiness_denominator_included} "
        f"linkage_included={linkage_denominator_included} "
        f"excluded_non_trade={readiness_excluded_non_trade} "
        f"excluded_invalid={readiness_excluded_invalid}"
    )

    healthy_tickers = {"NVDA", "MSFT", "GOOG", "JPM"}
    diversity = {
        "regime_count": len(
            {
                str(entry.get("detected_regime")).strip()
                for entry in rmse_usable_entries
                if str(entry.get("detected_regime") or "").strip()
            }
        ),
        "healthy_ticker_count": len(
            {
                str(entry.get("ticker")).strip().upper()
                for entry in rmse_usable_entries
                if str(entry.get("ticker") or "").strip().upper() in healthy_tickers
            }
        ),
        "distinct_trading_days": len(
            {
                str(entry.get("end_day")).strip()
                for entry in rmse_usable_entries
                if str(entry.get("end_day") or "").strip()
            }
        ),
    }
    print(
        "Diversity      : "
        f"regimes={diversity['regime_count']} "
        f"healthy_tickers={diversity['healthy_ticker_count']} "
        f"trading_days={diversity['distinct_trading_days']}"
    )

    summary = {
        "audit_dir": str(audit_dir),
        "audit_roots": [str(root) for root in audit_roots],
        "generated_utc": generated_utc,
        "source_script": "scripts/check_forecast_audits.py",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "effective_audits": effective_n,
        "effective_outcome_audits": outcome_windows_matched,
        "total_unique_audits": len(results),
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "max_violation_rate": max_violation_rate,
        "ensemble_missing_count": ensemble_missing_count,
        "ensemble_missing_rate": ensemble_missing_rate,
        "max_missing_ensemble_rate": max_missing_ensemble_rate,
        "min_forecast_horizon": min_forecast_horizon,
        "horizon_filtered_count": horizon_filtered_count,
        "manifest_integrity": manifest_stats,
        "holding_period_required": warmup_required,
        "lift_fraction": lift_fraction,
        "min_lift_fraction": min_lift_fraction,
        "percentiles": {
            "p10": pct.get(0.1) if pct else None,
            "p50": pct.get(0.5) if pct else None,
            "p90": pct.get(0.9) if pct else None,
        },
        "ratio_distribution": ratio_stats,
        "decision": decision,
        "decision_reason": decision_reason,
        "recent_window_audits": recent_window_audits,
        "recent_effective_audits": recent_effective_n,
        "recent_violation_count": recent_violation_count,
        "recent_violation_rate": recent_violation_rate,
        "recent_window_max_violation_rate": recent_window_max_violation_rate,
        "recent_rmse_ratio_p90": recent_pct.get(0.9) if recent_pct else None,
        "recent_window_max_p90_rmse_ratio": recent_window_max_p90_rmse_ratio,
        "outcome_join": {
            "db_path": str(db_path) if db_path is not None else None,
            "error": outcome_join_error,
            "eligibility_buffer_minutes": int(OUTCOME_ELIGIBILITY_BUFFER.total_seconds() // 60),
            "readiness_denominator_included": readiness_denominator_included,
            "linkage_denominator_included": linkage_denominator_included,
            "readiness_excluded_non_trade_context": readiness_excluded_non_trade,
            "readiness_excluded_invalid_context": readiness_excluded_invalid,
            "status_taxonomy": [
                "MATCHED",
                "OUTCOME_MISSING",
                "NOT_DUE",
                "INVALID_CONTEXT",
                "NON_TRADE_CONTEXT",
            ],
        },
        "telemetry_contract": {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "rmse_inputs_present": bool(results),
            "outcomes_loaded": outcomes_loaded,
            "execution_log_loaded": False,
            "outcome_join_attempted": outcome_join_attempted,
            "status": "PASS" if decision in {DEFAULT_DECISION_KEEP, DEFAULT_DECISION_RESEARCH} else decision,
            "reason_code": str(decision_reason).upper().replace(" ", "_"),
            "context_type": "TRADE",
            "severity": "LOW",
            "blocking": False,
            "counts_toward_readiness_denominator": True,
            "counts_toward_linkage_denominator": False,
            "generated_utc": generated_utc,
            "source_script": "scripts/check_forecast_audits.py",
        },
        "scope": {
            "include_research": bool(args.include_research),
            "production_audit_only": not bool(args.include_research),
        },
        "window_counts": {
            "n_raw_windows": len(files),
            "n_parseable_windows": parseable_count,
            "n_deduped_windows": len(unique_map),
            "n_outcome_deduped_windows": len(outcome_unique_map),
            "n_rmse_windows_processed": rmse_windows_processed,
            "n_rmse_windows_usable": rmse_windows_usable,
            "n_outcome_windows_eligible": outcome_windows_eligible,
            "n_outcome_windows_matched": outcome_windows_matched,
            "n_outcome_windows_missing": outcome_windows_missing,
            "n_outcome_windows_ambiguous": outcome_windows_ambiguous,
            "n_outcome_windows_not_due": outcome_windows_not_due,
            "n_outcome_windows_not_yet_eligible": outcome_windows_not_yet_eligible,
            "n_outcome_windows_invalid_context": outcome_windows_invalid_context,
            "n_outcome_windows_outcomes_not_loaded": outcome_windows_outcomes_not_loaded,
            "n_outcome_windows_no_signal_id": outcome_windows_no_signal_id,
            "n_outcome_windows_non_trade_context": outcome_windows_non_trade_context,
            "n_outcome_windows_missing_execution_metadata": outcome_windows_missing_execution_metadata,
            "n_readiness_denominator_included": readiness_denominator_included,
            "n_linkage_denominator_included": linkage_denominator_included,
            "n_readiness_excluded_non_trade_context": readiness_excluded_non_trade,
            "n_readiness_excluded_invalid_context": readiness_excluded_invalid,
        },
        "window_diversity": diversity,
        "dataset_windows": dataset_entries,
    }
    summary["telemetry_contract"] = normalize_telemetry_payload(
        summary.get("telemetry_contract", {}),
        source_script="scripts/check_forecast_audits.py",
        generated_utc=generated_utc,
    )
    cache_path = cache_dir / "latest_summary.json"
    dash_path = cache_dir / "ratio_distribution.json"
    cache_status = {"write_ok": True, "errors": []}
    try:
        _write_json_atomic(dash_path, ratio_stats)
    except Exception as exc:
        cache_status["write_ok"] = False
        cache_status["errors"].append(f"ratio_distribution:{exc}")
        print(f"[WARN] forecast_audits_cache_write_failed target=ratio_distribution error={exc}")
    summary["cache_status"] = cache_status
    try:
        _write_json_atomic(cache_path, summary)
    except Exception as exc:
        print(f"[WARN] forecast_audits_cache_write_failed target=latest_summary error={exc}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
