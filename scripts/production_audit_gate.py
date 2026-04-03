#!/usr/bin/env python3
"""
Production audit gate runner.

Combines:
1) Forecast lift gate (`scripts/check_forecast_audits.py`)
2) Profitability proof gate (`scripts/validate_profitability_proof.py`)

Outputs a machine-readable artifact for operators and batch wrappers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

try:
    from scripts.audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT
except Exception:  # pragma: no cover - script execution path fallback
    from audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT

try:
    from scripts.telemetry_adapter import normalize_telemetry_payload
except Exception:  # pragma: no cover - script execution path fallback
    from telemetry_adapter import normalize_telemetry_payload

from utils.evidence_io import write_promoted_json_artifact


# Phase 7.13-C1: central path constants
try:
    from etl.paths import DB_PATH as _DEFAULT_DB_PATH, FORECAST_AUDITS_DIR as _DEFAULT_AUDIT_DIR
except ImportError:
    _root = Path(__file__).resolve().parent.parent
    _DEFAULT_DB_PATH = _root / "data" / "portfolio_maximizer.db"
    _DEFAULT_AUDIT_DIR = _root / "logs" / "forecast_audits"

_LEGACY_FORECAST_AUDIT_ROOT = Path(_DEFAULT_AUDIT_DIR)
_PRODUCTION_AUDIT_DIR = _LEGACY_FORECAST_AUDIT_ROOT / "production"
if _PRODUCTION_AUDIT_DIR.exists():
    _DEFAULT_AUDIT_DIR = _PRODUCTION_AUDIT_DIR

PASS_SEMANTICS_VERSION = 3
DEFAULT_MAX_WARMUP_DAYS = 30


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    import logging as _logging_pag
    _pag_log = _logging_pag.getLogger(__name__)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        _pag_log.warning(
            "[PAG] Corrupted JSON artifact (parse failure): %s — %s. "
            "Gate will treat this as missing; run may produce incorrect PASS.",
            path,
            exc,
        )
        return None
    except Exception as exc:
        _pag_log.warning(
            "[PAG] Could not read artifact %s: %s",
            path,
            exc,
        )
        return None


def _validate_output_artifact(payload: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    required = [
        "timestamp_utc",
        "phase3_ready",
        "phase3_reason",
        "readiness",
        "production_profitability_gate",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"missing:{','.join(missing)}"
    return True, "ok"


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def _count_masked_unlinked_closes(db_path: Path) -> Tuple[int, List[int]]:
    """Count whitelisted IDs that are actually present as unlinked closes in the DB.

    This provides fail-open visibility: if the whitelist suppresses violations,
    operators can see exactly how many are masked vs genuinely resolved.
    Returns (masked_count, masked_ids_found_in_db).
    """
    _whitelist_raw = os.environ.get("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "")
    _whitelist_ids: set[int] = set()
    for _tok in _whitelist_raw.split(","):
        _tok = _tok.strip()
        if _tok.isdigit():
            _whitelist_ids.add(int(_tok))

    if not _whitelist_ids or not db_path.exists():
        return 0, []

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in _whitelist_ids)
        rows = conn.execute(
            f"SELECT id FROM trade_executions "
            f"WHERE is_close = 1 AND entry_trade_id IS NULL AND id IN ({placeholders})",
            tuple(sorted(_whitelist_ids)),
        ).fetchall()
        conn.close()
        found = [int(r["id"]) for r in rows]
        return len(found), found
    except Exception:
        return 0, []


def _count_unlinked_closes(db_path: Path, close_ids: Optional[list[int]] = None) -> Tuple[Optional[int], List[int], Optional[str]]:
    """
    Verify unlinked-close backlog directly from DB.

    Returns:
    - remaining_count (None on verification error)
    - sampled_close_ids (up to 20)
    - verification_error (None on success)
    """
    if not db_path.exists():
        return None, [], f"db_not_found:{db_path}"

    # POI-02 fix: apply the same INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS used by
    # PnLIntegrityEnforcer so that whitelisted resume-originated closes are excluded
    # from the unlinked-close count (prevents enforcer/gate divergence).
    _whitelist_raw = os.environ.get("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "")
    _whitelist_ids: set[int] = set()
    for _tok in _whitelist_raw.split(","):
        _tok = _tok.strip()
        if _tok.isdigit():
            _whitelist_ids.add(int(_tok))

    where = [
        "is_close = 1",
        "entry_trade_id IS NULL",
        # Note: intentionally no realized_pnl IS NOT NULL filter -- matches the scope
        # used by PnLIntegrityEnforcer.CLOSE_WITHOUT_ENTRY_LINK (pnl_integrity_enforcer.py:552)
        # so that reconcile PASS ↔ zero integrity violations, not just zero PnL-carrying ones.
    ]
    params: list[Any] = []

    # Exclude whitelisted resume-originated closes (same whitelist as enforcer).
    if _whitelist_ids:
        placeholders_wl = ",".join("?" for _ in _whitelist_ids)
        where.append(f"id NOT IN ({placeholders_wl})")
        params.extend(sorted(_whitelist_ids))

    filtered_ids = [int(x) for x in (close_ids or []) if int(x) > 0]
    if filtered_ids:
        placeholders = ",".join("?" for _ in filtered_ids)
        where.append(f"id IN ({placeholders})")
        params.extend(filtered_ids)

    sql_where = " AND ".join(where)
    count_sql = f"SELECT COUNT(*) AS n FROM trade_executions WHERE {sql_where}"
    sample_sql = f"SELECT id FROM trade_executions WHERE {sql_where} ORDER BY id LIMIT 20"

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(count_sql, tuple(params)).fetchone()
        count_val = int(row["n"]) if row and row["n"] is not None else 0
        sample_rows = conn.execute(sample_sql, tuple(params)).fetchall()
        sample_ids = [int(r["id"]) for r in sample_rows if r and r["id"] is not None]
        conn.close()
        return count_val, sample_ids, None
    except Exception as exc:
        return None, [], str(exc)


def _run_reconcile_step(
    *,
    python_bin: str,
    repo_root: Path,
    db_path: Path,
    close_ids: list[int],
    apply: bool,
) -> Dict[str, Any]:
    """Run close-link reconciliation as an optional pre-gate step."""
    repair_script = repo_root / "scripts" / "repair_unlinked_closes.py"
    cmd = [python_bin, str(repair_script), "--db", str(db_path)]
    if close_ids:
        cmd.extend(["--close-ids", *[str(int(x)) for x in close_ids if int(x) > 0]])
    if apply:
        cmd.append("--apply")
    proc = _run_command(cmd, cwd=repo_root)
    output = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
    result: Dict[str, Any] = {
        "requested": True,
        "apply": bool(apply),
        "close_ids": [int(x) for x in close_ids if int(x) > 0],
        "exit_code": int(proc.returncode),
        "status": "PASS" if int(proc.returncode) == 0 else "FAIL",
        "output_tail": _tail_lines(output),
    }
    if int(proc.returncode) != 0:
        result["status_reason"] = "reconcile_command_failed"
    else:
        result["status_reason"] = "reconcile_command_ok"

    remaining_count, remaining_ids, verification_error = _count_unlinked_closes(
        db_path,
        close_ids=[int(x) for x in close_ids if int(x) > 0],
    )
    result["remaining_unlinked_closes"] = remaining_count
    result["remaining_unlinked_close_ids"] = remaining_ids
    result["verification_error"] = verification_error

    if verification_error:
        result["status"] = "FAIL"
        result["status_reason"] = "reconcile_verification_failed"
    elif remaining_count is not None and int(remaining_count) > 0:
        # Strict semantics: do not report PASS when targeted closes remain unlinked.
        result["status"] = "FAIL"
        result["status_reason"] = (
            "remaining_unlinked_after_apply" if apply else "remaining_unlinked_detected"
        )
    elif int(proc.returncode) == 0:
        result["status"] = "PASS"
        result["status_reason"] = (
            "verified_zero_unlinked_after_apply" if apply else "verified_zero_unlinked"
        )
    else:
        result["status"] = "FAIL"
        result["status_reason"] = "reconcile_command_failed"

    return result


def _run_command_quiet(cmd: list[str], cwd: Path) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        return int(proc.returncode), proc.stdout or "", proc.stderr or ""
    except Exception as exc:
        return 127, "", str(exc)


def _sha256_file(path: Path, *, max_bytes: int = 5 * 1024 * 1024) -> Tuple[Optional[str], Optional[str]]:
    """Return (sha256, skip_reason). Never raises."""
    try:
        size = int(path.stat().st_size)
    except Exception:
        return None, "stat_failed"

    if max_bytes > 0 and size > max_bytes:
        return None, f"too_large>{max_bytes}"

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            remaining = size
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest(), None
    except Exception:
        return None, "read_failed"


def _looks_like_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith(".env") or name.endswith(".env") or name == ".env":
        return True
    markers = ("secret", "token", "password", "apikey", "api_key", "credential", "private")
    if any(m in name for m in markers):
        return True
    if path.suffix.lower() in {".key", ".pem", ".p12", ".pfx", ".crt", ".cer", ".der"}:
        return True
    return False


def _parse_git_status_porcelain(text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        if line.startswith("?? "):
            path_raw = line[3:].strip()
            entries.append(
                {
                    "kind": "untracked",
                    "code": "??",
                    "path": path_raw,
                    "path_orig": None,
                }
            )
            continue

        code = line[:2]
        path_raw = line[3:].strip() if len(line) > 3 else ""
        path_orig = None
        path = path_raw
        if "->" in path_raw:
            left, right = path_raw.split("->", 1)
            path_orig = left.strip() or None
            path = right.strip()
        entries.append(
            {
                "kind": "tracked",
                "code": code,
                "path": path,
                "path_orig": path_orig,
            }
        )
    return entries


def _collect_git_state(repo_root: Path) -> Dict[str, Any]:
    """Capture current repo + worktree state (paths only; no contents)."""
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return {"available": False, "reason": "no .git directory"}

    def _git(args: list[str]) -> Tuple[int, str, str]:
        return _run_command_quiet(["git", *args], cwd=repo_root)

    def _git1(args: list[str]) -> Optional[str]:
        rc, out, _ = _git(args)
        if rc != 0:
            return None
        return (out or "").strip() or None

    branch = _git1(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git1(["rev-parse", "HEAD"])
    upstream = _git1(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])

    ahead = behind = None
    if upstream:
        rc, out, _ = _git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
        if rc == 0:
            parts = (out or "").strip().split()
            if len(parts) >= 2:
                try:
                    ahead = int(parts[0])
                    behind = int(parts[1])
                except Exception:
                    ahead = behind = None

    rc, out, err = _git(["status", "--porcelain"])
    status_text = (out or "") if rc == 0 else ""
    entries = _parse_git_status_porcelain(status_text)

    tracked = [e for e in entries if e.get("kind") == "tracked"]
    untracked = [e for e in entries if e.get("kind") == "untracked"]

    staged = 0
    unstaged = 0
    for e in tracked:
        code = str(e.get("code") or "  ")
        if len(code) >= 1 and code[0] not in {" ", "?"}:
            staged += 1
        if len(code) >= 2 and code[1] != " ":
            unstaged += 1

    file_meta: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for e in entries:
        rel = str(e.get("path") or "").strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)

        meta: Dict[str, Any] = {"path": rel}
        p = (repo_root / rel)
        try:
            stat = p.stat()
        except Exception:
            meta.update({"exists": False})
            file_meta.append(meta)
            continue

        meta.update(
            {
                "exists": True,
                "bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                # NOTE: On Windows this is creation time; on POSIX it's metadata-change time.
                "ctime_utc": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            }
        )

        if e.get("kind") != "tracked":
            # Untracked files may include personal notes or secrets; do not hash by default.
            meta.update(
                {
                    "sha256": None,
                    "sha256_skipped": True,
                    "sha256_skip_reason": "untracked_not_hashed",
                }
            )
        elif _looks_like_secret_path(p):
            meta.update(
                {
                    "sha256": None,
                    "sha256_skipped": True,
                    "sha256_skip_reason": "possible_secret_path",
                }
            )
        else:
            sha, skip_reason = _sha256_file(p)
            meta.update(
                {
                    "sha256": sha,
                    "sha256_skipped": sha is None,
                    "sha256_skip_reason": skip_reason,
                }
            )

        if e.get("kind") == "tracked":
            last = _git1(["log", "-1", "--format=%H|%cI", "--", rel])
            if last and "|" in last:
                commit, committed_at = last.split("|", 1)
                meta.update({"last_commit": commit or None, "last_committed_at": committed_at or None})
            else:
                meta.update({"last_commit": None, "last_committed_at": None})

        file_meta.append(meta)

    return {
        "available": True,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "status": {
            "tracked_changed": len(tracked),
            "untracked": len(untracked),
            "staged": staged,
            "unstaged": unstaged,
            "entries": entries,
        },
        "files": file_meta,
        "attribution_note": (
            "This captures current git state only. To attribute changes to a session, capture a baseline "
            "at session start and compare later (git alone cannot prove who/what changed files)."
        ),
    }


def _tail_lines(text: str, *, limit: int = 40) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-limit:])


def _load_profitability_requirements(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw.get("profitability_proof_requirements") or {}
    except Exception:
        return {}


def _load_regression_monitoring_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fm = raw.get("forecaster_monitoring") if isinstance(raw, dict) else {}
        rmse = fm.get("regression_metrics") if isinstance(fm, dict) else {}
        return rmse if isinstance(rmse, dict) else {}
    except Exception:
        return {}


def _collect_thresholds(
    *,
    monitor_config: Path,
    proof_requirements: Dict[str, Any],
    lift_inconclusive_allowed: bool,
    proof_profitable_required: bool,
    require_holding_period: bool,
    warmup_policy: Dict[str, Any],
) -> Dict[str, Any]:
    rmse_cfg = _load_regression_monitoring_config(monitor_config)
    stat_req = (proof_requirements.get("statistical_significance") or {}) if isinstance(proof_requirements, dict) else {}
    perf_req = (proof_requirements.get("performance") or {}) if isinstance(proof_requirements, dict) else {}
    audit_req = (proof_requirements.get("audit_trail") or {}) if isinstance(proof_requirements, dict) else {}

    return {
        "lift": {
            "min_lift_fraction": rmse_cfg.get("min_lift_fraction"),
            "max_violation_rate": rmse_cfg.get("max_violation_rate"),
            "recent_window_max_violation_rate": rmse_cfg.get("recent_window_max_violation_rate"),
            "max_missing_ensemble_rate": rmse_cfg.get("max_missing_ensemble_rate"),
            "holding_period_audits": rmse_cfg.get("holding_period_audits"),
            "promotion_margin": rmse_cfg.get("promotion_margin"),
            "max_warmup_days": rmse_cfg.get("max_warmup_days", DEFAULT_MAX_WARMUP_DAYS),
            "require_holding_period": bool(require_holding_period),
        },
        "proof": {
            "min_closed_trades": stat_req.get("min_closed_trades"),
            "min_trading_days": stat_req.get("min_trading_days"),
            "max_win_rate": stat_req.get("max_win_rate"),
            "min_win_rate": stat_req.get("min_win_rate"),
            "min_profit_factor": perf_req.get("min_profit_factor"),
            "max_drawdown": perf_req.get("max_drawdown"),
            "min_sharpe_ratio": perf_req.get("min_sharpe_ratio"),
            "require_entry_exit_matching": audit_req.get("require_entry_exit_matching"),
        },
        "semantics": {
            "lift_inconclusive_allowed": bool(lift_inconclusive_allowed),
            "proof_profitable_required": bool(proof_profitable_required),
            "warmup_expired": bool(warmup_policy.get("warmup_expired", True)),
        },
    }


def _first_audit_ts_utc(audit_dir: Path) -> Optional[datetime]:
    try:
        files = list(audit_dir.glob("forecast_audit_*.json"))
    except Exception:
        return None
    if not files:
        return None
    try:
        oldest = min(files, key=lambda p: p.stat().st_mtime)
        return datetime.fromtimestamp(oldest.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _compute_warmup_window(
    *,
    audit_dir: Path,
    monitor_config: Path,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_utc = now or datetime.now(timezone.utc)
    rmse_cfg = _load_regression_monitoring_config(monitor_config)
    raw_days = rmse_cfg.get("max_warmup_days", DEFAULT_MAX_WARMUP_DAYS)
    try:
        max_warmup_days = max(int(raw_days), 0)
    except Exception:
        max_warmup_days = DEFAULT_MAX_WARMUP_DAYS
    first_audit_ts = _first_audit_ts_utc(audit_dir)
    allow_until: Optional[datetime] = None
    warmup_expired = True
    if first_audit_ts is not None:
        allow_until = first_audit_ts + timedelta(days=max_warmup_days)
        warmup_expired = now_utc >= allow_until
    return {
        "max_warmup_days": max_warmup_days,
        "first_audit_ts_utc": first_audit_ts.isoformat() if first_audit_ts else None,
        "allow_inconclusive_until_utc": allow_until.isoformat() if allow_until else None,
        "warmup_expired": bool(warmup_expired),
    }


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _parse_timestamp_utc(raw: Any) -> Optional[datetime]:
    """Best-effort ISO/SQLite timestamp parser returning timezone-aware UTC."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        # Handle trailing Z for strict UTC ISO strings.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except Exception:
        # SQLite defaults (YYYY-MM-DD HH:MM:SS) without timezone.
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except Exception:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_latest_live_cycle_binding(db_path: Path) -> Dict[str, Any]:
    """Load latest live-cycle provenance from trade_executions (best-effort)."""
    result: Dict[str, Any] = {
        "available": False,
        "latest_live_cycle_ts_utc": None,
        "latest_live_run_id": None,
        "latest_live_trade_id": None,
        "query_error": None,
    }
    if not db_path.exists():
        result["query_error"] = f"db_not_found:{db_path}"
        return result

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        cols = {
            str(row["name"]).strip()
            for row in conn.execute("PRAGMA table_info(trade_executions)").fetchall()
            if row and row["name"] is not None
        }
        if not cols:
            result["query_error"] = "trade_executions_missing_or_unreadable"
            return result

        ts_candidates = [c for c in ("created_at", "bar_timestamp", "trade_date") if c in cols]
        if not ts_candidates:
            result["query_error"] = "trade_executions_missing_time_columns"
            return result
        ts_expr = "COALESCE(" + ", ".join(ts_candidates) + ")"

        run_expr = "run_id" if "run_id" in cols else "NULL"
        id_expr = "id" if "id" in cols else "NULL"

        where = []
        params: List[Any] = []
        if "execution_mode" in cols:
            where.append("LOWER(COALESCE(execution_mode, '')) = 'live'")
        if "is_synthetic" in cols:
            where.append("is_synthetic = 0")
        if "is_diagnostic" in cols:
            where.append("is_diagnostic = 0")
        if "is_contaminated" in cols:
            where.append("is_contaminated = 0")
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        row = conn.execute(
            f"""
            SELECT
                {id_expr} AS trade_id,
                {run_expr} AS run_id,
                {ts_expr} AS cycle_ts
            FROM trade_executions
            {where_sql}
            ORDER BY cycle_ts DESC, trade_id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        if not row:
            result["query_error"] = "no_live_cycles_found"
            return result

        parsed_ts = _parse_timestamp_utc(row["cycle_ts"])
        result["available"] = True
        result["latest_live_cycle_ts_utc"] = parsed_ts.isoformat() if parsed_ts else None
        result["latest_live_run_id"] = str(row["run_id"]).strip() if row["run_id"] is not None else None
        if result["latest_live_run_id"] == "":
            result["latest_live_run_id"] = None
        result["latest_live_trade_id"] = _safe_int(row["trade_id"], 0) if row["trade_id"] is not None else None
        return result
    except Exception as exc:
        result["query_error"] = str(exc)
        return result
    finally:
        if conn is not None:
            conn.close()


def _build_linkage_waterfall(window_counts: Dict[str, Any], *, production_audit_only: bool) -> Dict[str, Any]:
    """Build denominator waterfall counts for readiness diagnostics."""
    raw_candidates = _safe_int(
        window_counts.get("n_outcome_deduped_windows"),
        _safe_int(window_counts.get("n_deduped_windows"), 0),
    )
    non_trade = _safe_int(window_counts.get("n_outcome_windows_non_trade_context"), 0)
    invalid_context = _safe_int(window_counts.get("n_outcome_windows_invalid_context"), 0)
    linked = _safe_int(window_counts.get("n_outcome_windows_eligible"), 0)
    matched = _safe_int(window_counts.get("n_outcome_windows_matched"), 0)
    readiness_included = _safe_int(window_counts.get("n_readiness_denominator_included"), 0)

    production_only_count = max(0, raw_candidates - non_trade) if production_audit_only else raw_candidates
    hygiene_pass_count = max(0, readiness_included)

    return {
        "raw_candidates": raw_candidates,
        "production_only": production_only_count,
        "linked": linked,
        "hygiene_pass": hygiene_pass_count,
        "matched": matched,
        "excluded_non_trade_context": non_trade,
        "excluded_invalid_context": invalid_context,
        "matched_over_linked": _safe_ratio(matched, linked),
        "linked_over_production_only": _safe_ratio(linked, production_only_count),
        "hygiene_over_production_only": _safe_ratio(hygiene_pass_count, production_only_count),
    }


def _evaluate_artifact_binding(
    *,
    lift_summary: Dict[str, Any],
    live_cycle_binding: Dict[str, Any],
    repo_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate freshness + run/commit binding for production evidence."""
    summary_generated_raw = (
        lift_summary.get("generated_utc")
        if isinstance(lift_summary, dict)
        else None
    )
    summary_generated_ts = _parse_timestamp_utc(summary_generated_raw)
    latest_live_raw = live_cycle_binding.get("latest_live_cycle_ts_utc")
    latest_live_ts = _parse_timestamp_utc(latest_live_raw)
    latest_live_run_id = (
        str(live_cycle_binding.get("latest_live_run_id")).strip()
        if live_cycle_binding.get("latest_live_run_id") is not None
        else None
    )
    if latest_live_run_id == "":
        latest_live_run_id = None

    repo_head = (
        str(repo_state.get("head")).strip()
        if isinstance(repo_state, dict) and repo_state.get("head") is not None
        else None
    )
    if repo_head == "":
        repo_head = None

    freshness_pass = False
    reasons: List[str] = []
    if latest_live_ts is None:
        reasons.append("NO_LIVE_CYCLE_TIMESTAMP")
    if summary_generated_ts is None:
        reasons.append("MISSING_SUMMARY_GENERATED_TS")
    if latest_live_ts is not None and summary_generated_ts is not None:
        if summary_generated_ts >= latest_live_ts:
            freshness_pass = True
        else:
            reasons.append("SUMMARY_STALE_BEFORE_LIVE_CYCLE")
    if not latest_live_run_id:
        reasons.append("MISSING_LIVE_RUN_ID")
    if not repo_head:
        reasons.append("MISSING_REPO_COMMIT_HASH")

    binding_pass = bool(freshness_pass and latest_live_run_id and repo_head)
    return {
        "pass": binding_pass,
        "reason_codes": reasons,
        "summary_generated_utc": summary_generated_ts.isoformat() if summary_generated_ts else None,
        "latest_live_cycle_ts_utc": latest_live_ts.isoformat() if latest_live_ts else None,
        "latest_live_run_id": latest_live_run_id,
        "repo_head": repo_head,
        "freshness_pass": bool(freshness_pass),
        "run_id_present": bool(latest_live_run_id),
        "commit_hash_present": bool(repo_head),
    }


def _compute_lifecycle_integrity(
    db_path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "close_before_entry_count": 0,
        "closed_missing_exit_reason_count": 0,
        "query_error": None,
    }
    if not db_path.exists():
        result["query_error"] = f"db_not_found:{db_path}"
        return result

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        close_before_entry = 0
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades c
                LEFT JOIN trade_executions e ON c.entry_trade_id = e.id
                WHERE c.entry_trade_id IS NOT NULL
                  AND c.bar_timestamp IS NOT NULL
                  AND e.bar_timestamp IS NOT NULL
                  AND c.bar_timestamp < e.bar_timestamp
                  AND COALESCE(c.ts_signal_id, '') NOT LIKE 'legacy_%'
                  AND COALESCE(e.ts_signal_id, '') NOT LIKE 'legacy_%'
                """
            ).fetchone()
            close_before_entry = _safe_int(row["n"] if row else 0)
        except sqlite3.OperationalError:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades c
                LEFT JOIN trade_executions e ON c.entry_trade_id = e.id
                WHERE c.entry_trade_id IS NOT NULL
                  AND c.trade_date IS NOT NULL
                  AND e.trade_date IS NOT NULL
                  AND c.trade_date < e.trade_date
                  AND COALESCE(c.ts_signal_id, '') NOT LIKE 'legacy_%'
                  AND COALESCE(e.ts_signal_id, '') NOT LIKE 'legacy_%'
                """
            ).fetchone()
            close_before_entry = _safe_int(row["n"] if row else 0)

        closed_missing_exit_reason = 0
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades
                WHERE COALESCE(TRIM(exit_reason), '') = ''
                """
            ).fetchone()
            closed_missing_exit_reason = _safe_int(row["n"] if row else 0)
        except sqlite3.OperationalError:
            closed_missing_exit_reason = 0

        result["close_before_entry_count"] = close_before_entry
        result["closed_missing_exit_reason_count"] = closed_missing_exit_reason
        return result
    except Exception as exc:
        result["query_error"] = str(exc)
        return result
    finally:
        if conn is not None:
            conn.close()


def _build_evidence_progress(
    *,
    metrics: Dict[str, Any],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    stat_req = (requirements.get("statistical_significance") or {}) if isinstance(requirements, dict) else {}
    min_closed = int(stat_req.get("min_closed_trades", 0) or 0)
    min_days = int(stat_req.get("min_trading_days", 0) or 0)

    closed_trades = int(metrics.get("closed_trades", 0) or 0)
    if closed_trades <= 0:
        winning = int(metrics.get("winning_trades", 0) or 0)
        losing = int(metrics.get("losing_trades", 0) or 0)
        closed_trades = winning + losing
    trading_days = int(metrics.get("trading_days", 0) or 0)

    remaining_closed = max(0, min_closed - closed_trades)
    remaining_days = max(0, min_days - trading_days)

    return {
        "ready": (remaining_closed == 0 and remaining_days == 0),
        "closed_trades": closed_trades,
        "min_closed_trades": min_closed,
        "remaining_closed_trades": remaining_closed,
        "trading_days": trading_days,
        "min_trading_days": min_days,
        "remaining_trading_days": remaining_days,
    }


def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _summary_matches_audit_dir(summary: Dict[str, Any], audit_dir: Path) -> bool:
    raw = summary.get("audit_dir")
    if not raw:
        return False
    try:
        summary_dir = Path(str(raw)).resolve()
    except Exception:
        return False
    return summary_dir == audit_dir.resolve()


def _summary_matches_invocation(
    summary: Dict[str, Any],
    *,
    audit_dir: Path,
    max_files: int,
    include_research: bool,
) -> bool:
    if not _summary_matches_audit_dir(summary, audit_dir):
        return False
    raw_max_files = summary.get("max_files")
    if raw_max_files is not None:
        try:
            if int(raw_max_files) != int(max_files):
                return False
        except Exception:
            return False

    scope = summary.get("scope", {})
    if isinstance(scope, dict):
        raw_include_research = scope.get("include_research")
        if raw_include_research is not None and bool(raw_include_research) != bool(include_research):
            return False

    return True


def _fail_on_unknown_args(unknown: List[str]) -> None:
    if not unknown:
        return
    msg = "Unknown args are not allowed (fail-closed to prevent wiring drift): " + " ".join(unknown)
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _safe_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _summary_scope_flag(summary: Dict[str, Any], name: str) -> Optional[bool]:
    scope = summary.get("scope", {})
    if not isinstance(scope, dict):
        return None
    raw = scope.get(name)
    if raw is None:
        return None
    return _safe_bool(raw)


def _binding_safe_lift_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Retain only provenance/binding fields when lift command fails."""
    if not isinstance(summary, dict):
        return {}
    keep = {
        "audit_dir",
        "audit_roots",
        "generated_utc",
        "source_script",
        "schema_version",
        "max_files",
        "scope",
        "window_counts",
        "status",
        "exit_code",
        "exit_reason",
    }
    return {k: summary.get(k) for k in keep if k in summary}


def _missing_summary_metric_keys(summary: Dict[str, Any]) -> List[str]:
    required_top_level = (
        "measurement_contract_version",
        "baseline_model",
        "lift_threshold_rmse_ratio",
        "effective_audits",
        "violation_rate",
        "max_violation_rate",
        "lift_fraction",
        "min_lift_fraction",
        "decision",
        "decision_reason",
        "window_counts",
    )
    missing = [key for key in required_top_level if key not in summary]
    window_counts = summary.get("window_counts")
    if "window_counts" not in missing and not isinstance(window_counts, dict):
        missing.append("window_counts")
        window_counts = None
    required_window_counts = (
        "n_rmse_windows_processed",
        "n_rmse_windows_usable",
        "n_outcome_windows_not_due",
        "n_readiness_denominator_included",
    )
    if isinstance(window_counts, dict):
        missing.extend(
            f"window_counts.{key}"
            for key in required_window_counts
            if key not in window_counts
        )
    return missing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Run production lift + profitability proof gates.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to run gate subprocesses (default: current interpreter).",
    )
    parser.add_argument(
        "--db",
        default=str(_DEFAULT_DB_PATH),
        help="Path to SQLite database (default: etl.paths.DB_PATH or data/portfolio_maximizer.db).",
    )
    parser.add_argument(
        "--proof-requirements",
        default="config/profitability_proof_requirements.yml",
        help="Profitability proof requirements config (default: config/profitability_proof_requirements.yml).",
    )
    parser.add_argument(
        "--audit-dir",
        default=str(_DEFAULT_AUDIT_DIR),
        help="Forecast audit directory (default: etl.paths.FORECAST_AUDITS_DIR).",
    )
    parser.add_argument(
        "--monitor-config",
        default="config/forecaster_monitoring.yml",
        help="Forecaster monitoring config path (default: config/forecaster_monitoring.yml).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=FORECAST_AUDIT_MAX_FILES_DEFAULT,
        help=f"Max forecast audit files to scan (default: {FORECAST_AUDIT_MAX_FILES_DEFAULT}).",
    )
    parser.add_argument(
        "--require-holding-period",
        action="store_true",
        help="Require holding-period completeness for the lift gate.",
    )
    parser.add_argument(
        "--allow-inconclusive-lift",
        action="store_true",
        help="Treat inconclusive lift checks as pass (default: fail).",
    )
    parser.add_argument(
        "--unattended-profile",
        action="store_true",
        help=(
            "Enable strict unattended semantics: INCONCLUSIVE lift allowed only before warmup expiry "
            "and profitability proof must be profitable."
        ),
    )
    parser.add_argument(
        "--require-profitable",
        action="store_true",
        help="Require strictly positive PnL (in addition to proof validity).",
    )
    parser.add_argument(
        "--include-research",
        action="store_true",
        help=(
            "Include research audit artifacts in lift checks (manual diagnostics only). "
            "Default behavior inspects production audits only."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="logs/audit_gate/production_gate_latest.json",
        help="Output path for latest gate artifact.",
    )
    parser.add_argument(
        "--reconcile",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional close IDs to reconcile via scripts/repair_unlinked_closes.py before gates run. "
            "If provided without IDs, scans all unlinked closes."
        ),
    )
    parser.add_argument(
        "--reconcile-apply",
        action="store_true",
        help="Apply reconciliation changes (default: dry-run reconciliation).",
    )
    parser.add_argument(
        "--notify-openclaw",
        action="store_true",
        help="Send gate summary via OpenClaw CLI (requires OPENCLAW_TARGETS/OPENCLAW_TO or --openclaw-to).",
    )
    parser.add_argument(
        "--openclaw-command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help='OpenClaw command (default: "openclaw"). Use "wsl openclaw" on Windows if needed.',
    )
    parser.add_argument(
        "--openclaw-to",
        default=os.getenv("OPENCLAW_TARGETS") or os.getenv("OPENCLAW_TO", ""),
        help=(
            "OpenClaw target(s). Supports a single target or a comma-separated list. "
            'Items may be "channel:target" (e.g. "whatsapp:+1555..., telegram:@mychat"). '
            "Can also be set via OPENCLAW_TARGETS or OPENCLAW_TO."
        ),
    )
    parser.add_argument(
        "--openclaw-timeout-seconds",
        type=float,
        default=20.0,
        help="OpenClaw command timeout in seconds (default: 20).",
    )
    args, unknown = parser.parse_known_args()
    _fail_on_unknown_args(unknown)
    python_bin = str(Path(args.python_bin))
    db_path = _resolve_path(repo_root, args.db)
    proof_requirements_path = _resolve_path(repo_root, args.proof_requirements)
    audit_dir = _resolve_path(repo_root, args.audit_dir)
    monitor_config = _resolve_path(repo_root, args.monitor_config)
    output_path = _resolve_path(repo_root, args.output_json)

    check_script = repo_root / "scripts" / "check_forecast_audits.py"
    proof_script = repo_root / "scripts" / "validate_profitability_proof.py"
    summary_cache_path = repo_root / "logs" / "forecast_audits_cache" / "latest_summary.json"

    warmup_policy = _compute_warmup_window(audit_dir=audit_dir, monitor_config=monitor_config)
    lift_inconclusive_allowed = bool(args.allow_inconclusive_lift)
    # Auto-allow INCONCLUSIVE during active warmup window — consistent with the
    # max_warmup_days provision in forecaster_monitoring.yml and the --unattended-profile
    # flag which already does the same. The warmup window exists specifically to absorb
    # INCONCLUSIVE lift states before enough audits accumulate (holding_period=20).
    # After warmup expires, require explicit --allow-inconclusive-lift to pass.
    if not lift_inconclusive_allowed:
        lift_inconclusive_allowed = not bool(warmup_policy.get("warmup_expired", True))
    proof_profitable_required = bool(args.require_profitable or args.unattended_profile)

    reconcile_result: Dict[str, Any] = {"requested": False}
    if args.reconcile is not None:
        reconcile_ids = [int(x) for x in (args.reconcile or []) if int(x) > 0]
        reconcile_result = _run_reconcile_step(
            python_bin=python_bin,
            repo_root=repo_root,
            db_path=db_path,
            close_ids=reconcile_ids,
            apply=bool(args.reconcile_apply),
        )

    lift_cmd = [
        python_bin,
        str(check_script),
        "--audit-dir",
        str(audit_dir),
        "--db",
        str(db_path),
        "--config-path",
        str(monitor_config),
        "--max-files",
        str(args.max_files),
    ]
    if args.require_holding_period:
        lift_cmd.append("--require-holding-period")
    if args.include_research:
        lift_cmd.append("--include-research")

    lift_proc = _run_command(lift_cmd, cwd=repo_root)
    lift_output = f"{lift_proc.stdout or ''}\n{lift_proc.stderr or ''}".strip()

    lift_summary = _safe_load_json(summary_cache_path) or {}
    summary_invocation_match = True
    summary_mismatch_details: Dict[str, Any] = {}
    if lift_summary:
        summary_invocation_match = _summary_matches_invocation(
            lift_summary,
            audit_dir=audit_dir,
            max_files=int(args.max_files),
            include_research=bool(args.include_research),
        )
        if not summary_invocation_match:
            summary_mismatch_details = {
                "audit_dir": str(lift_summary.get("audit_dir")),
                "max_files": lift_summary.get("max_files"),
                "scope_include_research": _summary_scope_flag(lift_summary, "include_research"),
                "expected_audit_dir": str(audit_dir),
                "expected_max_files": int(args.max_files),
                "expected_include_research": bool(args.include_research),
            }
            lift_summary = {}

    summary_metrics_error: Optional[str] = None
    missing_summary_metric_keys: List[str] = []
    if not lift_summary:
        summary_metrics_error = "SUMMARY_MISSING"
        if not summary_invocation_match:
            summary_metrics_error = "SUMMARY_INVOCATION_MISMATCH"
    else:
        missing_summary_metric_keys = _missing_summary_metric_keys(lift_summary)
        if missing_summary_metric_keys:
            summary_metrics_error = "SUMMARY_METRICS_MISSING"

    lift_decision = lift_summary.get("decision")
    lift_decision_reason = lift_summary.get("decision_reason")
    lift_inconclusive = str(lift_decision or "").strip().upper() == "INCONCLUSIVE"

    lift_status = "PASS"
    if summary_metrics_error:
        lift_status = "FAIL"
    elif lift_proc.returncode != 0:
        lift_status = "FAIL"
    elif lift_inconclusive:
        lift_status = "INCONCLUSIVE"

    lift_pass = summary_metrics_error is None and lift_proc.returncode == 0 and (
        lift_inconclusive_allowed or not lift_inconclusive
    )

    if summary_metrics_error == "SUMMARY_INVOCATION_MISMATCH":
        lift_decision = None
        lift_decision_reason = "SUMMARY_INVOCATION_MISMATCH"
    elif summary_metrics_error == "SUMMARY_METRICS_MISSING":
        lift_decision = None
        lift_decision_reason = (
            "SUMMARY_METRICS_MISSING:" + ",".join(missing_summary_metric_keys)
        )
    elif summary_metrics_error == "SUMMARY_MISSING":
        lift_decision = None
        lift_decision_reason = "SUMMARY_MISSING"
    elif lift_inconclusive:
        lift_decision = None
        lift_decision_reason = "insufficient effective audits for RMSE gate"

    lift_effective_audits = lift_summary.get("effective_audits")
    lift_violation_rate = lift_summary.get("violation_rate")
    lift_max_violation_rate = lift_summary.get("max_violation_rate")
    lift_fraction = lift_summary.get("lift_fraction")
    min_lift_fraction = lift_summary.get("min_lift_fraction")
    lift_measurement_contract_version = lift_summary.get("measurement_contract_version")
    lift_baseline_model = lift_summary.get("baseline_model")
    lift_threshold_rmse_ratio = lift_summary.get("lift_threshold_rmse_ratio")
    lift_window_counts = lift_summary.get("window_counts") if isinstance(lift_summary.get("window_counts"), dict) else {}
    lift_rmse_windows_processed = lift_window_counts.get("n_rmse_windows_processed")
    lift_rmse_windows_usable = lift_window_counts.get("n_rmse_windows_usable")
    lift_outcome_windows_not_due = lift_window_counts.get("n_outcome_windows_not_due")
    lift_readiness_denominator_included = lift_window_counts.get("n_readiness_denominator_included")

    proof_cmd = [
        python_bin,
        str(proof_script),
        "--db",
        str(db_path),
        "--json",
    ]
    proof_proc = _run_command(proof_cmd, cwd=repo_root)
    proof_payload = _parse_json_payload(f"{proof_proc.stdout or ''}\n{proof_proc.stderr or ''}") or {}

    proof_is_valid = bool(proof_payload.get("is_proof_valid", False))
    proof_is_profitable = bool(proof_payload.get("is_profitable", False))
    proof_pass = proof_is_valid and (
        proof_is_profitable if proof_profitable_required else True
    )
    proof_status = "PASS" if proof_pass and proof_proc.returncode == 0 else "FAIL"

    metrics = proof_payload.get("metrics") if isinstance(proof_payload.get("metrics"), dict) else {}
    winning = int(metrics.get("winning_trades", 0) or 0)
    losing = int(metrics.get("losing_trades", 0) or 0)
    proof_requirements = _load_profitability_requirements(proof_requirements_path)
    evidence_progress = _build_evidence_progress(metrics=metrics, requirements=proof_requirements)

    reconcile_pass = True
    if reconcile_result.get("requested"):
        reconcile_pass = str(reconcile_result.get("status") or "FAIL").upper() == "PASS"

    repo_state = _collect_git_state(repo_root)
    live_cycle_binding = _load_latest_live_cycle_binding(db_path)
    artifact_binding = _evaluate_artifact_binding(
        lift_summary=lift_summary,
        live_cycle_binding=live_cycle_binding,
        repo_state=repo_state,
    )
    artifact_binding_pass = bool(artifact_binding.get("pass", False))

    gate_pass = lift_pass and proof_pass and reconcile_pass and artifact_binding_pass
    gate_status = "PASS" if gate_pass else "FAIL"
    if lift_inconclusive:
        gate_semantics_status = (
            "INCONCLUSIVE_ALLOWED" if lift_inconclusive_allowed else "INCONCLUSIVE_BLOCKED"
        )
    elif gate_pass:
        gate_semantics_status = "PASS"
    else:
        gate_semantics_status = "FAIL"

    # A4+A5 fix: warn when gate passes via warmup/INCONCLUSIVE_ALLOWED exemption.
    # This makes the soft-pass visible to operators monitoring logs/notifications.
    if lift_inconclusive and lift_inconclusive_allowed:
        import logging as _pag_log_mod
        _pag_log_mod.getLogger(__name__).warning(
            "[PAG] Gate passes via INCONCLUSIVE_ALLOWED (warmup exemption). "
            "Ensemble lift evidence is INSUFFICIENT — this is NOT a genuine ensemble "
            "quality signal. effective_audits=%s, first_audit_utc=%s, "
            "allow_inconclusive_until_utc=%s",
            lift_effective_audits,
            warmup_policy.get("first_audit_ts_utc"),
            warmup_policy.get("allow_inconclusive_until_utc"),
        )

    window_counts = (
        lift_summary.get("window_counts", {})
        if isinstance(lift_summary.get("window_counts"), dict)
        else {}
    )
    outcome_matched = _safe_int(window_counts.get("n_outcome_windows_matched"), 0)
    outcome_eligible = _safe_int(window_counts.get("n_outcome_windows_eligible"), 0)
    matched_over_eligible = _safe_ratio(outcome_matched, outcome_eligible)
    # THR-02 fix: read linkage thresholds from forecaster_monitoring.yml so
    # they can be tuned without code changes.  Defaults match the prior
    # hardcoded values (10 / 0.8) and the yaml values above.
    _linkage_rmse_cfg = _load_regression_monitoring_config(monitor_config)
    _linkage_min_matched = int(_linkage_rmse_cfg.get("linkage_min_matched", 10) or 10)
    _linkage_min_ratio = float(_linkage_rmse_cfg.get("linkage_min_ratio", 0.8) or 0.8)
    # Phase 10: During warmup, relax THIN_LINKAGE to a 1-match floor so the
    # gate does not block when the system is still accumulating live closed
    # trades. Full thresholds apply once warmup has expired.
    _linkage_warmup_active = not bool(warmup_policy.get("warmup_expired", True))
    if _linkage_warmup_active:
        _linkage_min_matched = 1
        _linkage_min_ratio = 0.0
    # Vacuously pass when no eligible records exist yet (accumulation phase).
    _linkage_no_eligible = outcome_eligible == 0
    linkage_pass = _linkage_no_eligible or (
        outcome_matched >= _linkage_min_matched
        and matched_over_eligible >= _linkage_min_ratio
    )

    non_trade_count = _safe_int(window_counts.get("n_outcome_windows_non_trade_context"), 0)
    invalid_context_count = _safe_int(window_counts.get("n_outcome_windows_invalid_context"), 0)
    scope_block = lift_summary.get("scope", {}) if isinstance(lift_summary.get("scope"), dict) else {}
    production_audit_only = bool(
        scope_block.get(
            "production_audit_only",
            str(audit_dir.name).strip().lower() == "production",
        )
    )
    evidence_hygiene_pass = (
        production_audit_only and non_trade_count == 0 and invalid_context_count == 0
    )
    linkage_waterfall = _build_linkage_waterfall(
        window_counts,
        production_audit_only=production_audit_only,
    )
    admission_summary = lift_summary.get("admission_summary")
    admission_summary = admission_summary if isinstance(admission_summary, dict) else {}
    accepted_records = _safe_int(
        admission_summary.get("accepted_records"),
        _safe_int(window_counts.get("n_accepted_records"), 0),
    )
    accepted_noneligible_records = _safe_int(
        admission_summary.get("accepted_noneligible_records"),
        _safe_int(
            window_counts.get("n_accepted_noneligible_records"),
            max(0, linkage_waterfall.get("raw_candidates", 0) - linkage_waterfall.get("linked", 0)),
        ),
    )
    eligible_records = _safe_int(
        admission_summary.get("eligible_records"),
        _safe_int(window_counts.get("n_eligible_records"), outcome_eligible),
    )
    quarantined_records = _safe_int(
        admission_summary.get("quarantined_records"),
        _safe_int(window_counts.get("n_quarantined_records"), 0),
    )
    duplicate_conflicts = _safe_int(
        admission_summary.get("duplicate_conflicts"),
        _safe_int(window_counts.get("n_duplicate_conflicts"), 0),
    )
    admission_missing_execution_metadata_records = _safe_int(
        admission_summary.get("missing_execution_metadata_records"),
        _safe_int(window_counts.get("n_admission_missing_execution_metadata_records"), 0),
    )
    contract_version_count = _safe_int(window_counts.get("n_contract_versions"), 0)
    cohort_fingerprint_count = _safe_int(window_counts.get("n_cohort_fingerprints"), 0)
    contract_version_drift = contract_version_count > 1
    cohort_fingerprint_drift = cohort_fingerprint_count > 1

    integrity_metrics = _compute_lifecycle_integrity(db_path)
    close_before_entry_count = _safe_int(integrity_metrics.get("close_before_entry_count"), 0)
    missing_exit_reason_count = _safe_int(
        integrity_metrics.get("closed_missing_exit_reason_count"), 0
    )
    high_integrity_violation_count = close_before_entry_count + missing_exit_reason_count
    integrity_query_error = integrity_metrics.get("query_error")
    integrity_pass = high_integrity_violation_count == 0 and not integrity_query_error

    masked_violation_count, masked_violation_ids = _count_masked_unlinked_closes(db_path)

    gates_pass = bool(gate_pass)
    phase3_ready = bool(
        gates_pass
        and linkage_pass
        and evidence_hygiene_pass
        and integrity_pass
        and duplicate_conflicts == 0
        and quarantined_records == 0
        and not contract_version_drift
        and not cohort_fingerprint_drift
    )
    phase3_fail_reasons: List[str] = []
    if not gates_pass:
        phase3_fail_reasons.append("GATES_FAIL")
    if not linkage_pass:
        phase3_fail_reasons.append("THIN_LINKAGE")
    if not evidence_hygiene_pass:
        phase3_fail_reasons.append("EVIDENCE_HYGIENE_FAIL")
    if not integrity_pass:
        phase3_fail_reasons.append("HIGH_INTEGRITY_VIOLATION")
    if duplicate_conflicts > 0:
        phase3_fail_reasons.append("DUPLICATE_CONFLICT")
    if quarantined_records > 0:
        phase3_fail_reasons.append("QUARANTINED_RECORDS")
    if contract_version_drift:
        phase3_fail_reasons.append("CONTRACT_VERSION_DRIFT")
    if cohort_fingerprint_drift:
        phase3_fail_reasons.append("COHORT_FINGERPRINT_DRIFT")
    if summary_metrics_error:
        phase3_fail_reasons.append(summary_metrics_error)
    if not summary_invocation_match and summary_metrics_error != "SUMMARY_INVOCATION_MISMATCH":
        phase3_fail_reasons.append("SUMMARY_INVOCATION_MISMATCH")
    if not artifact_binding_pass:
        phase3_fail_reasons.append("ARTIFACT_STALE_OR_UNBOUND")
    phase3_reason = "READY" if phase3_ready else ",".join(phase3_fail_reasons)
    strict_gate_pass = bool(gate_pass and gate_semantics_status == "PASS")
    strict_phase3_ready = bool(phase3_ready and strict_gate_pass)
    strict_phase3_reason = phase3_reason
    if not strict_phase3_ready and gate_pass and gate_semantics_status != "PASS":
        strict_reason_code = f"GATE_SEMANTICS_{gate_semantics_status}"
        strict_phase3_reason = (
            f"{phase3_reason},{strict_reason_code}" if phase3_reason else strict_reason_code
        )

    thresholds = _collect_thresholds(
        monitor_config=monitor_config,
        proof_requirements=proof_requirements,
        lift_inconclusive_allowed=lift_inconclusive_allowed,
        proof_profitable_required=proof_profitable_required,
        require_holding_period=bool(args.require_holding_period),
        warmup_policy=warmup_policy,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stamped_output = output_path.parent / f"{output_path.stem}_{stamp}{output_path.suffix}"

    payload: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "pass_semantics_version": PASS_SEMANTICS_VERSION,
        "lift_inconclusive_allowed": bool(lift_inconclusive_allowed),
        "proof_profitable_required": bool(proof_profitable_required),
        "first_audit_ts_utc": warmup_policy.get("first_audit_ts_utc"),
        "allow_inconclusive_until_utc": warmup_policy.get("allow_inconclusive_until_utc"),
        "warmup_expired": bool(warmup_policy.get("warmup_expired", True)),
        "phase3_ready": bool(phase3_ready),
        "phase3_reason": phase3_reason,
        "phase3_strict_ready": bool(strict_phase3_ready),
        "phase3_strict_reason": strict_phase3_reason,
        "repo_state": repo_state,
        "inputs": {
            "db": str(db_path),
            "proof_requirements": str(proof_requirements_path),
            "audit_dir": str(audit_dir),
            "monitor_config": str(monitor_config),
            "max_files": int(args.max_files),
            "require_holding_period": bool(args.require_holding_period),
            "allow_inconclusive_lift": bool(args.allow_inconclusive_lift),
            "include_research": bool(args.include_research),
            "require_profitable": bool(args.require_profitable),
            "unattended_profile": bool(args.unattended_profile),
            "unknown_args_ignored": [],
        },
        "summary_invocation_match": bool(summary_invocation_match),
        "summary_invocation_mismatch": summary_mismatch_details,
        "artifact_binding": artifact_binding,
        "live_cycle_binding": live_cycle_binding,
        "reconciliation": reconcile_result,
        "lift_gate": {
            "status": lift_status,
            "pass": lift_pass,
            "exit_code": int(lift_proc.returncode),
            "inconclusive": lift_inconclusive,
            "lift_inconclusive_allowed": bool(lift_inconclusive_allowed),
            "summary_metrics_error": summary_metrics_error,
            "missing_summary_metric_keys": missing_summary_metric_keys,
            "measurement_contract_version": lift_measurement_contract_version,
            "baseline_model": lift_baseline_model,
            "lift_threshold_rmse_ratio": lift_threshold_rmse_ratio,
            "decision": lift_decision,
            "decision_reason": lift_decision_reason,
            "effective_audits": lift_effective_audits,
            "violation_rate": lift_violation_rate,
            "max_violation_rate": lift_max_violation_rate,
            "lift_fraction": lift_fraction,
            "min_lift_fraction": min_lift_fraction,
            "rmse_windows_processed": lift_rmse_windows_processed,
            "rmse_windows_usable": lift_rmse_windows_usable,
            "readiness_denominator_included": lift_readiness_denominator_included,
            "outcome_windows_not_due": lift_outcome_windows_not_due,
            "output_tail": _tail_lines(lift_output),
        },
        "profitability_proof": {
            "status": proof_status,
            "pass": proof_pass,
            "command_exit_code": int(proof_proc.returncode),
            "is_proof_valid": proof_is_valid,
            "is_profitable": proof_is_profitable,
            "proof_profitable_required": bool(proof_profitable_required),
            "total_pnl": metrics.get("total_pnl"),
            "profit_factor": metrics.get("profit_factor"),
            "win_rate": metrics.get("win_rate"),
            "closed_trades": winning + losing,
            "trading_days": metrics.get("trading_days"),
            "evidence_progress": evidence_progress,
            "violations": proof_payload.get("violations", []),
            "warnings": proof_payload.get("warnings", []),
            "recommendations": proof_payload.get("recommendations", []),
            "output_tail": _tail_lines(f"{proof_proc.stdout or ''}\n{proof_proc.stderr or ''}"),
        },
        "production_profitability_gate": {
            "status": gate_status,
            "pass": gate_pass,
            "strict_pass": bool(strict_gate_pass),
            "reconcile_pass": reconcile_pass,
            "gate_semantics_status": gate_semantics_status,
            "first_audit_ts_utc": warmup_policy.get("first_audit_ts_utc"),
            "allow_inconclusive_until_utc": warmup_policy.get("allow_inconclusive_until_utc"),
            "warmup_expired": bool(warmup_policy.get("warmup_expired", True)),
        },
        "readiness": {
            "phase3_ready": bool(phase3_ready),
            "phase3_reason": phase3_reason,
            "phase3_strict_ready": bool(strict_phase3_ready),
            "phase3_strict_reason": strict_phase3_reason,
            "gates_pass": bool(gates_pass),
            "linkage_pass": bool(linkage_pass),
            "evidence_hygiene_pass": bool(evidence_hygiene_pass),
            "integrity_pass": bool(integrity_pass),
            "artifact_binding_pass": bool(artifact_binding_pass),
            "outcome_matched": outcome_matched,
            "outcome_eligible": outcome_eligible,
            "matched_over_eligible": matched_over_eligible,
            "linkage_min_matched": _linkage_min_matched,
            "linkage_min_ratio": _linkage_min_ratio,
            "linkage_warmup_active": bool(_linkage_warmup_active),
            "linkage_no_eligible": bool(_linkage_no_eligible),
            "non_trade_context_count": non_trade_count,
            "invalid_context_count": invalid_context_count,
            "linkage_waterfall": linkage_waterfall,
            "production_audit_only": production_audit_only,
            "high_integrity_violation_count": high_integrity_violation_count,
            "close_before_entry_count": close_before_entry_count,
            "closed_missing_exit_reason_count": missing_exit_reason_count,
            "integrity_query_error": integrity_query_error,
            "masked_integrity_violations": masked_violation_count,
            "masked_integrity_violation_ids": masked_violation_ids,
            "admission_summary": admission_summary,
        },
        "readiness_v2": {
            "overall_ready": bool(phase3_ready),
            "strict_overall_ready": bool(strict_phase3_ready),
            "evidence_chain_ready": bool(
                linkage_pass
                and evidence_hygiene_pass
                and duplicate_conflicts == 0
                and quarantined_records == 0
                and not contract_version_drift
                and not cohort_fingerprint_drift
            ),
            "economic_ready": bool(gates_pass),
            "operational_ready": bool(integrity_pass and artifact_binding_pass and summary_invocation_match),
            "reason_code": phase3_reason,
            "strict_reason_code": strict_phase3_reason,
            "accepted_records": accepted_records,
            "accepted_noneligible_records": accepted_noneligible_records,
            "eligible_records": eligible_records,
            "eligible_opens": 0,
            "eligible_closes": outcome_eligible,
            "linked_closes": linkage_waterfall.get("linked"),
            "matched_closes": outcome_matched,
            "orphan_closes": close_before_entry_count,
            "quarantined_records": quarantined_records,
            "duplicate_conflicts": duplicate_conflicts,
            "missing_execution_metadata_records": admission_missing_execution_metadata_records,
            "contract_version_drift": bool(contract_version_drift),
            "cohort_fingerprint_drift": bool(cohort_fingerprint_drift),
            "production_audit_only": production_audit_only,
            "measurement_contract_version": lift_measurement_contract_version,
            "baseline_model": lift_baseline_model,
            "lift_threshold_rmse_ratio": lift_threshold_rmse_ratio,
            "rmse_windows_processed": lift_rmse_windows_processed,
            "rmse_windows_usable": lift_rmse_windows_usable,
            "readiness_denominator_included": lift_readiness_denominator_included,
            "outcome_windows_not_due": lift_outcome_windows_not_due,
            "linkage_waterfall": linkage_waterfall,
            "admission_summary": admission_summary,
        },
        "thresholds": thresholds,
    }
    payload["telemetry_contract"] = normalize_telemetry_payload(
        {
            "status": gate_semantics_status,
            "reason_code": phase3_reason,
            "context_type": "TRADE",
            "severity": "HIGH" if not gate_pass else "LOW",
            "blocking": not gate_pass,
            "counts_toward_readiness_denominator": True,
            "counts_toward_linkage_denominator": False,
            "generated_utc": timestamp_utc,
            "source_script": "scripts/production_audit_gate.py",
        },
        source_script="scripts/production_audit_gate.py",
        generated_utc=timestamp_utc,
    )

    write_promoted_json_artifact(
        stamped_path=stamped_output,
        latest_path=output_path,
        payload=payload,
        validate_fn=_validate_output_artifact,
        quarantine_dir=output_path.parent / "quarantine",
    )

    print("=== Production Audit Gate ===")
    print(f"Timestamp (UTC): {timestamp_utc}")
    print(f"Lift status    : {lift_status} (pass={lift_pass})")
    if payload["lift_gate"]["decision"]:
        print(
            f"Lift decision  : {payload['lift_gate']['decision']} "
            f"({payload['lift_gate']['decision_reason']})"
        )
    print(
        f"Proof status   : {proof_status} "
        f"(valid={proof_is_valid}, profitable={proof_is_profitable})"
    )
    print(
        "Semantics      : "
        f"v{PASS_SEMANTICS_VERSION} "
        f"inconclusive_allowed={int(lift_inconclusive_allowed)} "
        f"profitable_required={int(proof_profitable_required)} "
        f"warmup_expired={int(bool(warmup_policy.get('warmup_expired', True)))}"
    )
    lift_thr = thresholds.get("lift", {})
    proof_thr = thresholds.get("proof", {})
    print(
        "Thresholds    : "
        f"lift_min={lift_thr.get('min_lift_fraction')} "
        f"max_violation={lift_thr.get('max_violation_rate')} "
        f"missing_rate={lift_thr.get('max_missing_ensemble_rate')} | "
        f"proof_pf>={proof_thr.get('min_profit_factor')} "
        f"min_trades={proof_thr.get('min_closed_trades')} "
        f"min_days={proof_thr.get('min_trading_days')}"
    )
    print(
        "Proof runway   : "
        f"closed={evidence_progress.get('closed_trades')}/{evidence_progress.get('min_closed_trades')} "
        f"days={evidence_progress.get('trading_days')}/{evidence_progress.get('min_trading_days')} "
        f"remaining_days={evidence_progress.get('remaining_trading_days')}"
    )
    print(f"Gate status    : {gate_status} (semantics={gate_semantics_status})")
    print(
        "Phase3 ready   : "
        f"{int(phase3_ready)} (reason={phase3_reason}, matched={outcome_matched}/{outcome_eligible}, "
        f"integrity_high={high_integrity_violation_count})"
    )
    print(
        "Phase3 strict  : "
        f"{int(strict_phase3_ready)} (reason={strict_phase3_reason})"
    )
    print(
        "Artifact bind  : "
        f"{'PASS' if artifact_binding_pass else 'FAIL'} "
        f"(summary_ts={artifact_binding.get('summary_generated_utc')}, "
        f"live_ts={artifact_binding.get('latest_live_cycle_ts_utc')}, "
        f"run_id={artifact_binding.get('latest_live_run_id')}, "
        f"commit={artifact_binding.get('repo_head')})"
    )
    if not artifact_binding_pass:
        print(
            "Artifact reason: "
            f"{','.join(artifact_binding.get('reason_codes') or []) or 'UNKNOWN'}"
        )
    if reconcile_result.get("requested"):
        print(
            "Reconcile step : "
            f"{reconcile_result.get('status')} "
            f"(apply={bool(reconcile_result.get('apply'))}, "
            f"close_ids={reconcile_result.get('close_ids')})"
        )
        print(
            "Reconcile verify: "
            f"remaining_unlinked={reconcile_result.get('remaining_unlinked_closes')} "
            f"reason={reconcile_result.get('status_reason')}"
        )
    print(f"Artifact       : {output_path}")
    print(f"Artifact (run) : {stamped_output}")
    try:
        state = payload.get("repo_state") if isinstance(payload.get("repo_state"), dict) else {}
        if state.get("available") and isinstance(state.get("status"), dict):
            st = state["status"]
            print(
                "Repo state     : "
                f"tracked_changed={st.get('tracked_changed')} untracked={st.get('untracked')} "
                f"ahead={state.get('ahead')} behind={state.get('behind')}"
            )
    except Exception:
        pass

    def _truthy(value: str) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    openclaw_to_raw = (args.openclaw_to or "").strip()
    try:
        from utils.openclaw_cli import parse_openclaw_targets

        default_channel = (os.getenv("OPENCLAW_CHANNEL") or "").strip() or None
        openclaw_targets = parse_openclaw_targets(openclaw_to_raw, default_channel=default_channel)
    except Exception:
        openclaw_targets = []
    notify_openclaw = bool(args.notify_openclaw)
    if not notify_openclaw:
        raw_default = (os.getenv("PMX_NOTIFY_OPENCLAW") or "").strip()
        if raw_default:
            notify_openclaw = _truthy(raw_default)
        else:
            # Default: if OPENCLAW_TARGETS/OPENCLAW_TO is configured, send the summary.
            notify_openclaw = bool(openclaw_targets)

    if notify_openclaw:
        if not openclaw_targets:
            print(
                "OpenClaw notify requested but no targets configured (set --openclaw-to or OPENCLAW_TARGETS/OPENCLAW_TO).",
                file=sys.stderr,
            )
        else:
            try:
                from utils.openclaw_cli import send_message_multi

                lift_decision = payload["lift_gate"].get("decision")
                lift_reason = payload["lift_gate"].get("decision_reason")
                proof_pnl = payload["profitability_proof"].get("total_pnl")
                proof_pf = payload["profitability_proof"].get("profit_factor")
                proof_wr = payload["profitability_proof"].get("win_rate")

                msg_lines = [
                    f"[PMX] Production audit gate: {gate_status}",
                    f"UTC: {timestamp_utc}",
                    f"Lift: {lift_status} pass={lift_pass} decision={lift_decision} reason={lift_reason}",
                    f"Proof: {proof_status} pnl={proof_pnl} pf={proof_pf} win_rate={proof_wr}",
                    f"Artifact: {output_path}",
                ]
                message = "\n".join([line for line in msg_lines if line is not None])

                results = send_message_multi(
                    targets=openclaw_targets,
                    message=message,
                    command=str(args.openclaw_command or "openclaw"),
                    cwd=repo_root,
                    timeout_seconds=float(args.openclaw_timeout_seconds),
                )
                for result in results:
                    if result.ok:
                        continue
                    print(
                        f"OpenClaw notify failed (exit={result.returncode}): "
                        f"{(result.stderr or result.stdout or '').strip()[:200]}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"OpenClaw notify failed: {exc}", file=sys.stderr)

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
