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

try:
    from scripts.audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT
except Exception:  # pragma: no cover - script execution path fallback
    from audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT

try:
    from scripts.telemetry_adapter import normalize_telemetry_payload
except Exception:  # pragma: no cover - script execution path fallback
    from telemetry_adapter import normalize_telemetry_payload


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


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
) -> bool:
    if not _summary_matches_audit_dir(summary, audit_dir):
        return False
    raw_max_files = summary.get("max_files")
    if raw_max_files is None:
        return True
    try:
        return int(raw_max_files) == int(max_files)
    except Exception:
        return False


def _extract_lift_output_metrics(output: str) -> Dict[str, Any]:
    text = output or ""

    def _capture_int(pattern: str) -> Optional[int]:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _capture_ratio(pattern: str) -> Optional[float]:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1)) / 100.0
        except Exception:
            return None

    metrics: Dict[str, Any] = {}
    metrics["effective_audits"] = _capture_int(r"Effective audits with RMSE:\s*(\d+)")
    metrics["violation_count"] = _capture_int(
        r"Violations \(ensemble worse than baseline beyond tolerance\):\s*(\d+)"
    )
    metrics["violation_rate"] = _capture_ratio(
        r"Violation rate:\s*([0-9]+(?:\.[0-9]+)?)%\s*\(max allowed"
    )
    metrics["max_violation_rate"] = _capture_ratio(
        r"Violation rate:\s*[0-9]+(?:\.[0-9]+)?%\s*\(max allowed\s*([0-9]+(?:\.[0-9]+)?)%\)"
    )
    metrics["lift_fraction"] = _capture_ratio(
        r"Ensemble lift fraction:\s*([0-9]+(?:\.[0-9]+)?)%\s*\(required"
    )
    metrics["min_lift_fraction"] = _capture_ratio(
        r"Ensemble lift fraction:\s*[0-9]+(?:\.[0-9]+)?%\s*\(required\s*>=\s*([0-9]+(?:\.[0-9]+)?)%\)"
    )
    metrics["ensemble_missing_rate"] = _capture_ratio(
        r"Missing ensemble metrics\s*:\s*\d+/\d+\s*\(([0-9]+(?:\.[0-9]+)?)%\)"
    )
    metrics["max_missing_ensemble_rate"] = _capture_ratio(
        r"Missing ensemble metrics\s*:\s*\d+/\d+\s*\([0-9]+(?:\.[0-9]+)?%\)\s*\(max allowed\s*([0-9]+(?:\.[0-9]+)?)%\)"
    )

    decision_match = re.search(r"Decision:\s*([A-Z_]+)\s*\((.+?)\)", text)
    metrics["decision"] = decision_match.group(1) if decision_match else None
    metrics["decision_reason"] = decision_match.group(2) if decision_match else None
    return metrics


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
    if unknown:
        print(f"[WARN] Ignoring unknown args: {' '.join(unknown)}", file=sys.stderr)
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
    if args.unattended_profile:
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
    lift_inconclusive = "RMSE gate inconclusive" in lift_output
    lift_output_metrics = _extract_lift_output_metrics(lift_output)

    lift_summary = _safe_load_json(summary_cache_path) or {}
    if lift_summary and not _summary_matches_invocation(
        lift_summary,
        audit_dir=audit_dir,
        max_files=int(args.max_files),
    ):
        lift_summary = {}

    lift_status = "PASS"
    if lift_proc.returncode != 0:
        lift_status = "FAIL"
        # Do not attach stale cached decision metadata when the active lift run failed.
        lift_summary = {}
    elif lift_inconclusive:
        lift_status = "INCONCLUSIVE"

    lift_pass = lift_proc.returncode == 0 and (
        lift_inconclusive_allowed or not lift_inconclusive
    )

    lift_decision = lift_output_metrics.get("decision") or lift_summary.get("decision")
    lift_decision_reason = lift_output_metrics.get("decision_reason") or lift_summary.get(
        "decision_reason"
    )
    if lift_inconclusive:
        lift_decision = None
        lift_decision_reason = "insufficient effective audits for RMSE gate"

    lift_effective_audits = lift_output_metrics.get("effective_audits")
    if lift_effective_audits is None:
        lift_effective_audits = lift_summary.get("effective_audits")

    lift_violation_rate = lift_output_metrics.get("violation_rate")
    if lift_violation_rate is None:
        lift_violation_rate = lift_summary.get("violation_rate")

    lift_max_violation_rate = lift_output_metrics.get("max_violation_rate")
    if lift_max_violation_rate is None:
        lift_max_violation_rate = lift_summary.get("max_violation_rate")

    lift_fraction = lift_output_metrics.get("lift_fraction")
    if lift_fraction is None:
        lift_fraction = lift_summary.get("lift_fraction")

    min_lift_fraction = lift_output_metrics.get("min_lift_fraction")
    if min_lift_fraction is None:
        min_lift_fraction = lift_summary.get("min_lift_fraction")

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

    gate_pass = lift_pass and proof_pass and reconcile_pass
    gate_status = "PASS" if gate_pass else "FAIL"
    if lift_inconclusive:
        gate_semantics_status = (
            "INCONCLUSIVE_ALLOWED" if lift_inconclusive_allowed else "INCONCLUSIVE_BLOCKED"
        )
    elif gate_pass:
        gate_semantics_status = "PASS"
    else:
        gate_semantics_status = "FAIL"

    window_counts = (
        lift_summary.get("window_counts", {})
        if isinstance(lift_summary.get("window_counts"), dict)
        else {}
    )
    outcome_matched = _safe_int(window_counts.get("n_outcome_windows_matched"), 0)
    outcome_eligible = _safe_int(window_counts.get("n_outcome_windows_eligible"), 0)
    matched_over_eligible = _safe_ratio(outcome_matched, outcome_eligible)
    linkage_pass = outcome_matched >= 10 and matched_over_eligible >= 0.8

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

    integrity_metrics = _compute_lifecycle_integrity(db_path)
    close_before_entry_count = _safe_int(integrity_metrics.get("close_before_entry_count"), 0)
    missing_exit_reason_count = _safe_int(
        integrity_metrics.get("closed_missing_exit_reason_count"), 0
    )
    high_integrity_violation_count = close_before_entry_count + missing_exit_reason_count
    integrity_query_error = integrity_metrics.get("query_error")
    integrity_pass = high_integrity_violation_count == 0 and not integrity_query_error

    gates_pass = bool(gate_pass)
    phase3_ready = bool(gates_pass and linkage_pass and evidence_hygiene_pass and integrity_pass)
    phase3_fail_reasons: List[str] = []
    if not gates_pass:
        phase3_fail_reasons.append("GATES_FAIL")
    if not linkage_pass:
        phase3_fail_reasons.append("THIN_LINKAGE")
    if not evidence_hygiene_pass:
        phase3_fail_reasons.append("EVIDENCE_HYGIENE_FAIL")
    if not integrity_pass:
        phase3_fail_reasons.append("HIGH_INTEGRITY_VIOLATION")
    phase3_reason = "READY" if phase3_ready else ",".join(phase3_fail_reasons)

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
        "repo_state": _collect_git_state(repo_root),
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
            "unknown_args_ignored": unknown,
        },
        "reconciliation": reconcile_result,
        "lift_gate": {
            "status": lift_status,
            "pass": lift_pass,
            "exit_code": int(lift_proc.returncode),
            "inconclusive": lift_inconclusive,
            "lift_inconclusive_allowed": bool(lift_inconclusive_allowed),
            "decision": lift_decision,
            "decision_reason": lift_decision_reason,
            "effective_audits": lift_effective_audits,
            "violation_rate": lift_violation_rate,
            "max_violation_rate": lift_max_violation_rate,
            "lift_fraction": lift_fraction,
            "min_lift_fraction": min_lift_fraction,
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
            "reconcile_pass": reconcile_pass,
            "gate_semantics_status": gate_semantics_status,
            "first_audit_ts_utc": warmup_policy.get("first_audit_ts_utc"),
            "allow_inconclusive_until_utc": warmup_policy.get("allow_inconclusive_until_utc"),
            "warmup_expired": bool(warmup_policy.get("warmup_expired", True)),
        },
        "readiness": {
            "phase3_ready": bool(phase3_ready),
            "phase3_reason": phase3_reason,
            "gates_pass": bool(gates_pass),
            "linkage_pass": bool(linkage_pass),
            "evidence_hygiene_pass": bool(evidence_hygiene_pass),
            "integrity_pass": bool(integrity_pass),
            "outcome_matched": outcome_matched,
            "outcome_eligible": outcome_eligible,
            "matched_over_eligible": matched_over_eligible,
            "non_trade_context_count": non_trade_count,
            "invalid_context_count": invalid_context_count,
            "production_audit_only": production_audit_only,
            "high_integrity_violation_count": high_integrity_violation_count,
            "close_before_entry_count": close_before_entry_count,
            "closed_missing_exit_reason_count": missing_exit_reason_count,
            "integrity_query_error": integrity_query_error,
        },
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

    artifact_text = json.dumps(payload, indent=2)
    output_path.write_text(artifact_text, encoding="utf-8")
    stamped_output.write_text(artifact_text, encoding="utf-8")

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
    print(
        "Proof runway   : "
        f"closed={evidence_progress.get('closed_trades')}/{evidence_progress.get('min_closed_trades')} "
        f"days={evidence_progress.get('trading_days')}/{evidence_progress.get('min_trading_days')} "
        f"remaining_days={evidence_progress.get('remaining_trading_days')}"
    )
    print(f"Gate status    : {gate_status}")
    print(
        "Phase3 ready   : "
        f"{int(phase3_ready)} (reason={phase3_reason}, matched={outcome_matched}/{outcome_eligible}, "
        f"integrity_high={high_integrity_violation_count})"
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
