#!/usr/bin/env python3
"""
OpenClaw Training Autopilot (cron-safe)

Goal
  - Keep local models improving until BOTH:
    1) profitability proof meets benchmark requirements, AND
    2) forecaster adversarial-suite metrics meet CI thresholds,
    within a configurable tolerance factor (default: 0.989).

Design constraints
  - Must be safe to run from OpenClaw cron: fast, bounded, no command chaining.
  - Long training is launched in a detached background process so the cron
    session doesn't get stuck.
  - State is persisted on disk so behavior survives restarts.
  - Never prints or stores secret values (best-effort redaction on tails only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Needed when invoked as `python scripts/...py` (sys.path[0] would be `.../scripts`).
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_MONITOR_CONFIG = PROJECT_ROOT / "config" / "forecaster_monitoring_ci.yml"
DEFAULT_PROFIT_REQUIREMENTS = PROJECT_ROOT / "config" / "profitability_proof_requirements.yml"
DEFAULT_SUITE_OUTPUT = PROJECT_ROOT / "logs" / "automation" / "training_priority" / "adversarial_forecaster_suite.json"
DEFAULT_STATE_FILE = PROJECT_ROOT / "logs" / "automation" / "training_autopilot" / "state.json"

TRAINING_PROPOSALS_DIR = PROJECT_ROOT / "logs" / "llm_activity" / "proposals" / "training"
TRAINING_FEEDBACK_DIR = PROJECT_ROOT / "logs" / "llm_activity" / "feedback"

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-\._~\+/=]{16,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),
    re.compile(r"\b(token|secret|password|api[_-]?key)\s*[:=]\s*[^,\s]+", re.IGNORECASE),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned[:160] or "item"


def _redact_text(text: str) -> str:
    out = str(text or "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _tail_file(path: Path, max_lines: int) -> str:
    try:
        if max_lines <= 0 or not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass

    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    # Windows fallback without extra deps.
    try:
        import ctypes  # pragma: no cover

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:  # pragma: no cover
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    return raw if isinstance(raw, dict) else {}


def _apply_factor_min(threshold: float, factor: float) -> float:
    return float(threshold) * float(factor)


def _apply_factor_max(threshold: float, factor: float) -> float:
    f = float(factor)
    if f <= 0:
        return float(threshold)
    # factor < 1.0 loosens max thresholds by ~1/(factor) - 1.
    return float(threshold) / f


def _ceil_min(value: float) -> int:
    try:
        return int(math.ceil(float(value)))
    except Exception:
        return int(value)


def _profitability_eval(
    *,
    db_path: Path,
    requirements_path: Path,
    factor: float,
) -> dict[str, Any]:
    """
    Returns:
      {
        "status": "PASS"|"FAIL"|"BLOCKED",
        "reasons": [...],
        "metrics": {...},
        "requirements": {...},
      }
    """
    if not db_path.exists():
        return {"status": "BLOCKED", "reasons": [f"db_missing:{db_path}"], "metrics": {}, "requirements": {}}

    try:
        # Local import to avoid heavy import costs when cron is just probing.
        from scripts import validate_profitability_proof as proof_mod  # type: ignore

        result = proof_mod.validate_profitability_proof(str(db_path))
    except Exception as exc:
        return {"status": "FAIL", "reasons": [f"validator_exception:{exc}"], "metrics": {}, "requirements": {}}

    metrics = result.get("metrics") if isinstance(result, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}

    raw_req = _load_yaml(requirements_path)
    req = raw_req.get("profitability_proof_requirements") if isinstance(raw_req.get("profitability_proof_requirements"), dict) else {}

    reasons: list[str] = []
    ok = True

    total_trades = int(metrics.get("total_trades", 0) or 0)
    closed_trades = int((metrics.get("winning_trades", 0) or 0) + (metrics.get("losing_trades", 0) or 0))
    trading_days = int(metrics.get("trading_days", 0) or 0)
    total_pnl = float(metrics.get("total_pnl", 0.0) or 0.0)
    profit_factor = metrics.get("profit_factor")
    profit_factor = float(profit_factor) if isinstance(profit_factor, (int, float)) else None
    win_rate = metrics.get("win_rate")
    win_rate = float(win_rate) if isinstance(win_rate, (int, float)) else None

    null_pct = float(metrics.get("null_data_source_pct", 0.0) or 0.0)
    synthetic_count = int(metrics.get("synthetic_ticker_count", 0) or 0)

    dq = req.get("data_quality") or {}
    ss = req.get("statistical_significance") or {}
    perf = req.get("performance") or {}
    audit = req.get("audit_trail") or {}

    min_coverage = float(dq.get("min_data_source_coverage", 1.0))
    coverage = max(0.0, 1.0 - null_pct)
    cov_req = _apply_factor_min(min_coverage, factor)
    if coverage < cov_req:
        ok = False
        reasons.append(f"coverage:{coverage:.4f}<{cov_req:.4f}")

    max_syn_pct = float(dq.get("max_synthetic_ticker_pct", 0.0))
    syn_pct = (synthetic_count / total_trades) if total_trades else (1.0 if synthetic_count else 0.0)
    syn_req = _apply_factor_max(max_syn_pct, factor)
    if syn_pct > syn_req:
        ok = False
        reasons.append(f"synthetic_pct:{syn_pct:.4f}>{syn_req:.4f}")

    min_closed = _ceil_min(_apply_factor_min(float(ss.get("min_closed_trades", 30)), factor))
    if closed_trades < min_closed:
        ok = False
        reasons.append(f"closed_trades:{closed_trades}<{min_closed}")

    min_days = _ceil_min(_apply_factor_min(float(ss.get("min_trading_days", 21)), factor))
    if trading_days < min_days:
        ok = False
        reasons.append(f"trading_days:{trading_days}<{min_days}")

    max_wr = float(ss.get("max_win_rate", 0.85))
    min_wr = float(ss.get("min_win_rate", 0.35))
    if win_rate is None:
        ok = False
        reasons.append("win_rate:missing")
    else:
        max_wr_req = _apply_factor_max(max_wr, factor)
        min_wr_req = _apply_factor_min(min_wr, factor)
        if win_rate > max_wr_req:
            ok = False
            reasons.append(f"win_rate:{win_rate:.4f}>{max_wr_req:.4f}")
        if win_rate < min_wr_req:
            ok = False
            reasons.append(f"win_rate:{win_rate:.4f}<{min_wr_req:.4f}")

    min_pf = float(perf.get("min_profit_factor", 1.1))
    pf_req = _apply_factor_min(min_pf, factor)
    if profit_factor is None:
        ok = False
        reasons.append("profit_factor:missing")
    elif profit_factor < pf_req:
        ok = False
        reasons.append(f"profit_factor:{profit_factor:.4f}<{pf_req:.4f}")

    if total_pnl <= 0:
        ok = False
        reasons.append(f"total_pnl:{total_pnl:.2f}<=0")

    # Execution mode sanity: ensure no disallowed modes exist.
    allowed_modes = dq.get("allowed_execution_modes")
    allowed = set(str(m).strip().lower() for m in (allowed_modes or []) if str(m).strip())
    if allowed:
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT COALESCE(execution_mode,'') FROM trade_executions")
            modes = {str(r[0] or "").strip().lower() for r in cur.fetchall()}
            modes = {m for m in modes if m}
            disallowed = sorted(m for m in modes if m not in allowed)
            if disallowed:
                ok = False
                reasons.append(f"execution_modes_disallowed:{','.join(disallowed)}")
        except Exception as exc:
            ok = False
            reasons.append(f"execution_mode_check_failed:{exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Entry/exit matching: require all closes to have an entry_trade_id.
    if bool(audit.get("require_entry_exit_matching", True)):
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM trade_executions "
                "WHERE is_close=1 AND entry_trade_id IS NULL"
            )
            unlinked_closes = int(cur.fetchone()[0] or 0)
            if unlinked_closes > 0:
                ok = False
                reasons.append(f"unlinked_closes:{unlinked_closes}>0")
        except Exception as exc:
            ok = False
            reasons.append(f"entry_exit_check_failed:{exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    status = "PASS" if ok else "FAIL"
    return {"status": status, "reasons": reasons, "metrics": metrics, "requirements": req}


def _forecaster_eval(
    *,
    suite_path: Path,
    monitor_config: Path,
    factor: float,
    max_age_hours: float,
) -> dict[str, Any]:
    """
    Evaluate the most recent adversarial suite output.
    If missing/stale, returns status UNKNOWN.
    """
    if not suite_path.exists():
        return {"status": "UNKNOWN", "reasons": [f"suite_missing:{suite_path}"], "breaches": [], "age_seconds": None}

    try:
        age_seconds = max(0.0, time.time() - suite_path.stat().st_mtime)
    except Exception:
        age_seconds = None

    if age_seconds is not None and max_age_hours > 0 and age_seconds > float(max_age_hours) * 3600.0:
        return {"status": "UNKNOWN", "reasons": [f"suite_stale:{age_seconds:.0f}s"], "breaches": [], "age_seconds": age_seconds}

    payload = _read_json(suite_path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}

    if not thresholds:
        raw = _load_yaml(monitor_config)
        fm = raw.get("forecaster_monitoring") if isinstance(raw.get("forecaster_monitoring"), dict) else {}
        rmse = fm.get("regression_metrics") if isinstance(fm.get("regression_metrics"), dict) else {}
        suite_cfg = rmse.get("adversarial_suite") if isinstance(rmse.get("adversarial_suite"), dict) else {}
        thresholds = {
            "max_ensemble_under_best_rate": float(suite_cfg.get("max_ensemble_under_best_rate", 1.0)),
            "max_avg_ensemble_ratio_vs_best": float(suite_cfg.get("max_avg_ensemble_ratio_vs_best", 1.2)),
            "max_ensemble_worse_than_rw_rate": float(suite_cfg.get("max_ensemble_worse_than_rw_rate", 0.3)),
            "require_zero_errors": bool(suite_cfg.get("require_zero_errors", True)),
        }

    breaches: list[str] = []
    max_under_best = _apply_factor_max(float(thresholds.get("max_ensemble_under_best_rate", 1.0)), factor)
    max_ratio = _apply_factor_max(float(thresholds.get("max_avg_ensemble_ratio_vs_best", 1.2)), factor)
    max_worse_rw = _apply_factor_max(float(thresholds.get("max_ensemble_worse_than_rw_rate", 0.3)), factor)
    require_zero_errors = bool(thresholds.get("require_zero_errors", True))

    for variant, row in summary.items():
        if not isinstance(row, dict):
            continue
        errors = int(row.get("errors", 0) or 0)
        under_best = float(row.get("ensemble_under_best_rate", 0.0) or 0.0)
        ratio = row.get("avg_ensemble_ratio_vs_best")
        ratio = float(ratio) if isinstance(ratio, (int, float)) else None
        worse_rw = float(row.get("ensemble_worse_than_rw_rate", 0.0) or 0.0)

        if require_zero_errors and errors > 0:
            breaches.append(f"{variant}:errors={errors}")
        if under_best > max_under_best:
            breaches.append(f"{variant}:under_best={under_best:.4f}>{max_under_best:.4f}")
        if ratio is not None and ratio > max_ratio:
            breaches.append(f"{variant}:ratio_vs_best={ratio:.4f}>{max_ratio:.4f}")
        if worse_rw > max_worse_rw:
            breaches.append(f"{variant}:worse_rw={worse_rw:.4f}>{max_worse_rw:.4f}")

    status = "PASS" if not breaches else "FAIL"
    return {
        "status": status,
        "reasons": [],
        "breaches": breaches,
        "age_seconds": age_seconds,
        "thresholds_effective": {
            "max_ensemble_under_best_rate": max_under_best,
            "max_avg_ensemble_ratio_vs_best": max_ratio,
            "max_ensemble_worse_than_rw_rate": max_worse_rw,
            "require_zero_errors": require_zero_errors,
        },
        "suite_path": str(suite_path),
    }


def _launch_training_detached(
    *,
    python_bin: str,
    profile: str,
    target: str,
    output_json: str,
    log_file: Path,
) -> subprocess.Popen:
    cmd = [
        python_bin,
        str(PROJECT_ROOT / "scripts" / "run_training_priority_cycle.py"),
        "--profile",
        profile,
        "--target",
        target,
        "--continue-on-error",
        "--output-json",
        output_json,
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("a", encoding="utf-8", errors="replace")
    try:
        # Avoid inheriting OpenClaw's stdin; detach from the cron session.
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            creationflags=creationflags,
            close_fds=(os.name != "nt"),
        )
        return proc
    finally:
        try:
            handle.close()
        except Exception:
            pass


def _emit_autopilot_feedback(*, payload: dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = TRAINING_FEEDBACK_DIR / f"{ts}_openclaw_training_autopilot.json"
    try:
        _write_json(out_path, payload)
    except Exception:
        return


def _emit_training_proposal(*, payload: dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = TRAINING_PROPOSALS_DIR / f"{ts}_{_safe_filename('training_autopilot')}.json"
    try:
        _write_json(out_path, payload)
    except Exception:
        return


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG))
    parser.add_argument("--profit-requirements", default=str(DEFAULT_PROFIT_REQUIREMENTS))
    parser.add_argument("--suite-output", default=str(DEFAULT_SUITE_OUTPUT))
    parser.add_argument("--suite-max-age-hours", type=float, default=48.0)
    parser.add_argument("--benchmark-factor", type=float, default=0.989)
    parser.add_argument("--profile", choices=["forecasters", "llm", "all"], default="all")
    parser.add_argument("--target", default="local_cron")
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=720,
        help="Minimum minutes between training starts when any benchmark fails (default: 12h).",
    )
    parser.add_argument(
        "--profitability-only-cooldown-minutes",
        type=int,
        default=4320,
        help="Minimum minutes between training starts when ONLY profitability fails (default: 3d).",
    )
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--training-output-json", default="logs/automation/training_priority/training_autopilot_latest.json")
    parser.add_argument("--training-log-file", default="logs/automation/training_autopilot/training_autopilot_latest.log")
    parser.add_argument("--max-log-tail-lines", type=int, default=60)
    args = parser.parse_args(argv)

    factor = float(args.benchmark_factor)
    if factor <= 0:
        print("ERROR: benchmark-factor must be > 0")
        return 2

    db_path = Path(args.db)
    monitor_cfg = Path(args.monitor_config)
    req_path = Path(args.profit_requirements)
    suite_path = Path(args.suite_output)
    state_path = Path(args.state_file)
    training_output_json = str(args.training_output_json)
    training_log_file = Path(args.training_log_file)

    started_at = _utc_now_iso()
    run_id = f"training_autopilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    profit = _profitability_eval(db_path=db_path, requirements_path=req_path, factor=factor)
    forecaster = _forecaster_eval(
        suite_path=suite_path,
        monitor_config=monitor_cfg,
        factor=factor,
        max_age_hours=float(args.suite_max_age_hours),
    )

    profit_ok = profit.get("status") == "PASS"
    forecaster_ok = forecaster.get("status") == "PASS"

    decision: dict[str, Any] = {
        "type": "training_autopilot_feedback",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "benchmark_factor": factor,
        "profitability": profit,
        "forecaster": forecaster,
        "action": None,
        "state_file": str(state_path),
    }

    if profit_ok and forecaster_ok:
        decision["action"] = "NOOP_BENCHMARKS_MET"
        decision["completed_at_utc"] = _utc_now_iso()
        _emit_autopilot_feedback(payload=decision)
        print("NO_REPLY")
        return 0

    # Load last state and check for running training.
    state = _read_json(state_path)
    last_pid = int(state.get("pid", 0) or 0) if isinstance(state, dict) else 0
    last_started = str(state.get("started_at_utc") or "") if isinstance(state, dict) else ""
    last_started_ts = None
    try:
        last_started_ts = datetime.fromisoformat(last_started.replace("Z", "+00:00")) if last_started else None
    except Exception:
        last_started_ts = None

    if last_pid and _pid_is_running(last_pid):
        decision["action"] = "NOOP_TRAINING_ALREADY_RUNNING"
        decision["training_pid"] = last_pid
        decision["completed_at_utc"] = _utc_now_iso()
        _emit_autopilot_feedback(payload=decision)
        print("NO_REPLY")
        return 0

    # Cooldown logic: avoid restarting heavy training too frequently.
    now = datetime.now(timezone.utc)
    minutes_since_last = None
    if last_started_ts is not None:
        try:
            minutes_since_last = (now - last_started_ts).total_seconds() / 60.0
        except Exception:
            minutes_since_last = None

    only_profitability_fails = (not profit_ok) and forecaster_ok
    cooldown = int(args.profitability_only_cooldown_minutes if only_profitability_fails else args.cooldown_minutes)
    if minutes_since_last is not None and minutes_since_last < float(cooldown):
        decision["action"] = "NOOP_COOLDOWN"
        decision["cooldown_minutes"] = cooldown
        decision["minutes_since_last_start"] = minutes_since_last
        decision["completed_at_utc"] = _utc_now_iso()
        _emit_autopilot_feedback(payload=decision)
        print("NO_REPLY")
        return 0

    # Decide training profile.
    profile = str(args.profile)
    if profile not in {"forecasters", "llm", "all"}:
        profile = "all"

    reason_parts: list[str] = []
    if not forecaster_ok:
        reason_parts.append("forecaster")
    if not profit_ok:
        reason_parts.append("profitability")
    reason = "+".join(reason_parts) or "unknown"

    proposal = {
        "type": "training_proposal",
        "proposal_id": run_id,
        "proposed_at_utc": _utc_now_iso(),
        "benchmark_factor": factor,
        "reason": reason,
        "training": {
            "profile": profile,
            "target": str(args.target),
            "output_json": training_output_json,
            "log_file": str(training_log_file),
        },
        "notes": (
            "Cron-safe autopilot launched a detached training run. "
            "Detailed task-level proposals/feedback are written by scripts/run_training_priority_cycle.py."
        ),
    }
    _emit_training_proposal(payload=proposal)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = training_log_file
    if "{ts}" in str(log_file):
        log_file = Path(str(log_file).replace("{ts}", ts))

    try:
        proc = _launch_training_detached(
            python_bin=sys.executable,
            profile=profile,
            target=str(args.target),
            output_json=training_output_json,
            log_file=log_file,
        )
    except Exception as exc:
        decision["action"] = "FAIL_LAUNCH_EXCEPTION"
        decision["error"] = str(exc)
        decision["completed_at_utc"] = _utc_now_iso()
        _emit_autopilot_feedback(payload=decision)
        print(f"ERROR: training launch failed: {exc}")
        return 1

    # Persist state for restart-safe behavior.
    new_state = {
        "run_id": run_id,
        "pid": int(proc.pid or 0),
        "started_at_utc": _utc_now_iso(),
        "reason": reason,
        "profile": profile,
        "target": str(args.target),
        "benchmark_factor": factor,
        "training_output_json": training_output_json,
        "training_log_file": str(log_file),
    }
    try:
        _write_json(state_path, new_state)
    except Exception:
        pass

    decision["action"] = "STARTED_TRAINING"
    decision["training_pid"] = int(proc.pid or 0)
    decision["training_log_file"] = str(log_file)
    decision["completed_at_utc"] = _utc_now_iso()
    decision["stdout_tail_redacted"] = _redact_text(_tail_file(log_file, int(args.max_log_tail_lines)))
    _emit_autopilot_feedback(payload=decision)

    print(f"STARTED_TRAINING reason={reason} profile={profile} pid={int(proc.pid or 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
