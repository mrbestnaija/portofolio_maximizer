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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        default="data/portfolio_maximizer.db",
        help="Path to SQLite database (default: data/portfolio_maximizer.db).",
    )
    parser.add_argument(
        "--audit-dir",
        default="logs/forecast_audits",
        help="Forecast audit directory (default: logs/forecast_audits).",
    )
    parser.add_argument(
        "--monitor-config",
        default="config/forecaster_monitoring.yml",
        help="Forecaster monitoring config path (default: config/forecaster_monitoring.yml).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Max forecast audit files to scan (default: 500).",
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
        "--require-profitable",
        action="store_true",
        help="Require strictly positive PnL (in addition to proof validity).",
    )
    parser.add_argument(
        "--output-json",
        default="logs/audit_gate/production_gate_latest.json",
        help="Output path for latest gate artifact.",
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
    args = parser.parse_args()
    python_bin = str(Path(args.python_bin))
    db_path = _resolve_path(repo_root, args.db)
    audit_dir = _resolve_path(repo_root, args.audit_dir)
    monitor_config = _resolve_path(repo_root, args.monitor_config)
    output_path = _resolve_path(repo_root, args.output_json)

    check_script = repo_root / "scripts" / "check_forecast_audits.py"
    proof_script = repo_root / "scripts" / "validate_profitability_proof.py"
    summary_cache_path = repo_root / "logs" / "forecast_audits_cache" / "latest_summary.json"

    lift_cmd = [
        python_bin,
        str(check_script),
        "--audit-dir",
        str(audit_dir),
        "--config-path",
        str(monitor_config),
        "--max-files",
        str(args.max_files),
    ]
    if args.require_holding_period:
        lift_cmd.append("--require-holding-period")

    lift_proc = _run_command(lift_cmd, cwd=repo_root)
    lift_output = f"{lift_proc.stdout or ''}\n{lift_proc.stderr or ''}".strip()
    lift_inconclusive = "RMSE gate inconclusive" in lift_output

    lift_summary = _safe_load_json(summary_cache_path) or {}
    if lift_summary and not _summary_matches_audit_dir(lift_summary, audit_dir):
        lift_summary = {}

    lift_status = "PASS"
    if lift_proc.returncode != 0:
        lift_status = "FAIL"
    elif lift_inconclusive:
        lift_status = "INCONCLUSIVE"

    lift_pass = lift_proc.returncode == 0 and (
        args.allow_inconclusive_lift or not lift_inconclusive
    )

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
    proof_pass = proof_is_valid and (proof_is_profitable if args.require_profitable else True)
    proof_status = "PASS" if proof_pass and proof_proc.returncode == 0 else "FAIL"

    metrics = proof_payload.get("metrics") if isinstance(proof_payload.get("metrics"), dict) else {}
    winning = int(metrics.get("winning_trades", 0) or 0)
    losing = int(metrics.get("losing_trades", 0) or 0)

    gate_pass = lift_pass and proof_pass
    gate_status = "PASS" if gate_pass else "FAIL"

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stamped_output = output_path.parent / f"{output_path.stem}_{stamp}{output_path.suffix}"

    payload: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "repo_state": _collect_git_state(repo_root),
        "inputs": {
            "db": str(db_path),
            "audit_dir": str(audit_dir),
            "monitor_config": str(monitor_config),
            "max_files": int(args.max_files),
            "require_holding_period": bool(args.require_holding_period),
            "allow_inconclusive_lift": bool(args.allow_inconclusive_lift),
            "require_profitable": bool(args.require_profitable),
        },
        "lift_gate": {
            "status": lift_status,
            "pass": lift_pass,
            "exit_code": int(lift_proc.returncode),
            "inconclusive": lift_inconclusive,
            "decision": lift_summary.get("decision"),
            "decision_reason": lift_summary.get("decision_reason"),
            "effective_audits": lift_summary.get("effective_audits"),
            "violation_rate": lift_summary.get("violation_rate"),
            "max_violation_rate": lift_summary.get("max_violation_rate"),
            "lift_fraction": lift_summary.get("lift_fraction"),
            "min_lift_fraction": lift_summary.get("min_lift_fraction"),
            "output_tail": _tail_lines(lift_output),
        },
        "profitability_proof": {
            "status": proof_status,
            "pass": proof_pass,
            "command_exit_code": int(proof_proc.returncode),
            "is_proof_valid": proof_is_valid,
            "is_profitable": proof_is_profitable,
            "total_pnl": metrics.get("total_pnl"),
            "profit_factor": metrics.get("profit_factor"),
            "win_rate": metrics.get("win_rate"),
            "closed_trades": winning + losing,
            "trading_days": metrics.get("trading_days"),
            "violations": proof_payload.get("violations", []),
            "warnings": proof_payload.get("warnings", []),
            "recommendations": proof_payload.get("recommendations", []),
            "output_tail": _tail_lines(f"{proof_proc.stdout or ''}\n{proof_proc.stderr or ''}"),
        },
        "production_profitability_gate": {
            "status": gate_status,
            "pass": gate_pass,
        },
    }

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
    print(f"Gate status    : {gate_status}")
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
