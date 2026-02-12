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
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
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

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

