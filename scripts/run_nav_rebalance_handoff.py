#!/usr/bin/env python3
"""run_nav_rebalance_handoff.py

Thin auto-apply handoff wrapper for the NAV rebalance sidecar.

Responsibilities:
- Read the latest shadow-first NAV rebalance plan.
- Check rollout.live_apply_allowed.
- If allowed, call scripts/apply_nav_reallocation.py to materialize the allocation
  artifact and staged barbell config.
- If not allowed, no-op and emit a status artifact explaining the blockers.

This wrapper is intentionally narrow: it does not change thresholds, gate
semantics, or routing policy. It only bridges the plan artifact to the apply
step once the plan itself says live apply is allowed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import click

from utils.evidence_io import write_versioned_json_artifact

DEFAULT_PLAN_PATH = ROOT_PATH / "logs" / "automation" / "nav_rebalance_plan_latest.json"
DEFAULT_CONFIG_PATH = ROOT_PATH / "config" / "barbell.yml"
DEFAULT_OUTPUT_PATH = ROOT_PATH / "logs" / "automation" / "nav_allocation_latest.json"
DEFAULT_STAGED_CONFIG_PATH = ROOT_PATH / "config" / "barbell.staged.yml"
DEFAULT_STATUS_PATH = ROOT_PATH / "logs" / "automation" / "nav_rebalance_handoff_latest.json"
DEFAULT_STATUS_HISTORY_ROOT = ROOT_PATH / "logs" / "automation" / "nav_rebalance_handoff_history"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_rollout(payload: dict[str, Any]) -> dict[str, Any]:
    rollout = payload.get("rollout")
    if isinstance(rollout, dict):
        return rollout
    rollout = payload.get("_rollout")
    if isinstance(rollout, dict):
        return rollout
    return {}


def _tail_text(text: str | None, *, lines: int = 30) -> str:
    if not text:
        return ""
    parts = text.splitlines()
    if len(parts) <= lines:
        return text
    return "\n".join(parts[-lines:])


def _validate_status(payload: dict[str, Any]) -> bool | tuple[bool, str]:
    required = {
        "generated_utc",
        "plan_path",
        "config_path",
        "status",
        "action_taken",
        "exit_code",
        "live_apply_allowed",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        return False, f"missing_required_keys:{','.join(missing)}"
    return True, "ok"


def _write_status_artifact(
    *,
    status_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return write_versioned_json_artifact(
        latest_path=status_path,
        payload=payload,
        archive_name="nav_rebalance_handoff",
        archive_root=DEFAULT_STATUS_HISTORY_ROOT,
        validate_fn=_validate_status,
    )


def run_nav_rebalance_handoff(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    staged_config_path: Optional[Path] = DEFAULT_STAGED_CONFIG_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    config_path = Path(config_path)
    output_path = Path(output_path)
    staged_config_path = Path(staged_config_path) if staged_config_path else None
    status_path = Path(status_path)

    plan = _load_json(plan_path)
    rollout = _load_rollout(plan)
    blockers = [str(item) for item in rollout.get("live_apply_blockers") or [] if str(item).strip()]
    live_apply_allowed = bool(rollout.get("live_apply_allowed", False))

    payload: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "plan_path": str(plan_path),
        "config_path": str(config_path),
        "output_path": str(output_path),
        "staged_config_path": str(staged_config_path) if staged_config_path else None,
        "status": "BLOCKED",
        "action_taken": "NO_OP",
        "exit_code": 0,
        "live_apply_allowed": live_apply_allowed,
        "rollout_mode": rollout.get("mode"),
        "gate_lift_candidate": bool(rollout.get("gate_lift_candidate", False)),
        "gate_lift_ready": bool(rollout.get("gate_lift_ready", False)),
        "blockers": blockers,
        "plan_summary": plan.get("summary") if isinstance(plan.get("summary"), dict) else {},
        "plan_evidence_contract": plan.get("evidence_contract") if isinstance(plan.get("evidence_contract"), dict) else {},
        "apply_rc": None,
        "apply_stdout_tail": "",
        "apply_stderr_tail": "",
        "apply_command": [],
    }

    if not blockers and not live_apply_allowed:
        blockers = ["live_apply_not_allowed"]
        payload["blockers"] = blockers

    if not plan_path.exists():
        payload["status"] = "BLOCKED"
        payload["exit_code"] = 1
        payload["blockers"] = sorted({*payload["blockers"], "plan_missing"})
        payload["note"] = "NAV rebalance plan not found; skipping handoff."
        _write_status_artifact(status_path=status_path, payload=payload)
        return payload

    if not live_apply_allowed:
        payload["status"] = "NO_OP"
        payload["exit_code"] = 0
        payload["note"] = "live_apply_allowed is false; handoff remains inert."
        _write_status_artifact(status_path=status_path, payload=payload)
        return payload

    if not config_path.exists():
        payload["status"] = "BLOCKED"
        payload["exit_code"] = 1
        payload["blockers"] = sorted({*payload["blockers"], "config_missing"})
        payload["note"] = "Barbell config not found; cannot stage allocation."
        _write_status_artifact(status_path=status_path, payload=payload)
        return payload

    apply_script = ROOT_PATH / "scripts" / "apply_nav_reallocation.py"
    cmd = [
        str(sys.executable),
        str(apply_script),
        "--plan-path",
        str(plan_path),
        "--config-path",
        str(config_path),
        "--output",
        str(output_path),
    ]
    if staged_config_path:
        cmd.extend(["--staged-config", str(staged_config_path)])

    run = subprocess.run(cmd, capture_output=True, text=True)
    payload["apply_command"] = cmd
    payload["apply_rc"] = int(run.returncode)
    payload["apply_stdout_tail"] = _tail_text(run.stdout)
    payload["apply_stderr_tail"] = _tail_text(run.stderr)

    if run.returncode == 0:
        payload["status"] = "APPLIED"
        payload["action_taken"] = "APPLY"
        payload["exit_code"] = 0
        payload["note"] = "live_apply_allowed true; allocation materialized."
    else:
        payload["status"] = "ERROR"
        payload["action_taken"] = "APPLY_FAILED"
        payload["exit_code"] = int(run.returncode)
        payload["blockers"] = sorted({*payload["blockers"], "apply_subprocess_failed"})
        payload["note"] = "apply_nav_reallocation.py returned non-zero."

    _write_status_artifact(status_path=status_path, payload=payload)
    return payload


@click.command()
@click.option("--plan-path", default=str(DEFAULT_PLAN_PATH), show_default=True)
@click.option("--config-path", default=str(DEFAULT_CONFIG_PATH), show_default=True)
@click.option("--output", default=str(DEFAULT_OUTPUT_PATH), show_default=True)
@click.option("--staged-config", default=str(DEFAULT_STAGED_CONFIG_PATH), show_default=True)
@click.option("--status-output", default=str(DEFAULT_STATUS_PATH), show_default=True)
@click.option("--json", "emit_json", is_flag=True, default=False, help="Print the handoff status as JSON.")
def main(
    plan_path: str,
    config_path: str,
    output: str,
    staged_config: str,
    status_output: str,
    emit_json: bool,
) -> None:
    result = run_nav_rebalance_handoff(
        plan_path=Path(plan_path),
        config_path=Path(config_path),
        output_path=Path(output),
        staged_config_path=Path(staged_config) if staged_config else None,
        status_path=Path(status_output),
    )

    if emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            "NAV rebalance handoff status={status} action={action} live_apply_allowed={allowed}".format(
                status=result["status"],
                action=result["action_taken"],
                allowed=result["live_apply_allowed"],
            )
        )
        if result.get("blockers"):
            print(f"  blockers={result['blockers']}")
        if result.get("apply_rc") is not None:
            print(f"  apply_rc={result['apply_rc']}")

    raise SystemExit(int(result["exit_code"]))


if __name__ == "__main__":
    main()
