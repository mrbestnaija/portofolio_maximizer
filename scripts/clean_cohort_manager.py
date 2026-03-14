#!/usr/bin/env python3
"""
Freeze and operate a clean evidence cohort without mutating legacy production evidence.

This helper is intentionally narrow:
- freeze an immutable cohort identity
- emit a PowerShell activation script for the cohort
- run a cohort-scoped proof loop with explicit audit/output paths
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
from typing import Any, Dict, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from utils.evidence_io import atomic_write_json, load_json_file


CONTRACT_VERSION = 2
DEFAULT_COHORT_ROOT = ROOT_PATH / "logs" / "forecast_audits" / "cohorts"
DEFAULT_REPLAY_ROOT = ROOT_PATH / "logs" / "evidence_replay"


def _stable_fingerprint(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_files(paths: list[Path]) -> Optional[str]:
    pieces: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.relative_to(ROOT_PATH)
        except ValueError:
            rel = path
        pieces.append(str(rel))
        pieces.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if not pieces:
        return None
    return hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()


def _default_build_fingerprint() -> str:
    for name in ("PMX_BUILD_FINGERPRINT", "GIT_COMMIT"):
        candidate = str(os.environ.get(name) or "").strip()
        if candidate:
            return candidate
    return "workspace_uncommitted"


def _cohort_paths(cohort_id: str, cohort_root: Path) -> Dict[str, Path]:
    root = cohort_root / cohort_id
    return {
        "root": root,
        "production_dir": root / "production",
        "research_dir": root / "research",
        "identity_path": root / "cohort_identity.json",
        "activation_path": root / "activate_clean_cohort.ps1",
        "proof_output_path": root / "proof_loop_latest.json",
        "gate_output_path": root / "production_gate_latest.json",
    }


def build_cohort_identity(
    *,
    cohort_id: str,
    build_fingerprint: Optional[str] = None,
    routing_mode: str = "explicit_env",
    contract_version: int = CONTRACT_VERSION,
    config_paths: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    if config_paths is None:
        config_paths = [
            ROOT_PATH / "config" / "forecasting_config.yml",
            ROOT_PATH / "config" / "signal_routing_config.yml",
        ]
    identity = {
        "cohort_id": cohort_id,
        "build_fingerprint": str(build_fingerprint or _default_build_fingerprint()).strip(),
        "contract_version": int(contract_version),
        "routing_mode": routing_mode,
        "strategy_config_fingerprint": _fingerprint_files(config_paths),
    }
    identity["contract_fingerprint"] = _stable_fingerprint(identity)
    return identity


def freeze_clean_cohort(
    *,
    cohort_id: str,
    cohort_root: Path = DEFAULT_COHORT_ROOT,
    build_fingerprint: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    paths = _cohort_paths(cohort_id, cohort_root)
    identity = build_cohort_identity(cohort_id=cohort_id, build_fingerprint=build_fingerprint)

    existing = load_json_file(paths["identity_path"]) if paths["identity_path"].exists() else None
    if isinstance(existing, dict):
        existing_identity = existing.get("cohort_identity")
        if isinstance(existing_identity, dict):
            if build_fingerprint is None and not force:
                identity = existing_identity
            elif existing_identity.get("contract_fingerprint") != identity.get("contract_fingerprint"):
                if not force:
                    raise ValueError(
                        "existing cohort fingerprint differs; refusing to mutate frozen cohort identity"
                    )
            else:
                identity = existing_identity

    for key in ("root", "production_dir", "research_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_identity": identity,
        "paths": {
            "root": str(paths["root"]),
            "production_dir": str(paths["production_dir"]),
            "research_dir": str(paths["research_dir"]),
        },
        "env": {
            "PMX_EVIDENCE_COHORT_ID": cohort_id,
            "PMX_BUILD_FINGERPRINT": identity.get("build_fingerprint"),
            "TS_FORECAST_AUDIT_DIR": str(paths["production_dir"]),
        },
    }
    atomic_write_json(paths["identity_path"], payload)

    activation_script = "\n".join(
        [
            f"$env:PMX_EVIDENCE_COHORT_ID='{cohort_id}'",
            f"$env:PMX_BUILD_FINGERPRINT='{identity.get('build_fingerprint')}'",
            f"$env:TS_FORECAST_AUDIT_DIR='{paths['production_dir']}'",
            "",
        ]
    )
    paths["activation_path"].write_text(activation_script, encoding="utf-8")
    return payload


def _run_command(
    cmd: list[str],
    *,
    env: Dict[str, str],
    cwd: Path = ROOT_PATH,
) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "passed": int(proc.returncode) == 0,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def run_clean_cohort_proof_loop(
    *,
    cohort_id: str,
    cohort_root: Path = DEFAULT_COHORT_ROOT,
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    include_global_gates: bool = False,
    build_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    frozen = freeze_clean_cohort(
        cohort_id=cohort_id,
        cohort_root=cohort_root,
        build_fingerprint=build_fingerprint,
    )
    paths = _cohort_paths(cohort_id, cohort_root)
    identity = frozen["cohort_identity"]

    env = os.environ.copy()
    env.update(
        {
            "PMX_EVIDENCE_COHORT_ID": cohort_id,
            "PMX_BUILD_FINGERPRINT": str(identity.get("build_fingerprint") or ""),
            "TS_FORECAST_AUDIT_DIR": str(paths["production_dir"]),
        }
    )

    replay_dir = replay_root / cohort_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    results: Dict[str, Dict[str, Any]] = {}
    results["replay_trade_evidence_chain"] = _run_command(
        [
            python,
            "scripts/replay_trade_evidence_chain.py",
            "--scenario",
            "happy_path",
            "--output-dir",
            str(replay_dir),
            "--json",
        ],
        env=env,
    )
    results["pnl_integrity_enforcer"] = _run_command(
        [python, "-m", "integrity.pnl_integrity_enforcer"],
        env=env,
    )
    results["production_audit_gate_clean_cohort"] = _run_command(
        [
            python,
            "scripts/production_audit_gate.py",
            "--audit-dir",
            str(paths["production_dir"]),
            "--output",
            str(paths["gate_output_path"]),
            "--unattended-profile",
        ],
        env=env,
    )
    if include_global_gates:
        results["run_all_gates_global"] = _run_command(
            [python, "scripts/run_all_gates.py", "--json"],
            env=env,
        )

    overall_passed = all(bool(result.get("passed")) for result in results.values())
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_identity": identity,
        "paths": {
            "cohort_root": str(paths["root"]),
            "production_dir": str(paths["production_dir"]),
            "production_gate_output": str(paths["gate_output_path"]),
            "proof_output": str(paths["proof_output_path"]),
            "replay_dir": str(replay_dir),
        },
        "include_global_gates": bool(include_global_gates),
        "overall_passed": overall_passed,
        "steps": results,
    }
    atomic_write_json(paths["proof_output_path"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="Freeze a clean cohort identity.")
    freeze_parser.add_argument("--cohort-id", required=True)
    freeze_parser.add_argument("--cohort-root", default=str(DEFAULT_COHORT_ROOT))
    freeze_parser.add_argument("--build-fingerprint", default=None)
    freeze_parser.add_argument("--force", action="store_true")
    freeze_parser.add_argument("--json", action="store_true", dest="emit_json")

    proof_parser = subparsers.add_parser("proof-loop", help="Run the clean cohort proof loop.")
    proof_parser.add_argument("--cohort-id", required=True)
    proof_parser.add_argument("--cohort-root", default=str(DEFAULT_COHORT_ROOT))
    proof_parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    proof_parser.add_argument("--include-global-gates", action="store_true")
    proof_parser.add_argument("--build-fingerprint", default=None)
    proof_parser.add_argument("--json", action="store_true", dest="emit_json")

    args = parser.parse_args()

    if args.command == "freeze":
        payload = freeze_clean_cohort(
            cohort_id=args.cohort_id,
            cohort_root=Path(args.cohort_root),
            build_fingerprint=args.build_fingerprint,
            force=bool(args.force),
        )
        if args.emit_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Frozen cohort   : {args.cohort_id}")
            print(f"Identity path   : {payload['paths']['root']}")
            print(f"Production dir  : {payload['paths']['production_dir']}")
        return 0

    summary = run_clean_cohort_proof_loop(
        cohort_id=args.cohort_id,
        cohort_root=Path(args.cohort_root),
        replay_root=Path(args.replay_root),
        include_global_gates=bool(args.include_global_gates),
        build_fingerprint=args.build_fingerprint,
    )
    if args.emit_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Clean cohort    : {args.cohort_id}")
        print(f"Overall passed  : {int(bool(summary['overall_passed']))}")
        print(f"Proof output    : {summary['paths']['proof_output']}")
    return 0 if bool(summary["overall_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
