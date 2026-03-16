#!/usr/bin/env python3
"""Institutional unattended-run readiness gate.

Phased checks that enforce security, operational consistency, and repo hygiene.
This is designed to fail fast on regressions that can destabilize autonomous runs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Finding:
    phase: str
    check: str
    status: str  # PASS | WARN | FAIL
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
        }


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def _has_all(text: str, needles: Iterable[str]) -> bool:
    return all(n in text for n in needles)


def _phase_p0_security() -> List[Finding]:
    out: List[Finding] = []

    llm_src = _read("scripts/llm_multi_model_orchestrator.py")
    if _has_all(
        llm_src,
        [
            "PMX_ALLOW_RUNTIME_PIP_INSTALL",
            "Direct URL/VCS package specs are disabled",
            "must start with https://",
            "PMX_RUNTIME_PIP_ALLOWED_INDEX_HOSTS",
        ],
    ):
        out.append(Finding("P0", "runtime_install_policy", "PASS", "Runtime pip install guardrails present."))
    else:
        out.append(Finding("P0", "runtime_install_policy", "FAIL", "Runtime pip install guardrails missing."))

    autonomy_src = _read("utils/openclaw_cli.py")
    if 'OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS", default=True' in autonomy_src:
        out.append(Finding("P0", "prompt_injection_default", "PASS", "Prompt-injection block defaults to enabled."))
    else:
        out.append(Finding("P0", "prompt_injection_default", "FAIL", "Prompt-injection block is not default-on."))

    checkpoint_src = _read("etl/checkpoint_manager.py")
    if _has_all(
        checkpoint_src,
        [
            "_validate_checkpoint_id",
            "_safe_pickle_load",
            "Checkpoint path escapes checkpoint_dir",
        ],
    ):
        out.append(Finding("P0", "checkpoint_load_safety", "PASS", "Checkpoint ID/path validation and safe unpickling present."))
    else:
        out.append(Finding("P0", "checkpoint_load_safety", "FAIL", "Checkpoint hardening missing."))

    return out


def _phase_p1_operational() -> List[Finding]:
    out: List[Finding] = []

    check_src = _read("scripts/check_forecast_audits.py")
    prod_src = _read("scripts/production_audit_gate.py")
    if "FORECAST_AUDIT_MAX_FILES_DEFAULT" in check_src and "FORECAST_AUDIT_MAX_FILES_DEFAULT" in prod_src:
        out.append(Finding("P1", "gate_max_files_repo_sync", "PASS", "Shared max-files default contract in place."))
    else:
        out.append(Finding("P1", "gate_max_files_repo_sync", "FAIL", "Forecast/audit gate max-files defaults are not repo-synced."))

    forecaster_src = _read("forcester_ts/forecaster.py")
    if "_maybe_reweight_ensemble_from_holdout" in forecaster_src:
        out.append(
            Finding(
                "P1",
                "holdout_leakage_block",
                "FAIL",
                "Forecaster evaluate() still performs post-hoc holdout reweighting.",
            )
        )
    else:
        out.append(
            Finding(
                "P1",
                "holdout_leakage_block",
                "PASS",
                "Forecaster evaluate() is read-only (no post-hoc holdout reweighting).",
            )
        )

    if (
        "max_missing_ensemble_rate" in check_src
        and "manifest_integrity_mode" in check_src
        and "ensemble or sarimax or garch or samossa" not in check_src
    ):
        out.append(
            Finding(
                "P1",
                "audit_evidence_hardening",
                "PASS",
                "Forecast audit gate enforces manifest + missing-ensemble contracts.",
            )
        )
    else:
        out.append(
            Finding(
                "P1",
                "audit_evidence_hardening",
                "FAIL",
                "Forecast audit gate still allows fallback/missing provenance contracts.",
            )
        )

    ci_cfg = _read("config/forecaster_monitoring_ci.yml")
    if all(
        token in ci_cfg
        for token in (
            "max_ensemble_under_best_rate: 0.35",
            "manifest_integrity_mode: fail",
            "max_missing_ensemble_rate: 0.00",
        )
    ):
        out.append(
            Finding(
                "P1",
                "ci_overfit_poison_thresholds",
                "PASS",
                "CI thresholds enforce overfit and provenance hardening contracts.",
            )
        )
    else:
        out.append(
            Finding(
                "P1",
                "ci_overfit_poison_thresholds",
                "FAIL",
                "CI thresholds are too permissive for overfit/provenance risks.",
            )
        )

    refresh_src = _read("scripts/run_overnight_refresh.py")
    required = [
        r'rc = py\("scripts/check_quant_validation_health.py".*?errors \+= 1',
        r'rc = py\("scripts/quant_validation_headroom.py".*?errors \+= 1',
        r'rc = py\("scripts/production_audit_gate.py".*?errors \+= 1',
    ]
    if all(re.search(pat, refresh_src, re.DOTALL) for pat in required):
        out.append(Finding("P1", "overnight_error_accounting", "PASS", "Final health checks increment error counter on failures."))
    else:
        out.append(Finding("P1", "overnight_error_accounting", "FAIL", "Final health failures can be dropped from overnight error count."))

    return out


def _phase_p2_platt_data() -> List[Finding]:
    out: List[Finding] = []
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "platt_contract_audit.py"), "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    merged = f"{proc.stdout}\n{proc.stderr}"
    if "No module named 'models'" in merged:
        out.append(Finding("P2", "platt_contract_bootstrap", "FAIL", "platt_contract_audit.py requires manual PYTHONPATH setup."))
        return out

    if proc.returncode == 2:
        out.append(Finding("P2", "platt_contract_bootstrap", "FAIL", "platt_contract_audit.py crashed at runtime."))
        return out

    if proc.returncode not in {0, 1}:
        out.append(
            Finding(
                "P2",
                "platt_contract_bootstrap",
                "FAIL",
                f"Unexpected platt_contract_audit exit code: {proc.returncode}",
            )
        )
        return out

    try:
        parsed = json.loads(proc.stdout or "[]")
    except Exception as exc:
        out.append(
            Finding(
                "P2",
                "platt_contract_bootstrap",
                "FAIL",
                f"Unable to parse platt_contract_audit JSON output: {exc}",
            )
        )
        return out

    if not isinstance(parsed, list):
        out.append(Finding("P2", "platt_contract_bootstrap", "FAIL", "platt_contract_audit JSON output is not a list."))
        return out
    if not parsed:
        out.append(Finding("P2", "platt_contract_bootstrap", "FAIL", "platt_contract_audit returned empty findings list."))
        return out

    findings = parsed
    failing = [f for f in findings if str(f.get("status")) == "FAIL"]
    if failing:
        out.append(Finding("P2", "platt_contract_bootstrap", "FAIL", f"Platt contract failures: {len(failing)}"))
    else:
        out.append(Finding("P2", "platt_contract_bootstrap", "PASS", "Platt contract checks are executable and passing/warning only."))
    return out


def _phase_p3_repo_hygiene() -> List[Finding]:
    out: List[Finding] = []
    shadow_files = [
        ROOT / "Dockerfile (1)",
        ROOT / "execution" / "lob_simulator (1).py",
    ]
    existing = [str(p.relative_to(ROOT)) for p in shadow_files if p.exists()]
    if existing:
        out.append(Finding("P3", "shadow_duplicate_files", "FAIL", f"Shadow duplicates present: {existing}"))
    else:
        out.append(Finding("P3", "shadow_duplicate_files", "PASS", "No tracked shadow duplicates found."))
    return out


def _phase_p4_prior_gate_verification() -> List[Finding]:
    """BYP-03 fix: verify that run_all_gates.py ran recently and overall_passed=True.

    Reads logs/gate_status_latest.json written by run_all_gates.py on each run.
    Fail-closed if:
      - The artifact does not exist
      - overall_passed is False in the artifact
      - The artifact timestamp is more than 26 hours old
    """
    out: List[Finding] = []
    artifact = ROOT / "logs" / "gate_status_latest.json"

    if not artifact.exists():
        out.append(Finding(
            "P4", "prior_gate_execution",
            "FAIL",
            "logs/gate_status_latest.json not found. run_all_gates.py has not produced "
            "a verifiable status artifact for unattended validation.",
        ))
        return out

    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        out.append(Finding(
            "P4", "prior_gate_execution",
            "FAIL",
            f"Could not parse logs/gate_status_latest.json: {exc}",
        ))
        return out

    # Check overall_passed
    overall_passed = bool(data.get("overall_passed", False))
    ts_str = data.get("timestamp_utc", "")

    # Check freshness
    age_ok = False
    age_detail = "timestamp unknown"
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - ts
            age_hours = age.total_seconds() / 3600
            age_ok = age_hours <= 26
            age_detail = f"{age_hours:.1f}h ago"
        except Exception:
            age_detail = "unparseable timestamp"

    if not overall_passed:
        out.append(Finding(
            "P4", "prior_gate_execution",
            "FAIL",
            f"run_all_gates.py last result: overall_passed=False ({age_detail}). "
            "Fix failing gates before autonomous runs.",
        ))
    elif not age_ok:
        out.append(Finding(
            "P4", "prior_gate_execution",
            "FAIL",
            f"run_all_gates.py status artifact is stale ({age_detail}, limit: 26h). "
            "Run gates before next autonomous cycle.",
        ))
    else:
        out.append(Finding(
            "P4", "prior_gate_execution",
            "PASS",
            f"run_all_gates.py: overall_passed=True, run {age_detail}.",
        ))

    return out


def run_gate() -> List[Finding]:
    findings: List[Finding] = []
    findings.extend(_phase_p0_security())
    findings.extend(_phase_p1_operational())
    findings.extend(_phase_p2_platt_data())
    findings.extend(_phase_p3_repo_hygiene())
    findings.extend(_phase_p4_prior_gate_verification())
    return findings


def _print_report(findings: List[Finding]) -> None:
    print("=" * 72)
    print("INSTITUTIONAL UNATTENDED-RUN READINESS GATE")
    print("=" * 72)
    for f in findings:
        print(f"[{f.status}] [{f.phase}] {f.check}")
        print(f"       {f.detail}")
    n_fail = sum(1 for f in findings if f.status == "FAIL")
    n_warn = sum(1 for f in findings if f.status == "WARN")
    n_pass = sum(1 for f in findings if f.status == "PASS")
    print("-" * 72)
    print(f"PASS={n_pass} WARN={n_warn} FAIL={n_fail}")
    print("[RESULT] PASS" if n_fail == 0 else "[RESULT] FAIL")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Institutional unattended-run readiness gate.")
    parser.add_argument("--json", action="store_true", help="Emit JSON findings.")
    args = parser.parse_args(argv)

    findings = run_gate()
    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        _print_report(findings)
    return 1 if any(f.status == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
