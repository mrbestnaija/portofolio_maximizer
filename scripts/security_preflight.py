#!/usr/bin/env python3
"""
security_preflight.py
---------------------

Run security-oriented dependency checks before production workflows:
1) `pip check` for broken dependency state.
2) Optional `pip-audit` CVE scan when module is available.

Outputs a machine-readable JSON artifact for each run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: List[str], timeout: int) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _tail(text: str, limit: int = 30) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-limit:])


def _has_pip_audit(python_bin: str) -> bool:
    code = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('pip_audit') else 1)"
    rc, _, _ = _run([python_bin, "-c", code], timeout=15)
    return rc == 0


def _parse_pip_audit_json(stdout: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Returns: (packages_scanned, vulnerabilities_found)
    """
    text = (stdout or "").strip()
    if not text:
        return 0, 0, []

    parsed: Any
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
        # pip-audit may append human-readable lines after JSON.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start < 0 or end <= start:
                continue
            try:
                parsed = json.loads(text[start : end + 1])
                break
            except Exception:
                continue
        if parsed is None:
            return 0, 0, []

    vuln_count = 0
    pkg_count = 0
    vulnerable_packages: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        deps = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("dependencies"), list):
        deps = parsed["dependencies"]
    else:
        return 0, 0, []

    for item in deps:
        if not isinstance(item, dict):
            continue
        pkg_count += 1
        vulns = item.get("vulns")
        if isinstance(vulns, list):
            vuln_count += len(vulns)
            if vulns:
                vuln_ids = []
                for vuln in vulns:
                    if isinstance(vuln, dict):
                        vuln_id = vuln.get("id")
                        if vuln_id:
                            vuln_ids.append(str(vuln_id))
                vulnerable_packages.append(
                    {
                        "name": item.get("name"),
                        "version": item.get("version"),
                        "vulnerability_count": len(vulns),
                        "vulnerability_ids": vuln_ids,
                    }
                )
    return pkg_count, vuln_count, vulnerable_packages


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dependency + CVE preflight checks.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for checks.")
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--caller", default="", help="Calling script label.")
    parser.add_argument("--run-id", default="", help="Run ID for traceability.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Timeout per check command.")
    parser.add_argument("--require-pip-audit", action="store_true", help="Fail if pip-audit module is unavailable.")
    parser.add_argument(
        "--ignore-vuln-id",
        action="append",
        default=[],
        help="Vulnerability ID to ignore (repeatable). Example: --ignore-vuln-id CVE-2026-24486",
    )
    parser.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail on security check failures.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Best-effort mode; never fail process.")
    args = parser.parse_args()

    python_bin = str(Path(args.python_bin).expanduser())

    checks: Dict[str, Any] = {}
    warnings: List[str] = []
    failures: List[str] = []

    # Baseline interpreter metadata
    checks["python"] = {
        "executable": python_bin,
        "version": sys.version.split()[0],
        "timestamp_utc": _utc_now(),
    }

    # 1) pip check (dependency consistency)
    pip_check_cmd = [python_bin, "-m", "pip", "check"]
    pip_rc, pip_out, pip_err = _run(pip_check_cmd, timeout=args.timeout_seconds)
    pip_ok = pip_rc == 0
    checks["pip_check"] = {
        "ok": pip_ok,
        "exit_code": pip_rc,
        "output_tail": _tail(f"{pip_out}\n{pip_err}"),
    }
    if not pip_ok:
        failures.append("pip_check_failed")

    # 2) CVE scan with pip-audit when available
    has_pip_audit = _has_pip_audit(python_bin)
    cve_check: Dict[str, Any] = {
        "available": has_pip_audit,
        "ran": False,
        "ok": True,
        "exit_code": 0,
        "packages_scanned": 0,
        "vulnerabilities_found": 0,
        "output_tail": "",
    }
    if has_pip_audit:
        audit_cmd = [python_bin, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"]
        audit_rc, audit_out, audit_err = _run(audit_cmd, timeout=args.timeout_seconds)
        pkg_count, vuln_count, vulnerable_packages = _parse_pip_audit_json(audit_out)
        ignored_ids = {str(v).strip().upper() for v in args.ignore_vuln_id if str(v).strip()}
        effective_vulnerable_packages: List[Dict[str, Any]] = []
        ignored_vulnerabilities: List[Dict[str, Any]] = []
        effective_vuln_count = 0

        for pkg in vulnerable_packages:
            pkg_ids = [str(v).strip().upper() for v in pkg.get("vulnerability_ids", []) if str(v).strip()]
            kept_ids = [v for v in pkg_ids if v not in ignored_ids]
            ignored_for_pkg = [v for v in pkg_ids if v in ignored_ids]
            if ignored_for_pkg:
                ignored_vulnerabilities.append(
                    {
                        "name": pkg.get("name"),
                        "version": pkg.get("version"),
                        "ignored_vulnerability_ids": ignored_for_pkg,
                    }
                )
            if kept_ids:
                effective_vulnerable_packages.append(
                    {
                        "name": pkg.get("name"),
                        "version": pkg.get("version"),
                        "vulnerability_count": len(kept_ids),
                        "vulnerability_ids": kept_ids,
                    }
                )
                effective_vuln_count += len(kept_ids)

        scan_completed = bool(pkg_count > 0 or audit_rc == 0)
        cve_ok = scan_completed and effective_vuln_count == 0
        cve_check.update(
            {
                "ran": True,
                "ok": cve_ok,
                "exit_code": audit_rc,
                "scan_completed": scan_completed,
                "packages_scanned": pkg_count,
                "vulnerabilities_found_raw": vuln_count,
                "vulnerabilities_found": effective_vuln_count,
                "ignored_vulnerability_ids": sorted(ignored_ids),
                "ignored_vulnerabilities": ignored_vulnerabilities[:25],
                "vulnerable_packages": effective_vulnerable_packages[:25],
                "output_tail": _tail(f"{audit_out}\n{audit_err}"),
            }
        )
        if not cve_ok:
            failures.append("pip_audit_failed_or_vulnerabilities_found")
    else:
        msg = "pip_audit_not_installed"
        cve_check["ok"] = False if args.require_pip_audit else True
        warnings.append(msg)
        if args.require_pip_audit:
            failures.append(msg)

    checks["pip_audit"] = cve_check

    passed = len(failures) == 0
    payload: Dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "caller": str(args.caller or ""),
        "run_id": str(args.run_id or ""),
        "strict": bool(args.strict),
        "require_pip_audit": bool(args.require_pip_audit),
        "passed": passed,
        "warnings": warnings,
        "failures": failures,
        "checks": checks,
    }

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("=== Security Preflight ===")
    print(f"Caller         : {payload['caller']}")
    print(f"Run ID         : {payload['run_id']}")
    print(f"pip check      : {'PASS' if pip_ok else 'FAIL'}")
    if has_pip_audit:
        print(
            f"pip-audit      : {'PASS' if cve_check['ok'] else 'FAIL'} "
            f"(vulns={cve_check['vulnerabilities_found']}, pkgs={cve_check['packages_scanned']})"
        )
    else:
        print("pip-audit      : MISSING (warning)")
    print(f"Overall        : {'PASS' if passed else 'FAIL'}")

    if args.strict and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
