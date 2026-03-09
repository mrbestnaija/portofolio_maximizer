"""
Single-source truth snapshot for EXP-R5-001 activation state.

This status script enforces the canonical residual summary path:
  visualizations/performance/residual_experiment_summary.json

Contradictions (exit 1):
1) active audit(s) exist but summary status is SKIP
2) active audit(s) exist but summary measured windows is 0

Exit codes:
  0 -> no contradictions
  1 -> contradiction(s) detected
  2 -> argument/io error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

CANONICAL_SUMMARY_PATH = Path("visualizations/performance/residual_experiment_summary.json")
DEFAULT_AUDIT_DIR = Path("logs/forecast_audits")
FORECASTING_CONFIG_PATH = Path("config/forecasting_config.yml")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_residual_enabled(cfg: dict[str, Any] | None) -> bool | None:
    if not isinstance(cfg, dict):
        return None
    forecasting = cfg.get("forecasting")
    if isinstance(forecasting, dict):
        residual = forecasting.get("residual_experiment")
        if isinstance(residual, dict) and "enabled" in residual:
            return bool(residual.get("enabled"))
    residual_root = cfg.get("residual_experiment")
    if isinstance(residual_root, dict) and "enabled" in residual_root:
        return bool(residual_root.get("enabled"))
    return None


def _scan_residual_audits(audit_dir: Path) -> dict[str, Any]:
    files = sorted(audit_dir.glob("forecast_audit_*.json"))
    active_paths: list[str] = []
    inactive_paths: list[str] = []
    missing_block = 0
    unreadable = 0

    for path in files:
        payload = _read_json(path)
        if payload is None:
            unreadable += 1
            continue
        artifacts = payload.get("artifacts")
        block: dict[str, Any] | None = None
        if isinstance(artifacts, dict):
            candidate = artifacts.get("residual_experiment")
            if isinstance(candidate, dict):
                block = candidate
        if block is None:
            candidate = payload.get("residual_experiment")
            if isinstance(candidate, dict):
                block = candidate
        if block is None:
            missing_block += 1
            continue

        status = block.get("residual_status")
        if status == "active":
            active_paths.append(str(path))
        else:
            inactive_paths.append(str(path))

    return {
        "n_scanned": len(files),
        "n_active": len(active_paths),
        "n_inactive": len(inactive_paths),
        "n_missing_block": missing_block,
        "n_unreadable": unreadable,
        "latest_active_path": active_paths[-1] if active_paths else None,
        "latest_inactive_path": inactive_paths[-1] if inactive_paths else None,
    }


def build_truth_snapshot(
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    summary_path: Path = CANONICAL_SUMMARY_PATH,
    forecasting_cfg_path: Path = FORECASTING_CONFIG_PATH,
) -> dict[str, Any]:
    cfg = _read_yaml(forecasting_cfg_path) if forecasting_cfg_path.exists() else None
    cfg_enabled = _extract_residual_enabled(cfg)

    summary = _read_json(summary_path) if summary_path.exists() else None
    summary_status = summary.get("status") if isinstance(summary, dict) else None
    summary_reason = summary.get("reason_code") if isinstance(summary, dict) else None
    n_windows = summary.get("n_windows_with_residual_metrics") if isinstance(summary, dict) else None
    if not isinstance(n_windows, int):
        n_windows = None
    n_realized_windows = (
        summary.get("n_windows_with_realized_residual_metrics") if isinstance(summary, dict) else None
    )
    if not isinstance(n_realized_windows, int):
        n_realized_windows = None
    n_structural_windows = (
        summary.get("n_windows_structural_only_metrics") if isinstance(summary, dict) else None
    )
    if not isinstance(n_structural_windows, int):
        n_structural_windows = None
    m2_review_ready = summary.get("m2_review_ready") if isinstance(summary, dict) else None
    if not isinstance(m2_review_ready, bool):
        m2_review_ready = None

    audits = _scan_residual_audits(audit_dir)
    contradictions: list[str] = []
    if audits["n_active"] > 0 and summary_status == "SKIP":
        contradictions.append("ACTIVE_AUDITS_BUT_SUMMARY_SKIP")
    if audits["n_active"] > 0 and n_windows == 0:
        contradictions.append("ACTIVE_AUDITS_BUT_ZERO_MEASURED_WINDOWS")

    return {
        "ok": len(contradictions) == 0,
        "canonical_summary_path": str(summary_path),
        "forecasting_config_path": str(forecasting_cfg_path),
        "residual_experiment_enabled": cfg_enabled,
        "summary_exists": summary is not None,
        "summary_status": summary_status,
        "summary_reason_code": summary_reason,
        "n_windows_with_residual_metrics": n_windows,
        "n_windows_with_realized_residual_metrics": n_realized_windows,
        "n_windows_structural_only_metrics": n_structural_windows,
        "m2_review_ready": m2_review_ready,
        "audits": audits,
        "active": audits["n_active"] > 0,
        "contradictions": contradictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-source EXP-R5-001 truth snapshot with contradiction checks."
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="Audit directory to scan (default: logs/forecast_audits).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    if not args.audit_dir.exists():
        print(f"[ERROR] audit_dir not found: {args.audit_dir}", file=sys.stderr)
        return 2

    snapshot = build_truth_snapshot(audit_dir=args.audit_dir)
    if args.emit_json:
        print(json.dumps(snapshot, indent=2))
    else:
        print("=== EXP-R5-001 Truth Snapshot ===")
        print(f"config_enabled: {snapshot['residual_experiment_enabled']}")
        print(f"summary_path  : {snapshot['canonical_summary_path']}")
        print(
            "summary       : "
            f"{snapshot['summary_status']} / {snapshot['summary_reason_code']} "
            f"(windows={snapshot['n_windows_with_residual_metrics']}, "
            f"realized={snapshot['n_windows_with_realized_residual_metrics']}, "
            f"structural_only={snapshot['n_windows_structural_only_metrics']}, "
            f"m2_ready={snapshot['m2_review_ready']})"
        )
        audits = snapshot["audits"]
        print(
            "audits        : "
            f"active={audits['n_active']} inactive={audits['n_inactive']} "
            f"missing={audits['n_missing_block']} unreadable={audits['n_unreadable']}"
        )
        if snapshot["contradictions"]:
            print("contradictions:")
            for code in snapshot["contradictions"]:
                print(f"  - {code}")
        else:
            print("contradictions: none")

    return 0 if snapshot["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
