"""
Verify EXP-R5-001 residual experiment activation in forecast audits.

This status script also reports the canonical residual summary sidecar:
  visualizations/performance/residual_experiment_summary.json

Exit codes:
  0 -> at least one active residual artifact was found
  1 -> no active residual artifact (inactive/not fitted/not run)
  2 -> argument or IO error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CANONICAL_SUMMARY_PATH = Path("visualizations/performance/residual_experiment_summary.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_residual_block(doc: dict[str, Any]) -> dict[str, Any] | None:
    artifacts = doc.get("artifacts")
    if isinstance(artifacts, dict):
        candidate = artifacts.get("residual_experiment")
        if isinstance(candidate, dict):
            return candidate
    candidate = doc.get("residual_experiment")
    return candidate if isinstance(candidate, dict) else None


def _fmt_field(val: Any) -> str:
    if val is None:
        return "None"
    if isinstance(val, list):
        size = len(val)
        preview = [f"{v:.4f}" for v in val[:3] if isinstance(v, (int, float))]
        return f"[{', '.join(preview)}{'...' if size > 3 else ''}] (len={size})"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _read_summary(path: Path = CANONICAL_SUMMARY_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {
            "summary_path": str(path),
            "summary_exists": False,
            "status": None,
            "reason_code": None,
            "n_windows_with_residual_metrics": None,
        }
    return {
        "summary_path": str(path),
        "summary_exists": True,
        "status": payload.get("status"),
        "reason_code": payload.get("reason_code"),
        "n_windows_with_residual_metrics": payload.get("n_windows_with_residual_metrics"),
    }


def _print_report(path: Path, block: dict[str, Any], base: Path) -> None:
    try:
        short = str(path.relative_to(base))
    except ValueError:
        short = path.name
    status = block.get("residual_status", "UNKNOWN")
    marker = "[ACTIVE]" if status == "active" else f"[{str(status).upper()}]"
    print(f"\n{marker}  {short}")
    ordered_keys = [
        "residual_status",
        "residual_active",
        "reason",
        "n_corrected",
        "residual_mean",
        "residual_std",
        "y_hat_anchor",
        "y_hat_residual_ensemble",
        "rmse_anchor",
        "rmse_residual_ensemble",
        "rmse_ratio",
        "da_anchor",
        "da_residual_ensemble",
        "phase",
        "experiment_id",
        "anchor_model_id",
    ]
    for key in ordered_keys:
        if key in block:
            print(f"    {key:<35} {_fmt_field(block[key])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify EXP-R5-001 residual activation in audit files."
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("logs/forecast_audits"),
        help="Directory containing forecast_audit_*.json files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Report all matching audit files (default: latest file only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    if not args.audit_dir.exists():
        print(f"[ERROR] audit_dir does not exist: {args.audit_dir}", file=sys.stderr)
        return 2

    files = sorted(args.audit_dir.glob("forecast_audit_*.json"))
    if not files:
        files = sorted(args.audit_dir.glob("*.json"))
    summary = _read_summary()
    if not files:
        if args.emit_json:
            print(
                json.dumps(
                    {
                        "active": False,
                        "n_active": 0,
                        "n_inactive": 0,
                        "n_missing_block": 0,
                        "n_scanned": 0,
                        "summary": summary,
                        "results": [],
                    },
                    indent=2,
                )
            )
        else:
            print(f"[WARN] No audit JSON files found in {args.audit_dir}", file=sys.stderr)
        return 1

    targets = files if args.show_all else [files[-1]]
    n_active = 0
    n_inactive = 0
    n_missing = 0
    results: list[dict[str, Any]] = []

    for path in targets:
        doc = _load_json(path)
        if doc is None:
            n_missing += 1
            continue
        block = _extract_residual_block(doc)
        if block is None:
            n_missing += 1
            continue
        status = block.get("residual_status", "unknown")
        if status == "active":
            n_active += 1
        else:
            n_inactive += 1
        results.append({"path": str(path), "status": status, "block": block})
        if not args.emit_json:
            _print_report(path, block, args.audit_dir)

    if args.emit_json:
        print(
            json.dumps(
                {
                    "active": n_active > 0,
                    "n_active": n_active,
                    "n_inactive": n_inactive,
                    "n_missing_block": n_missing,
                    "n_scanned": len(targets),
                    "summary": summary,
                    "results": [{"path": r["path"], "status": r["status"]} for r in results],
                },
                indent=2,
            )
        )
    else:
        print(f"\n{'=' * 60}")
        print(f"Scanned : {len(targets)} file(s)")
        print(f"Active  : {n_active}")
        print(f"Inactive: {n_inactive}")
        print(f"Missing : {n_missing}")
        print(
            "Summary : "
            f"{summary['summary_path']} "
            f"(exists={summary['summary_exists']}, status={summary['status']}, "
            f"reason={summary['reason_code']}, windows={summary['n_windows_with_residual_metrics']})"
        )
        if n_active > 0:
            print("\n[OK] EXP-R5-001 is ACTIVE - residual corrections are being emitted.")
            print("     Next: run_quality_pipeline.py --enable-residual-experiment")
        else:
            print("\n[INFO] No active residual artifact found.")
            print("       To activate: set residual_experiment.enabled: true")
            print("       in config/forecasting_config.yml then run a forecast pipeline.")

    return 0 if n_active > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
