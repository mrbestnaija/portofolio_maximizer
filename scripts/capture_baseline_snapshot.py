#!/usr/bin/env python3
"""
Capture a baseline snapshot of configs, code, and key run artefacts.

This script is intentionally read-only with respect to DB/trading state: it
copies files and writes a manifest under reports/ so baseline comparisons are
repeatable across roadmap phases.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_PATHS = (
    Path("config/yfinance_config.yml"),
    Path("config/execution_cost_model.yml"),
    Path("config/signal_routing_config.yml"),
    Path("config/quant_success_config.yml"),
    Path("config/forecaster_monitoring.yml"),
)

DEFAULT_CODE_PATHS = (
    Path("scripts/run_auto_trader.py"),
    Path("models/time_series_signal_generator.py"),
    Path("execution/paper_trading_engine.py"),
)

RUN_SUMMARY_PATH = Path("logs/automation/run_summary.jsonl")
EXECUTION_LOG_PATH = Path("logs/automation/execution_log.jsonl")
QUANT_LOG_PATH = Path("logs/signals/quant_validation.jsonl")
DASHBOARD_DATA_PATH = Path("visualizations/dashboard_data.json")
DASHBOARD_PNG_PATH = Path("visualizations/dashboard_snapshot.png")


@dataclass(frozen=True)
class CaptureResult:
    snapshot_dir: Path
    manifest_path: Path
    run_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_lines(path: Path, n_lines: int) -> list[str]:
    if n_lines <= 0:
        return []

    # Efficient tail for potentially large files.
    # Read backwards in blocks and stop once enough newlines are found.
    block_size = 8192
    data = bytearray()
    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        offset = file_size
        while offset > 0 and data.count(b"\n") <= n_lines:
            read_size = min(block_size, offset)
            offset -= read_size
            handle.seek(offset)
            data[:0] = handle.read(read_size)
            if offset == 0:
                break
    lines = data.splitlines()[-n_lines:]
    return [ln.decode("utf-8", errors="replace") for ln in lines]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_copy(src: Path, dst: Path) -> Optional[Dict[str, Any]]:
    if not src.exists():
        return None
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": _sha256_file(dst),
    }


def _read_last_jsonl_record(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def _infer_run_id(root: Path) -> Optional[str]:
    record = _read_last_jsonl_record(root / RUN_SUMMARY_PATH)
    if isinstance(record, dict) and record.get("run_id"):
        return str(record["run_id"])
    return None


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return cleaned or "baseline"


def _git_metadata(root: Path) -> Dict[str, Any]:
    if not (root / ".git").exists():
        return {"available": False}

    def _run(args: list[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                args,
                cwd=root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    return {
        "available": True,
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def capture_baseline_snapshot(
    *,
    root: Path,
    out_dir: Path,
    tag: str = "baseline",
    run_id: Optional[str] = None,
    include_code: bool = True,
    execution_log_tail_lines: int = 10,
    extra_config_paths: Optional[Iterable[Path]] = None,
    extra_code_paths: Optional[Iterable[Path]] = None,
) -> CaptureResult:
    root = root.resolve()
    out_dir = out_dir.resolve()
    tag_sanitized = _sanitize_tag(tag)

    resolved_run_id = run_id or _infer_run_id(root) or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    snapshot_dir = out_dir / f"{resolved_run_id}_{tag_sanitized}"
    suffix = 1
    while snapshot_dir.exists():
        snapshot_dir = out_dir / f"{resolved_run_id}_{tag_sanitized}_{suffix}"
        suffix += 1
    _safe_mkdir(snapshot_dir)

    manifest: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "run_id": resolved_run_id,
        "tag": tag_sanitized,
        "python": {"executable": sys.executable, "version": sys.version},
        "git": _git_metadata(root),
        "files": {"configs": [], "code": [], "artifacts": []},
        "missing": {"configs": [], "code": [], "artifacts": []},
    }

    config_paths = list(DEFAULT_CONFIG_PATHS)
    if extra_config_paths:
        config_paths.extend(extra_config_paths)

    for rel in config_paths:
        src = root / rel
        dst = snapshot_dir / "configs" / rel.name
        copied = _safe_copy(src, dst)
        if copied is None:
            manifest["missing"]["configs"].append(str(rel))
        else:
            manifest["files"]["configs"].append(copied)

    if include_code:
        code_paths = list(DEFAULT_CODE_PATHS)
        if extra_code_paths:
            code_paths.extend(extra_code_paths)
        for rel in code_paths:
            src = root / rel
            dst = snapshot_dir / "code" / rel.name
            copied = _safe_copy(src, dst)
            if copied is None:
                manifest["missing"]["code"].append(str(rel))
            else:
                manifest["files"]["code"].append(copied)

    # Run summary (last record)
    run_summary_record = _read_last_jsonl_record(root / RUN_SUMMARY_PATH)
    if run_summary_record is None:
        manifest["missing"]["artifacts"].append(str(RUN_SUMMARY_PATH))
    else:
        out_path = snapshot_dir / "artifacts" / "run_summary_last.json"
        _safe_mkdir(out_path.parent)
        out_path.write_text(json.dumps(run_summary_record, indent=2, sort_keys=True), encoding="utf-8")
        manifest["files"]["artifacts"].append(
            {
                "src": str(root / RUN_SUMMARY_PATH),
                "dst": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": _sha256_file(out_path),
            }
        )

    # Execution log tail
    exec_log = root / EXECUTION_LOG_PATH
    if not exec_log.exists():
        manifest["missing"]["artifacts"].append(str(EXECUTION_LOG_PATH))
    else:
        tail = _tail_lines(exec_log, execution_log_tail_lines)
        out_path = snapshot_dir / "artifacts" / "execution_log_tail.jsonl"
        _safe_mkdir(out_path.parent)
        out_path.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")
        manifest["files"]["artifacts"].append(
            {
                "src": str(exec_log),
                "dst": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": _sha256_file(out_path),
            }
        )

    # Quant validation summary (best-effort)
    quant_log = root / QUANT_LOG_PATH
    if not quant_log.exists():
        manifest["missing"]["artifacts"].append(str(QUANT_LOG_PATH))
    else:
        out_buf = io.StringIO()
        try:
            from scripts.summarize_quant_validation import (
                _load_monitoring_thresholds,
                load_entries,
                summarize,
            )

            entries = load_entries(quant_log)
            monitoring_cfg = _load_monitoring_thresholds(root / Path("config/forecaster_monitoring.yml"))
            with redirect_stdout(out_buf):
                summarize(entries, monitoring_cfg=monitoring_cfg)
        except SystemExit as exc:
            out_buf.write(f"Quant validation summary unavailable: {exc}\n")
        except Exception as exc:  # pragma: no cover - best effort
            out_buf.write(f"Quant validation summary failed: {exc}\n")

        out_path = snapshot_dir / "artifacts" / "quant_validation_summary.txt"
        _safe_mkdir(out_path.parent)
        out_path.write_text(out_buf.getvalue(), encoding="utf-8")
        manifest["files"]["artifacts"].append(
            {
                "src": str(quant_log),
                "dst": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": _sha256_file(out_path),
            }
        )

    # Dashboard outputs
    for rel in (DASHBOARD_DATA_PATH, DASHBOARD_PNG_PATH):
        src = root / rel
        dst = snapshot_dir / "artifacts" / rel.name
        copied = _safe_copy(src, dst)
        if copied is None:
            manifest["missing"]["artifacts"].append(str(rel))
        else:
            manifest["files"]["artifacts"].append(copied)

    # Optional horizon backtest report (latest), when present.
    reports_dir = root / "reports"
    if reports_dir.exists():
        candidates = sorted(
            reports_dir.glob("horizon_backtest_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            src = candidates[0]
            dst = snapshot_dir / "artifacts" / "horizon_backtest_latest.json"
            copied = _safe_copy(src, dst)
            if copied is not None:
                manifest["files"]["artifacts"].append(copied)

    # Optional per-run DB provenance payload.
    provenance = root / "logs" / "automation" / f"db_provenance_{resolved_run_id}.json"
    copied = _safe_copy(provenance, snapshot_dir / "artifacts" / provenance.name)
    if copied is None:
        manifest["missing"]["artifacts"].append(str(provenance.relative_to(root)))
    else:
        manifest["files"]["artifacts"].append(copied)

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return CaptureResult(snapshot_dir=snapshot_dir, manifest_path=manifest_path, run_id=resolved_run_id)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture baseline snapshot artefacts for roadmap comparisons.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root (defaults to auto-detected).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/baselines"),
        help="Output directory for snapshots (relative to root if not absolute).",
    )
    parser.add_argument("--tag", default="baseline", help="Snapshot tag/label (folder suffix).")
    parser.add_argument("--run-id", default=None, help="Optional run_id override (defaults to last run_summary record).")
    parser.add_argument("--no-code", action="store_true", help="Do not copy code files into the snapshot.")
    parser.add_argument(
        "--execution-log-tail-lines",
        type=int,
        default=10,
        help="How many execution_log.jsonl lines to capture.",
    )

    args = parser.parse_args(argv)
    root = args.root
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    result = capture_baseline_snapshot(
        root=root,
        out_dir=out_dir,
        tag=args.tag,
        run_id=args.run_id,
        include_code=not args.no_code,
        execution_log_tail_lines=args.execution_log_tail_lines,
    )
    print(f"Baseline snapshot captured under {result.snapshot_dir}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
