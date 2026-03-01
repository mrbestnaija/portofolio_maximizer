"""
Detect and optionally remove duplicate forecast audit windows.

Duplicate windows have the same (ticker, start, end, length, horizon) fingerprint.
Keeps the newest file per fingerprint group; older files are the duplicates.
Writes a summary JSON to logs/ensemble_health/ for trend monitoring.

Usage:
    python scripts/dedupe_audit_windows.py [--audit-dir logs/forecast_audits] [--apply]

Exit codes:
    0 = no duplicates found
    1 = duplicates found (warning; not CI-blocking)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_HEALTH_DIR = REPO_ROOT / "logs" / "ensemble_health"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _window_fingerprint(audit: dict) -> str:
    """Hash (ticker, start, end, length, horizon) to identify duplicate windows."""
    ds = audit.get("dataset", {})
    summary = audit.get("summary", {})
    return hashlib.sha1(
        json.dumps(
            {
                "ticker": ds.get("ticker"),
                "start": ds.get("start"),
                "end": ds.get("end"),
                "length": ds.get("length"),
                "horizon": summary.get("forecast_horizon") or ds.get("forecast_horizon"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def scan(audit_dir: Path) -> dict[str, dict]:
    """
    Scan audit JSONs and group by window fingerprint.

    Returns:
        {fingerprint: {"keep": Path, "remove": [Path, ...]}}
        Only includes fingerprints that have more than one file.
    """
    audit_dir = Path(audit_dir)
    groups: dict[str, list[Path]] = {}
    for json_file in sorted(audit_dir.glob("forecast_audit_*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Skipping malformed JSON %s: %s", json_file.name, exc)
            continue
        fp = _window_fingerprint(data)
        groups.setdefault(fp, []).append(json_file)

    duplicates: dict[str, dict] = {}
    for fp, paths in groups.items():
        if len(paths) <= 1:
            continue
        # Keep newest (sort by mtime descending, then filename descending as tiebreaker)
        paths_sorted = sorted(
            paths,
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        duplicates[fp] = {
            "keep": paths_sorted[0],
            "remove": paths_sorted[1:],
        }
    return duplicates


def report(duplicates: dict[str, dict]) -> None:
    if not duplicates:
        log.info("[OK] No duplicate audit windows found.")
        return
    total_remove = sum(len(v["remove"]) for v in duplicates.values())
    log.warning(
        "[WARN] %d duplicate group(s) found (%d file(s) eligible for removal):",
        len(duplicates),
        total_remove,
    )
    for fp, entry in duplicates.items():
        log.warning(
            "  Hash %s: KEEP=%s | REMOVE=%s",
            fp[:8],
            entry["keep"].name,
            [p.name for p in entry["remove"]],
        )


def apply_removals(duplicates: dict[str, dict]) -> int:
    """Remove duplicate files. Returns count of removed files."""
    removed = 0
    for entry in duplicates.values():
        for path in entry["remove"]:
            try:
                os.remove(path)
                log.info("Removed: %s", path.name)
                removed += 1
            except OSError as exc:
                log.error("Failed to remove %s: %s", path.name, exc)
    return removed


def write_summary(duplicate_count: int, removed_count: int) -> None:
    """Write summary JSON to logs/ensemble_health/ for trend monitoring."""
    ENSEMBLE_HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    summary_path = ENSEMBLE_HEALTH_DIR / f"dedupe_summary_{date_str}.json"
    summary = {
        "date": date_str,
        "duplicate_count": duplicate_count,
        "removed_count": removed_count,
    }
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Summary written: %s", summary_path)
    except OSError as exc:
        log.warning("Could not write summary: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and optionally remove duplicate forecast audit windows."
    )
    parser.add_argument(
        "--audit-dir",
        default=str(REPO_ROOT / "logs" / "forecast_audits"),
        help="Directory containing forecast_audit_*.json files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicate files (default: dry-run)",
    )
    args = parser.parse_args(argv)

    audit_dir = Path(args.audit_dir)
    if not audit_dir.exists():
        log.error("Audit directory not found: %s", audit_dir)
        return 1

    duplicates = scan(audit_dir)
    duplicate_count = sum(len(v["remove"]) for v in duplicates.values())
    report(duplicates)

    removed_count = 0
    if duplicates and args.apply:
        removed_count = apply_removals(duplicates)
        log.info("[DONE] Removed %d duplicate file(s).", removed_count)
    elif duplicates and not args.apply:
        log.info(
            "[DRY-RUN] Pass --apply to delete %d duplicate file(s).", duplicate_count
        )

    write_summary(duplicate_count, removed_count)
    return 1 if duplicates else 0


if __name__ == "__main__":
    sys.exit(main())
