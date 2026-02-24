"""
update_platt_outcomes.py — Platt scaling outcome reconciliation.

Reads logs/signals/quant_validation.jsonl, queries trade_executions for closed
trades matching each entry's signal_id, and writes an 'outcome' field back into
the JSONL.  The file is rewritten atomically (temp file + os.replace).

Usage:
    python scripts/update_platt_outcomes.py [--db PATH] [--log PATH] [--dry-run]

Exit codes:
    0  success (even if 0 entries were updated)
    1  DB not found or JSONL parse error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Phase 7.13-C1: central path constants (fallback gracefully if etl not in sys.path)
try:
    from etl.paths import DB_PATH as _DEFAULT_DB_PATH, QUANT_VALIDATION_JSONL as _DEFAULT_JSONL_PATH
except ImportError:
    _DEFAULT_DB_PATH = Path("data/portfolio_maximizer.db")
    _DEFAULT_JSONL_PATH = Path("logs/signals/quant_validation.jsonl")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=5.0)


def _fetch_outcomes_for_signals(
    conn: sqlite3.Connection,
    signal_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return {ts_signal_id: outcome_dict} for closed trades matching any of the given ids.

    Phase 7.13-A2: queries ts_signal_id TEXT column (globally unique per TS signal).
    When multiple closing trades share the same ts_signal_id (partial fills),
    the one with the largest abs(realized_pnl) is used.
    """
    if not signal_ids:
        return {}

    placeholders = ",".join("?" * len(signal_ids))
    query = f"""
        SELECT ts_signal_id, realized_pnl, realized_pnl_pct
        FROM trade_executions
        WHERE ts_signal_id IN ({placeholders})
          AND is_close = 1
          AND realized_pnl IS NOT NULL
        ORDER BY ABS(realized_pnl) DESC
    """
    try:
        cur = conn.cursor()
        cur.execute(query, signal_ids)
        rows = cur.fetchall()
    except Exception as exc:
        logger.error("DB query failed: %s", exc)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid, pnl, pnl_pct = row
        if sid is None:
            continue
        # First row per ts_signal_id wins (ordered by abs(pnl) DESC)
        if str(sid) not in result:
            try:
                pnl_f = float(pnl)
                result[str(sid)] = {
                    "win": pnl_f > 0,
                    "pnl": round(pnl_f, 4),
                    "pnl_pct": round(float(pnl_pct), 6) if pnl_pct is not None else None,
                }
            except (TypeError, ValueError):
                pass
    return result


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return entries


def _write_jsonl_atomic(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write entries to a temp file then atomically replace path."""
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".platt_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, default=_json_default) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    import numpy as np  # optional; only used if numpy objects sneak in
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Main reconciliation logic
# ---------------------------------------------------------------------------

def reconcile(
    *,
    db_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Run the reconciliation.

    Returns (total_entries, updated, already_had_outcome).
    """
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        sys.exit(1)

    if not log_path.exists():
        print(f"[INFO] JSONL log not found: {log_path} — nothing to reconcile.")
        return 0, 0, 0

    entries = _load_jsonl(log_path)
    total = len(entries)

    # Collect ts_signal_ids that still need outcome (Phase 7.13-A2: string IDs)
    pending: List[str] = []
    already_done = 0
    for entry in entries:
        if "outcome" in entry:
            already_done += 1
            continue
        sid = entry.get("signal_id")
        if sid is not None:
            sid_str = str(sid).strip()
            if sid_str:
                pending.append(sid_str)

    if not pending:
        print(f"[INFO] 0 entries need outcome (total={total}, already_done={already_done}).")
        return total, 0, already_done

    # Query DB
    conn = _connect(db_path)
    try:
        outcomes = _fetch_outcomes_for_signals(conn, pending)
    finally:
        conn.close()

    # Patch entries
    updated = 0
    for entry in entries:
        if "outcome" in entry:
            continue
        sid = entry.get("signal_id")
        if sid is None:
            continue
        sid_str = str(sid).strip()
        if sid_str in outcomes:
            entry["outcome"] = outcomes[sid_str]
            updated += 1

    print(
        f"[update_platt_outcomes] total={total} pending={len(pending)} "
        f"matched={updated} already_done={already_done} "
        f"still_pending={len(pending) - updated}"
    )

    if updated > 0 and not dry_run:
        _write_jsonl_atomic(log_path, entries)
        print(f"[update_platt_outcomes] Wrote {total} entries back to {log_path}")
    elif dry_run and updated > 0:
        print(f"[update_platt_outcomes] dry-run: would update {updated} entries (no file written)")

    return total, updated, already_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconcile quant_validation.jsonl with trade_executions outcomes for Platt scaling."
    )
    p.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Path to portfolio_maximizer.db (default: PORTFOLIO_DB_PATH env or data/portfolio_maximizer.db)",
    )
    p.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="Path to quant_validation.jsonl (default: logs/signals/quant_validation.jsonl)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing anything.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else _DEFAULT_DB_PATH
    log_path = Path(args.log) if args.log else _DEFAULT_JSONL_PATH

    reconcile(db_path=db_path, log_path=log_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
