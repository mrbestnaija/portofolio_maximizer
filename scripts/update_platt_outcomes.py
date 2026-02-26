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
    """Return {open_ts_signal_id: outcome_dict} for closed trades matching any of the given ids.

    Two-pass approach (Phase 7.15 fix):

    Pass 1 — direct match: close leg's ts_signal_id IN (signal_ids).
    Works for same-run round-trips where the exit signal reuses the entry signal ID.

    Pass 2 — open-leg join: join close leg to its open leg via entry_trade_id, then
    match on open leg's ts_signal_id. This handles the common case where ladder-resume
    exits generate a NEW ts_signal_id on the close leg that differs from the open signal
    (the one recorded in JSONL). The JSONL entry always uses the OPEN signal's ID, so
    matching must trace close -> open via entry_trade_id to find it.

    Result is keyed by the OPEN signal's ts_signal_id so callers can look up outcomes
    using the same signal_id stored in the JSONL entry.
    """
    if not signal_ids:
        return {}

    placeholders = ",".join("?" * len(signal_ids))
    result: Dict[str, Dict[str, Any]] = {}
    cur = conn.cursor()

    def _add_rows(rows: list, key_col_index: int = 0) -> None:
        for row in rows:
            sid, pnl, pnl_pct = row[key_col_index], row[1], row[2]
            if sid is None:
                continue
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

    # Pass 1: direct match on close leg's ts_signal_id (same-run round trips).
    try:
        cur.execute(
            f"""
            SELECT ts_signal_id, realized_pnl, realized_pnl_pct
            FROM trade_executions
            WHERE ts_signal_id IN ({placeholders})
              AND is_close = 1
              AND realized_pnl IS NOT NULL
            ORDER BY ABS(realized_pnl) DESC
            """,
            signal_ids,
        )
        _add_rows(cur.fetchall())
    except Exception as exc:
        logger.error("DB query (pass 1) failed: %s", exc)

    # Pass 2: join close leg to open leg via entry_trade_id; match on open
    # leg's ts_signal_id. This covers ladder-resume exits where the close leg
    # carries a different ts_signal_id than the opening signal recorded in JSONL.
    try:
        cur.execute(
            f"""
            SELECT o.ts_signal_id, c.realized_pnl, c.realized_pnl_pct
            FROM trade_executions c
            JOIN trade_executions o ON c.entry_trade_id = o.id
            WHERE o.ts_signal_id IN ({placeholders})
              AND c.is_close = 1
              AND c.realized_pnl IS NOT NULL
            ORDER BY ABS(c.realized_pnl) DESC
            """,
            signal_ids,
        )
        _add_rows(cur.fetchall())
    except Exception as exc:
        logger.error("DB query (pass 2) failed: %s", exc)

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

    # Collect ts_signal_ids that still need outcome (Phase 7.13-A2: string IDs).
    # Only BUY/SELL actions are collected: HOLD signals structurally cannot produce
    # is_close=1 trades and must not inflate the still_pending starvation metric.
    pending: List[str] = []
    already_done = 0
    hold_skipped = 0
    other_no_sid = 0
    for entry in entries:
        if "outcome" in entry:
            already_done += 1
            continue
        action = str(entry.get("action", "")).upper()
        if action == "HOLD":
            hold_skipped += 1
            continue
        sid = entry.get("signal_id")
        if sid is not None:
            sid_str = str(sid).strip()
            if sid_str:
                pending.append(sid_str)
            else:
                other_no_sid += 1
        else:
            other_no_sid += 1

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
        f"still_pending={len(pending) - updated} "
        f"hold_skipped={hold_skipped} no_sid={other_no_sid}"
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
