#!/usr/bin/env python3
"""Repair script: backfill entry_trade_id for unlinked closing legs.

Primary strategy:
- Match unlinked closes to existing orphan BUY legs (same ticker/date context).

Optional forensic strategy (`--reconstruct-from-state`):
- When no matching BUY leg exists, reconstruct a deterministic BUY entry from
  close-row historical position state (entry_price/position_before/holding_days),
  insert it, and link the close to the reconstructed leg.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "portfolio_maximizer.db"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.sqlite_guardrails import guarded_sqlite_connect


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _position_sign(value: Any) -> int:
    numeric = _safe_float(value, 0.0)
    if numeric > 1e-9:
        return 1
    if numeric < -1e-9:
        return -1
    return 0


def _parse_trade_date(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None


def _parse_iso_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _close_to_entry_action(close_action: Any) -> str | None:
    side = str(close_action or "").strip().upper()
    if side == "SELL":
        return "BUY"
    if side == "BUY":
        return "SELL"
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_close_allocation_schema(conn: sqlite3.Connection) -> None:
    schema_sql = """
        CREATE TABLE IF NOT EXISTS trade_close_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            close_trade_id INTEGER NOT NULL,
            entry_trade_id INTEGER NOT NULL,
            allocated_shares REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(close_trade_id, entry_trade_id)
        );
        CREATE INDEX IF NOT EXISTS idx_trade_close_allocations_close
            ON trade_close_allocations(close_trade_id);
        CREATE INDEX IF NOT EXISTS idx_trade_close_allocations_entry
            ON trade_close_allocations(entry_trade_id);
        CREATE VIEW IF NOT EXISTS trade_close_linkages AS
        SELECT
            a.close_trade_id,
            a.entry_trade_id,
            a.allocated_shares
        FROM trade_close_allocations a
        UNION ALL
        SELECT
            c.id AS close_trade_id,
            c.entry_trade_id,
            COALESCE(c.close_size, c.shares, 0.0) AS allocated_shares
        FROM trade_executions c
        WHERE COALESCE(c.is_close, 0) = 1
          AND c.entry_trade_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM trade_close_allocations a
              WHERE a.close_trade_id = c.id
          );
        """
    try:
        conn.executescript(schema_sql)
        return
    except sqlite3.DatabaseError as exc:
        if "not authorized" not in str(exc).lower():
            raise

    db_list = conn.execute("PRAGMA database_list").fetchall()
    db_path = None
    for row in db_list:
        try:
            candidate = str(row[2] or "")
        except (IndexError, TypeError):
            candidate = ""
        if candidate:
            db_path = candidate
            break
    if not db_path:
        raise

    fallback = sqlite3.connect(db_path)
    try:
        fallback.executescript(schema_sql)
        fallback.commit()
    finally:
        fallback.close()


def find_unlinked_closes(conn: sqlite3.Connection, close_ids: set[int] | None = None) -> list[sqlite3.Row]:
    """Find closing legs with no entry_trade_id."""
    _ensure_close_allocation_schema(conn)
    base_sql = """
        SELECT
            id, ticker, trade_date, action, shares, price, realized_pnl, bar_timestamp, run_id,
            entry_price, close_size, position_before, position_after, holding_period_days,
            data_source, execution_mode, asset_class, instrument_type, multiplier,
            commission, mid_price, mid_slippage_bps, effective_confidence,
            COALESCE(is_synthetic, 0) AS is_synthetic,
            COALESCE(is_contaminated, 0) AS is_contaminated
        FROM trade_executions
        WHERE is_close = 1
          AND NOT EXISTS (
              SELECT 1
              FROM trade_close_linkages link
              WHERE link.close_trade_id = trade_executions.id
          )
    """
    params: tuple[Any, ...] = ()
    if close_ids:
        placeholders = ",".join("?" for _ in sorted(close_ids))
        base_sql += f" AND id IN ({placeholders})"
        params = tuple(sorted(close_ids))
    base_sql += " ORDER BY trade_date, id"
    return conn.execute(base_sql, params).fetchall()


def find_orphaned_entries(conn: sqlite3.Connection, ticker: str, entry_action: str) -> list[sqlite3.Row]:
    """Find orphaned entries for a ticker/action pair (no close linkage)."""
    _ensure_close_allocation_schema(conn)
    return conn.execute(
        """
        SELECT
            o.id,
            o.ticker,
            o.trade_date,
            o.execution_mode,
            o.shares,
            o.price,
            o.bar_timestamp,
            o.run_id,
            COALESCE(o.is_synthetic, 0) AS is_synthetic,
            o.position_before,
            o.position_after,
            COALESCE(link.used_qty, 0.0) AS used_qty,
            MAX(0.0, COALESCE(o.shares, 0.0) - COALESCE(link.used_qty, 0.0)) AS remaining_qty
        FROM trade_executions
        AS o
        LEFT JOIN (
            SELECT entry_trade_id, SUM(COALESCE(allocated_shares, 0.0)) AS used_qty
            FROM trade_close_linkages
            GROUP BY entry_trade_id
        ) AS link
          ON link.entry_trade_id = o.id
        WHERE o.ticker = ?
          AND o.action = ?
          AND o.is_close = 0
          AND MAX(0.0, COALESCE(o.shares, 0.0) - COALESCE(link.used_qty, 0.0)) > 0.0
        ORDER BY o.trade_date, o.id
        """,
        (ticker, entry_action),
    ).fetchall()


def _is_clean_live_close(close_row: sqlite3.Row) -> bool:
    return (
        str(close_row["execution_mode"] or "").strip().lower() == "live"
        and _safe_int(close_row["is_synthetic"]) == 0
        and _safe_int(close_row["is_contaminated"]) == 0
    )


def _current_position_run_entry_ids(
    conn: sqlite3.Connection,
    close_row: sqlite3.Row,
) -> set[int]:
    ticker = str(close_row["ticker"] or "").strip()
    close_date = str(close_row["trade_date"] or "").strip()
    close_id = _safe_int(close_row["id"])
    target_sign = _position_sign(close_row["position_before"])
    if not ticker or not close_date or close_id <= 0 or target_sign == 0:
        return set()

    prior_rows = conn.execute(
        """
        SELECT id, position_after
        FROM trade_executions
        WHERE ticker = ?
          AND (trade_date < ? OR (trade_date = ? AND id < ?))
        ORDER BY trade_date, id
        """,
        (ticker, close_date, close_date, close_id),
    ).fetchall()

    if not prior_rows:
        return set()

    boundary_index = -1
    for idx, row in enumerate(prior_rows):
        sign_after = _position_sign(row["position_after"])
        if sign_after == 0 or sign_after != target_sign:
            boundary_index = idx

    return {
        _safe_int(row["id"])
        for row in prior_rows[boundary_index + 1 :]
        if _safe_int(row["id"]) > 0
    }


def _sum_remaining_qty(entries: list[sqlite3.Row]) -> float:
    return sum(_safe_float(entry["remaining_qty"], _safe_float(entry["shares"])) for entry in entries)


def _select_candidate_entries(
    conn: sqlite3.Connection,
    close_row: sqlite3.Row,
    orphaned_entries: list[sqlite3.Row],
) -> tuple[list[sqlite3.Row], str]:
    close_size = _safe_float(close_row["close_size"], _safe_float(close_row["shares"]))
    clean_live_close = _is_clean_live_close(close_row)

    def _non_synthetic_only(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        return [
            row
            for row in rows
            if _safe_int(row["is_synthetic"]) == 0
            and str(row["execution_mode"] or "").strip().lower() != "synthetic"
        ]

    current_run_ids = _current_position_run_entry_ids(conn, close_row)
    if current_run_ids:
        run_entries = [row for row in orphaned_entries if _safe_int(row["id"]) in current_run_ids]
        if clean_live_close:
            run_entries = _non_synthetic_only(run_entries)
        if _sum_remaining_qty(run_entries) >= close_size > 0.0:
            return run_entries, "current_position_run"

    if clean_live_close:
        return [], "clean_live_no_current_run_match"

    broad_entries = list(orphaned_entries)
    if _sum_remaining_qty(broad_entries) >= close_size > 0.0:
        return broad_entries, "broad_orphan_match"
    return [], "insufficient_orphan_inventory"


def match_fifo_allocations(unlinked_close: sqlite3.Row, orphaned_entries: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Allocate a close leg across remaining opposite-side open lots."""
    close_date = str(unlinked_close["trade_date"] or "")
    close_shares = _safe_float(unlinked_close["close_size"], _safe_float(unlinked_close["shares"]))
    close_run = str(unlinked_close["run_id"] or "")

    candidates: list[tuple[int, str, int, float]] = []
    for entry in orphaned_entries:
        entry_date = str(entry["trade_date"] or "")
        if entry_date > close_date:
            continue

        entry_run = str(entry["run_id"] or "")
        try:
            run_distance = abs(int(close_run.split("_")[1]) - int(entry_run.split("_")[1]))
        except (ValueError, IndexError):
            run_distance = 10_000

        remaining_qty = _safe_float(entry["remaining_qty"], _safe_float(entry["shares"]))
        if remaining_qty <= 0.0:
            continue
        candidates.append((run_distance, entry_date, _safe_int(entry["id"]), remaining_qty))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[1], item[2], item[0]))
    remaining_close = close_shares
    allocations: list[dict[str, Any]] = []
    for _, _, entry_id, remaining_qty in candidates:
        if remaining_close <= 1e-9:
            break
        allocated = min(remaining_close, remaining_qty)
        if allocated <= 1e-9:
            continue
        allocations.append(
            {
                "entry_id": int(entry_id),
                "allocated_shares": float(allocated),
            }
        )
        remaining_close -= allocated

    if remaining_close > 1e-9:
        return []
    return allocations


def _apply_allocations_to_close(
    conn: sqlite3.Connection,
    close_id: int,
    allocations: list[dict[str, Any]],
) -> None:
    if not allocations:
        return
    representative_entry_id = int(allocations[0]["entry_id"])
    conn.execute(
        "UPDATE trade_executions SET entry_trade_id = ? WHERE id = ?",
        (representative_entry_id, close_id),
    )
    conn.execute(
        "DELETE FROM trade_close_allocations WHERE close_trade_id = ?",
        (close_id,),
    )
    conn.executemany(
        """
        INSERT INTO trade_close_allocations (close_trade_id, entry_trade_id, allocated_shares)
        VALUES (?, ?, ?)
        """,
        [
            (int(close_id), int(allocation["entry_id"]), float(allocation["allocated_shares"]))
            for allocation in allocations
        ],
    )


def _collect_forensic_evidence(close_row: sqlite3.Row, logs_root: Path) -> dict[str, Any]:
    ticker = str(close_row["ticker"] or "").strip()
    entry_price = _safe_float(close_row["entry_price"], 0.0)
    close_run_id = str(close_row["run_id"] or "").strip()
    close_warning_token = f"Closing {ticker} position but no entry_trade_id found"
    entry_price_token = f"{entry_price:.5f}".rstrip("0").rstrip(".")

    open_hits: list[str] = []
    close_hits: list[str] = []
    scanned_files = 0
    audit_root = logs_root / "audit_sprint"
    if audit_root.exists():
        for log_file in sorted(audit_root.rglob("*.log")):
            scanned_files += 1
            try:
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                text = line.strip()
                if (
                    len(open_hits) < 10
                    and "Open positions:" in text
                    and ticker in text
                    and entry_price_token in text
                ):
                    open_hits.append(f"{log_file}:{line_no}:{text}")
                if len(close_hits) < 10 and close_warning_token in text:
                    close_hits.append(f"{log_file}:{line_no}:{text}")
                if len(open_hits) >= 10 and len(close_hits) >= 10:
                    break

    return {
        "scanned_log_files": scanned_files,
        "open_position_hits": open_hits,
        "close_warning_hits": close_hits,
        "run_id": close_run_id,
    }


def _derive_reconstructed_entry(close_row: sqlite3.Row) -> dict[str, Any] | None:
    close_id = _safe_int(close_row["id"])
    ticker = str(close_row["ticker"] or "").strip()
    close_action = str(close_row["action"] or "").strip().upper()
    entry_action = _close_to_entry_action(close_action)
    close_qty = _safe_float(close_row["close_size"], _safe_float(close_row["shares"]))
    entry_price = _safe_float(close_row["entry_price"])
    position_before = _safe_float(close_row["position_before"])
    holding_days = max(0, _safe_int(close_row["holding_period_days"], 0))

    if close_id <= 0 or not ticker:
        return None
    if entry_action not in {"BUY", "SELL"}:
        return None
    if close_qty <= 0.0 or entry_price <= 0.0:
        return None
    if entry_action == "BUY":
        # Closing SELL should unwind a long position.
        if position_before + 1e-9 < close_qty:
            return None
        entry_position_after = close_qty
    else:
        # Closing BUY should unwind a short position.
        if abs(position_before) + 1e-9 < close_qty:
            return None
        if position_before >= 0:
            return None
        entry_position_after = -close_qty

    close_trade_date = _parse_trade_date(close_row["trade_date"])
    if close_trade_date is None:
        return None
    entry_trade_date = (close_trade_date - timedelta(days=holding_days)).date().isoformat()

    entry_bar_ts = None
    close_bar_ts = _parse_iso_ts(close_row["bar_timestamp"])
    if close_bar_ts is not None:
        entry_bar_ts = (close_bar_ts - timedelta(days=holding_days)).isoformat()

    close_run = str(close_row["run_id"] or "").strip() or "unknown_run"
    base_exec = str(close_row["execution_mode"] or "").strip() or "live"
    recon_run_id = f"{close_run}_recon_entry_{close_id}"
    recon_exec_mode = f"{base_exec}_entry_reconstructed"

    return {
        "close_id": close_id,
        "ticker": ticker,
        "entry_action": entry_action,
        "shares": close_qty,
        "entry_price": entry_price,
        "entry_position_after": entry_position_after,
        "entry_trade_date": entry_trade_date,
        "entry_bar_timestamp": entry_bar_ts,
        "run_id": recon_run_id,
        "execution_mode": recon_exec_mode,
        "data_source": str(close_row["data_source"] or "").strip() or None,
        "asset_class": str(close_row["asset_class"] or "").strip() or "equity",
        "instrument_type": str(close_row["instrument_type"] or "").strip() or "spot",
        "multiplier": max(0.0, _safe_float(close_row["multiplier"], 1.0)) or 1.0,
        "mid_price": entry_price,
        "mid_slippage_bps": 0.0,
        "commission": 0.0,
        "effective_confidence": close_row["effective_confidence"],
    }


def _find_existing_reconstructed_entry(conn: sqlite3.Connection, candidate: dict[str, Any]) -> int | None:
    row = conn.execute(
        """
        SELECT id
        FROM trade_executions
        WHERE run_id = ?
          AND action = ?
          AND is_close = 0
          AND ticker = ?
        LIMIT 1
        """,
        (candidate["run_id"], candidate["entry_action"], candidate["ticker"]),
    ).fetchone()
    if not row:
        return None
    return _safe_int(row["id"], 0) or None


def _insert_reconstructed_entry(conn: sqlite3.Connection, candidate: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO trade_executions (
            ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size,
            position_before, position_after, is_close,
            bar_timestamp, exit_reason,
            asset_class, instrument_type, multiplier,
            effective_confidence, is_diagnostic, is_synthetic
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, 0, ?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (
            candidate["ticker"],
            candidate["entry_trade_date"],
            candidate["entry_action"],
            candidate["shares"],
            candidate["entry_price"],
            candidate["shares"] * candidate["entry_price"],
            candidate["commission"],
            candidate["mid_price"],
            candidate["mid_slippage_bps"],
            candidate["data_source"],
            candidate["execution_mode"],
            candidate["run_id"],
            0.0,
            candidate["entry_position_after"],
            candidate["entry_bar_timestamp"],
            "RECONSTRUCTED_MISSING_ENTRY",
            candidate["asset_class"],
            candidate["instrument_type"],
            candidate["multiplier"],
            candidate["effective_confidence"],
        ),
    )
    return _safe_int(cur.lastrowid, 0)


def repair_linkage(
    db_path: Path,
    dry_run: bool = True,
    close_ids: list[int] | None = None,
    reconstruct_from_state: bool = False,
    forensic_report_file: Path | None = None,
    logs_root: Path | None = None,
) -> int:
    """Backfill entry_trade_id for unlinked closes."""
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    logs_root = (logs_root or (ROOT / "logs")).resolve()

    conn = guarded_sqlite_connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_close_allocation_schema(conn)

    print("=" * 70)
    print("REPAIR UNLINKED CLOSES")
    print("=" * 70)
    print()

    selected_ids = {int(x) for x in (close_ids or []) if int(x) > 0}
    if selected_ids:
        print(f"Target close IDs: {sorted(selected_ids)}")
    unlinked = find_unlinked_closes(conn, close_ids=selected_ids or None)
    if not unlinked:
        print("[OK] No unlinked closes found")
        conn.close()
        return 0

    print(f"Found {len(unlinked)} unlinked closing legs")
    print()

    direct_repairs: list[dict[str, Any]] = []
    reconstruct_repairs: list[dict[str, Any]] = []
    forensic_entries: list[dict[str, Any]] = []

    for close_leg in unlinked:
        close_id = _safe_int(close_leg["id"])
        close_action = str(close_leg["action"] or "").strip().upper()
        entry_action = _close_to_entry_action(close_action)
        ticker = str(close_leg["ticker"] or "")
        close_date = str(close_leg["trade_date"] or "")
        shares = _safe_float(close_leg["shares"])
        price = _safe_float(close_leg["price"])
        pnl = _safe_float(close_leg["realized_pnl"])
        run_id = str(close_leg["run_id"] or "")
        print(
            f"Unlinked CLOSE ID {close_id}: action={close_action or 'UNKNOWN'} "
            f"{ticker} {close_date} - {shares} shares @ {price:.2f}"
        )
        print(f"  PnL: ${pnl:.2f}, Run: {run_id}")

        if entry_action not in {"BUY", "SELL"}:
            print("  [WARNING] Unsupported close action; cannot determine opposite-side entry action")
            print()
            continue

        orphans = find_orphaned_entries(conn, ticker, entry_action=entry_action)
        selected_orphans: list[sqlite3.Row] = []
        candidate_strategy = "no_orphans"
        if orphans:
            selected_orphans, candidate_strategy = _select_candidate_entries(conn, close_leg, orphans)
        matched_allocations = match_fifo_allocations(close_leg, selected_orphans) if selected_orphans else []
        if matched_allocations:
            for allocation in matched_allocations:
                entry = next(b for b in selected_orphans if _safe_int(b["id"]) == int(allocation["entry_id"]))
                print(
                    f"  [MATCH] {entry_action} ID {allocation['entry_id']}: {entry['trade_date']} - "
                    f"{float(allocation['allocated_shares']):.4f}/{_safe_float(entry['shares']):.4f} shares "
                    f"@ {_safe_float(entry['price']):.2f}"
                )
                print(f"          Run: {entry['run_id']}")
            direct_repairs.append(
                {
                    "close_id": close_id,
                    "entry_id": int(matched_allocations[0]["entry_id"]),
                    "allocations": matched_allocations,
                    "entry_action": entry_action,
                    "ticker": ticker,
                    "reason": f"matched_existing_orphan_entry:{candidate_strategy}",
                }
            )
            print()
            continue

        if not reconstruct_from_state:
            if candidate_strategy == "clean_live_no_current_run_match":
                print(
                    f"  [WARNING] Refusing broad repair for clean live close; "
                    f"no current-position {entry_action} lots matched {ticker}"
                )
            else:
                print(f"  [WARNING] No orphaned {entry_action} entries found for {ticker}")
            print()
            continue

        candidate = _derive_reconstructed_entry(close_leg)
        evidence = _collect_forensic_evidence(close_leg, logs_root=logs_root)
        forensic_entry: dict[str, Any] = {
            "close_id": close_id,
            "close_action": close_action,
            "entry_action": entry_action,
            "ticker": ticker,
            "trade_date": close_date,
            "run_id": run_id,
            "entry_price": _safe_float(close_leg["entry_price"]),
            "close_size": _safe_float(close_leg["close_size"], shares),
            "position_before": _safe_float(close_leg["position_before"]),
            "position_after": _safe_float(close_leg["position_after"]),
            "holding_period_days": _safe_int(close_leg["holding_period_days"], 0),
            "evidence": evidence,
        }

        if not candidate:
            forensic_entry["status"] = "insufficient_state"
            forensic_entries.append(forensic_entry)
            print("  [WARNING] Reconstruction from state not possible (insufficient deterministic fields)")
            print()
            continue

        existing_recon_id = _find_existing_reconstructed_entry(conn, candidate)
        if existing_recon_id:
            direct_repairs.append(
                {
                    "close_id": close_id,
                    "entry_id": existing_recon_id,
                    "entry_action": entry_action,
                    "ticker": ticker,
                    "reason": "matched_existing_reconstructed_entry",
                }
            )
            forensic_entry["status"] = "reused_existing_reconstructed_entry"
            forensic_entry["reconstructed_entry_id"] = existing_recon_id
            forensic_entries.append(forensic_entry)
            print(f"  [RECON] Reusing reconstructed {candidate['entry_action']} ID {existing_recon_id}")
            print()
            continue

        reconstruct_repairs.append({"close_id": close_id, "ticker": ticker, "candidate": candidate})
        forensic_entry["status"] = "planned_reconstruction"
        forensic_entry["candidate"] = candidate
        forensic_entries.append(forensic_entry)
        print(
            f"  [RECON] Planned reconstructed {candidate['entry_action']} "
            f"(date={candidate['entry_trade_date']}, shares={candidate['shares']:.4f}, price={candidate['entry_price']:.2f})"
        )
        print()

    total_ops = len(direct_repairs) + len(reconstruct_repairs)
    if total_ops == 0:
        print("[WARNING] No repairs identified")
        if forensic_entries and forensic_report_file:
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "db_path": str(db_path),
                "dry_run": bool(dry_run),
                "reconstruct_from_state": bool(reconstruct_from_state),
                "forensic_entries": forensic_entries,
            }
            _write_json(forensic_report_file, payload)
            print(f"Forensic report: {forensic_report_file}")
        conn.close()
        return 0

    print(f"Identified {len(direct_repairs)} direct repair(s)")
    if reconstruct_repairs:
        print(f"Identified {len(reconstruct_repairs)} reconstruction repair(s)")
    print()

    if dry_run:
        print("[DRY RUN] Would apply the following repairs:")
        for op in direct_repairs:
            alloc_desc = ", ".join(
                f"{allocation['entry_id']}:{float(allocation['allocated_shares']):.4f}"
                for allocation in op.get("allocations", [])
            )
            print(
                f"  LINK CLOSE {op['close_id']} -> [{alloc_desc}] "
                f"(representative entry_trade_id={op['entry_id']})"
            )
        for op in reconstruct_repairs:
            candidate = op["candidate"]
            print(
                f"  INSERT reconstructed {candidate['entry_action']} "
                f"(ticker={candidate['ticker']}, date={candidate['entry_trade_date']}, "
                f"shares={candidate['shares']:.4f}, price={candidate['entry_price']:.2f}, run_id={candidate['run_id']})"
            )
            print(f"  UPDATE trade_executions SET entry_trade_id = <new_id> WHERE id = {op['close_id']}")
        print()
        print("Re-run with --apply to execute")
        if forensic_entries and forensic_report_file:
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "db_path": str(db_path),
                "dry_run": True,
                "reconstruct_from_state": bool(reconstruct_from_state),
                "direct_repairs": direct_repairs,
                "reconstruct_repairs_planned": len(reconstruct_repairs),
                "forensic_entries": forensic_entries,
            }
            _write_json(forensic_report_file, payload)
            print(f"Forensic report: {forensic_report_file}")
        conn.close()
        return 0
    else:
        print("Applying repairs...")
        try:
            for op in direct_repairs:
                _apply_allocations_to_close(conn, int(op["close_id"]), list(op.get("allocations", [])))
                print(
                    f"  [OK] Linked CLOSE {op['close_id']} -> {op['entry_action']} {op['entry_id']} "
                    f"({op['ticker']}, reason={op['reason']})"
                )

            for op in reconstruct_repairs:
                candidate = op["candidate"]
                close_id = _safe_int(op["close_id"])
                entry_id = _insert_reconstructed_entry(conn, candidate)
                _apply_allocations_to_close(
                    conn,
                    close_id,
                    [{"entry_id": entry_id, "allocated_shares": float(candidate["shares"])}],
                )
                print(
                    f"  [OK] Reconstructed {candidate['entry_action']} {entry_id} and linked CLOSE {close_id} "
                    f"({candidate['ticker']})"
                )
                for entry in forensic_entries:
                    if _safe_int(entry.get("close_id")) == close_id and entry.get("status") == "planned_reconstruction":
                        entry["status"] = "applied_reconstruction"
                        entry["reconstructed_entry_id"] = entry_id
                        break

            conn.commit()
            print()
            print("[SUCCESS] Repairs applied")
        except sqlite3.Error as exc:
            print(f"[ERROR] Repair failed: {exc}")
            conn.rollback()
            conn.close()
            return 1

    if forensic_entries and forensic_report_file:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "dry_run": bool(dry_run),
            "reconstruct_from_state": bool(reconstruct_from_state),
            "direct_repairs": direct_repairs,
            "reconstruct_repairs_planned": len(reconstruct_repairs),
            "forensic_entries": forensic_entries,
        }
        _write_json(forensic_report_file, payload)
        print(f"Forensic report: {forensic_report_file}")

    print()
    print("Verifying repairs...")
    remaining = find_unlinked_closes(conn, close_ids=selected_ids or None)
    conn.close()

    print(f"Remaining unlinked closes: {len(remaining)}")
    if remaining:
        print("[WARNING] Some closes still unlinked:")
        for row in remaining:
            print(f"  ID {row['id']}: {row['ticker']} {row['trade_date']}")
        return 1

    print("[OK] All closes now linked")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--apply", action="store_true", help="Apply repairs (default is dry-run)")
    parser.add_argument(
        "--close-ids",
        nargs="*",
        type=int,
        default=[],
        help="Optional close leg IDs to reconcile. If omitted, scans all unlinked closes.",
    )
    parser.add_argument(
        "--reconstruct-from-state",
        action="store_true",
        help="When no orphan BUY can be matched, reconstruct missing BUY from close-row historical state.",
    )
    parser.add_argument(
        "--forensic-report-file",
        type=Path,
        default=ROOT / "logs" / "automation" / "repair_unlinked_closes_forensic_latest.json",
        help="JSON forensic report output path.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=ROOT / "logs",
        help="Logs root used for forensic evidence lookup.",
    )

    args = parser.parse_args()
    report_file = args.forensic_report_file
    if not report_file.is_absolute():
        report_file = (ROOT / report_file).resolve()
    logs_root = args.logs_root
    if not logs_root.is_absolute():
        logs_root = (ROOT / logs_root).resolve()

    sys.exit(
        repair_linkage(
            args.db,
            dry_run=not args.apply,
            close_ids=args.close_ids,
            reconstruct_from_state=bool(args.reconstruct_from_state),
            forensic_report_file=report_file if bool(args.reconstruct_from_state) else None,
            logs_root=logs_root,
        )
    )
