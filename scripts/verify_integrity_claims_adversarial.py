#!/usr/bin/env python3
"""Independent adversarial verifier for integrity claims.

This script validates claim sets against:
1) Current database facts (canonical metrics + raw linkage state)
2) Integrity enforcer behavior
3) Adversarial mutation attempts on isolated DB snapshots

It never mutates the live DB file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
from integrity.sqlite_guardrails import guarded_sqlite_connect


EXPECTED_LINKS = {9: 7, 10: 8, 15: 18, 23: 21}
KNOWN_HISTORICAL_ORPHANS = {5, 6, 11, 13}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _approx_equal(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(float(a) - float(b)) <= tol


def _snapshot_db(src_db: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pmx_claim_verify_"))
    snap_db = temp_dir / "snapshot.db"
    with guarded_sqlite_connect(str(src_db)) as src, guarded_sqlite_connect(
        str(snap_db),
        enable_guardrails=False,
    ) as dst:
        src.backup(dst)
    return snap_db


def _cleanup_snapshot(snap_db: Path) -> None:
    try:
        parent = snap_db.parent
        if snap_db.exists():
            snap_db.unlink()
        parent.rmdir()
    except Exception:
        pass


def _run_enforcer(db_path: Path) -> Dict[str, Any]:
    with PnLIntegrityEnforcer(str(db_path), auto_create_views=False) as enforcer:
        metrics = enforcer.get_canonical_metrics()
        violations = enforcer.run_full_integrity_audit()
    violation_dict = [
        {
            "check_name": v.check_name,
            "severity": v.severity,
            "count": v.count,
            "affected_ids": list(v.affected_ids),
            "description": v.description,
        }
        for v in violations
    ]
    overall = (
        "HEALTHY"
        if not any(v["severity"] in {"CRITICAL", "HIGH"} for v in violation_dict)
        else "CRITICAL_FAIL"
    )
    return {
        "canonical_metrics": {
            "closed_trades": metrics.total_round_trips,
            "total_pnl": round(metrics.total_realized_pnl, 2),
            "win_rate": round(metrics.win_rate, 4),
            "profit_factor": round(metrics.profit_factor, 2),
            "wins": metrics.win_count,
            "losses": metrics.loss_count,
            "diagnostic_trades_excluded": metrics.diagnostic_trades_excluded,
            "synthetic_trades_excluded": metrics.synthetic_trades_excluded,
            "opening_legs_with_pnl": metrics.opening_legs_with_pnl,
        },
        "violations": violation_dict,
        "overall_status": overall,
    }


def _query_baseline(db_path: Path) -> Dict[str, Any]:
    with guarded_sqlite_connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        raw_orphans = conn.execute(
            """
            SELECT id, ticker, trade_date
            FROM trade_executions
            WHERE action = 'BUY'
              AND is_close = 0
              AND realized_pnl IS NULL
              AND id NOT IN (
                SELECT COALESCE(entry_trade_id, -1)
                FROM trade_executions
                WHERE is_close = 1
              )
            ORDER BY id
            """
        ).fetchall()
        close_without_link = conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE is_close = 1 AND entry_trade_id IS NULL"
        ).fetchone()[0]
        link_rows = conn.execute(
            """
            SELECT id, entry_trade_id
            FROM trade_executions
            WHERE id IN (9, 10, 15, 23)
            ORDER BY id
            """
        ).fetchall()
        trigger_sql_row = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type='trigger' AND name='enforce_immutable_closed_trades'
            """
        ).fetchone()

    trigger_sql = trigger_sql_row[0] if trigger_sql_row else ""
    mapped_links = {int(r["id"]): int(r["entry_trade_id"] or 0) for r in link_rows}
    raw_orphan_ids = [int(r["id"]) for r in raw_orphans]
    recent_cutoff = (datetime.now() - timedelta(days=3)).date().isoformat()
    stale_orphans = [int(r["id"]) for r in raw_orphans if str(r["trade_date"]) < recent_cutoff]
    recent_orphans = [int(r["id"]) for r in raw_orphans if str(r["trade_date"]) >= recent_cutoff]

    return {
        "raw_orphaned_buy_entries": {
            "count": len(raw_orphans),
            "ids": raw_orphan_ids,
            "stale_ids": stale_orphans,
            "recent_ids": recent_orphans,
        },
        "close_without_entry_link_count": int(close_without_link),
        "entry_link_mapping": mapped_links,
        "trigger_sql": trigger_sql,
        "trigger_allows_entry_id_backfill": (
            "OLD.entry_trade_id IS NULL AND NEW.entry_trade_id IS NOT NULL" in trigger_sql
        ),
        "trigger_blocks_core_field_update": (
            "Cannot modify core fields of closed trades" in trigger_sql
        ),
    }


def _attack_stale_orphan_detection(snap_db: Path) -> Dict[str, Any]:
    stale_date = (datetime.now() - timedelta(days=8)).date().isoformat()
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        conn.execute(
            """
            INSERT INTO trade_executions (
                ticker, trade_date, action, shares, price, total_value,
                is_close, execution_mode, run_id
            ) VALUES (?, ?, 'BUY', 1.0, 100.0, 100.0, 0, 'live', 'adv_stale_orphan')
            """,
            ("ADV_ORPHAN", stale_date),
        )
        inserted_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    out = _run_enforcer(snap_db)
    orphan_v = next((v for v in out["violations"] if v["check_name"] == "ORPHANED_POSITION"), None)
    detected = bool(orphan_v and inserted_id in orphan_v["affected_ids"])
    return {
        "attack": "stale_orphan_should_fail_gate",
        "inserted_id": inserted_id,
        "expected": "ORPHANED_POSITION includes inserted stale orphan",
        "actual_detected": detected,
        "pass": detected,
        "orphan_violation": orphan_v,
    }


def _attack_unlinked_close_detection(snap_db: Path) -> Dict[str, Any]:
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        conn.execute(
            """
            INSERT INTO trade_executions (
                ticker, trade_date, action, shares, price, total_value,
                is_close, realized_pnl, execution_mode, run_id
            ) VALUES ('ADV_CLOSE', ?, 'SELL', 1.0, 120.0, 120.0, 1, 20.0, 'live', 'adv_unlinked_close')
            """,
            ((datetime.now() - timedelta(days=1)).date().isoformat(),),
        )
        inserted_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    out = _run_enforcer(snap_db)
    close_v = next(
        (v for v in out["violations"] if v["check_name"] == "CLOSE_WITHOUT_ENTRY_LINK"),
        None,
    )
    detected = bool(close_v and inserted_id in close_v["affected_ids"])
    return {
        "attack": "unlinked_close_should_be_detected",
        "inserted_id": inserted_id,
        "expected": "CLOSE_WITHOUT_ENTRY_LINK includes inserted close",
        "actual_detected": detected,
        "pass": detected,
        "close_violation": close_v,
    }


def _attack_core_field_immutability(snap_db: Path) -> Dict[str, Any]:
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        row = conn.execute(
            "SELECT id, realized_pnl FROM trade_executions WHERE is_close = 1 AND realized_pnl IS NOT NULL LIMIT 1"
        ).fetchone()
        if row is None:
            return {
                "attack": "closed_trade_core_field_immutable",
                "pass": False,
                "error": "No suitable closed trade found in snapshot.",
            }
        close_id, original_pnl = int(row[0]), float(row[1])
        blocked = False
        message = ""
        try:
            conn.execute(
                "UPDATE trade_executions SET realized_pnl = ? WHERE id = ?",
                (original_pnl + 1.0, close_id),
            )
            conn.commit()
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            blocked = True
            message = str(exc)

    return {
        "attack": "closed_trade_core_field_immutable",
        "close_id": close_id,
        "expected": "UPDATE on realized_pnl should be blocked",
        "actual_blocked": blocked,
        "error_message": message,
        "pass": blocked,
    }


def _attack_entry_id_backfill_allowed(snap_db: Path) -> Dict[str, Any]:
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        conn.execute(
            """
            INSERT INTO trade_executions (
                ticker, trade_date, action, shares, price, total_value,
                is_close, realized_pnl, execution_mode, run_id
            ) VALUES ('ADV_BACKFILL', ?, 'SELL', 1.0, 111.0, 111.0, 1, 11.0, 'live', 'adv_backfill')
            """,
            ((datetime.now() - timedelta(days=1)).date().isoformat(),),
        )
        close_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()

        allowed = True
        error_message = ""
        try:
            conn.execute(
                "UPDATE trade_executions SET entry_trade_id = ? WHERE id = ? AND entry_trade_id IS NULL",
                (123456, close_id),
            )
            conn.commit()
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            allowed = False
            error_message = str(exc)

        entry_id = conn.execute(
            "SELECT entry_trade_id FROM trade_executions WHERE id = ?",
            (close_id,),
        ).fetchone()[0]

    passed = allowed and int(entry_id or 0) == 123456
    return {
        "attack": "entry_trade_id_backfill_allowed",
        "close_id": close_id,
        "expected": "NULL -> value backfill should succeed",
        "actual_allowed": allowed,
        "entry_trade_id_after": int(entry_id or 0),
        "error_message": error_message,
        "pass": passed,
    }


def _attack_opening_pnl_direct_blocked(snap_db: Path) -> Dict[str, Any]:
    blocked = False
    error_message = ""
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        try:
            conn.execute(
                """
                INSERT INTO trade_executions (
                    ticker, trade_date, action, shares, price, total_value,
                    is_close, realized_pnl, execution_mode, run_id
                ) VALUES ('ADV_OPENING_PNL', ?, 'BUY', 1.0, 10.0, 10.0, 0, 5.0, 'live', 'adv_opening_pnl')
                """,
                ((datetime.now() - timedelta(days=1)).date().isoformat(),),
            )
            conn.commit()
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            blocked = True
            error_message = str(exc)
    return {
        "attack": "opening_leg_with_pnl_direct_insert_blocked",
        "expected": "insert should fail due CHECK constraint",
        "actual_blocked": blocked,
        "error_message": error_message,
        "pass": blocked,
    }


def _attack_pragma_bypass(snap_db: Path) -> Dict[str, Any]:
    bypassed = False
    error_message = ""
    pragma_value = None
    with guarded_sqlite_connect(str(snap_db), enable_guardrails=False) as conn:
        try:
            conn.execute("PRAGMA ignore_check_constraints = ON")
            pragma_value = conn.execute("PRAGMA ignore_check_constraints").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_executions (
                    ticker, trade_date, action, shares, price, total_value,
                    is_close, realized_pnl, execution_mode, run_id
                ) VALUES ('ADV_PRAGMA', ?, 'BUY', 1.0, 10.0, 10.0, 0, 5.0, 'live', 'adv_pragma')
                """,
                ((datetime.now() - timedelta(days=1)).date().isoformat(),),
            )
            conn.commit()
            bypassed = True
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            error_message = str(exc)
    return {
        "attack": "pragma_ignore_check_constraints_bypass",
        "expected": "ideally blocked; if bypassed, this is a hardening gap",
        "pragma_ignore_check_constraints": pragma_value,
        "bypassed": bypassed,
        "error_message": error_message,
        "pass": not bypassed,
    }


def _run_on_snapshot(db_path: Path, attack_fn) -> Dict[str, Any]:
    snap_db = _snapshot_db(db_path)
    try:
        return attack_fn(snap_db)
    finally:
        _cleanup_snapshot(snap_db)


def _run_existing_adversarial_suite(db_path: Path) -> Dict[str, Any]:
    script_path = ROOT / "scripts" / "adversarial_integrity_test.py"
    if not script_path.exists():
        return {
            "available": False,
            "status": "INCONCLUSIVE",
            "reason": f"missing script: {script_path}",
        }
    snap_db = _snapshot_db(db_path)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), "--db", str(snap_db)],
            capture_output=True,
            text=True,
            check=False,
        )
        out = f"{proc.stdout}\n{proc.stderr}"
        blocked_match = re.search(r"Attacks blocked:\s+(\d+)", out)
        bypassed_match = re.search(r"Attacks bypassed:\s+(\d+)", out)
        blocked = int(blocked_match.group(1)) if blocked_match else None
        bypassed = int(bypassed_match.group(1)) if bypassed_match else None
        return {
            "available": True,
            "exit_code": int(proc.returncode),
            "attacks_blocked": blocked,
            "attacks_bypassed": bypassed,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "output_tail": "\n".join([ln for ln in out.splitlines() if ln.strip()][-40:]),
        }
    finally:
        _cleanup_snapshot(snap_db)


def _claim(status: str, statement: str, expected: Any, actual: Any, evidence: List[str]) -> Dict[str, Any]:
    return {
        "status": status,
        "statement": statement,
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
    }


def _evaluate_claims(
    baseline: Dict[str, Any],
    enforcer_state: Dict[str, Any],
    attack_results: Dict[str, Dict[str, Any]],
    existing_suite: Dict[str, Any],
) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    metrics = enforcer_state["canonical_metrics"]
    raw_orphans = baseline["raw_orphaned_buy_entries"]
    close_wo_link = baseline["close_without_entry_link_count"]
    violations = enforcer_state["violations"]

    metrics_ok = (
        metrics["closed_trades"] == 20
        and _approx_equal(metrics["total_pnl"], 909.18, tol=0.05)
        and _approx_equal(metrics["win_rate"], 0.6, tol=1e-4)
        and _approx_equal(metrics["profit_factor"], 2.78, tol=0.05)
    )
    claims.append(
        _claim(
            "SUPPORTED" if metrics_ok else "UNFOUNDED",
            "Canonical metrics are 20 round-trips, $909.18 PnL, 60% win rate, PF 2.78.",
            {"closed_trades": 20, "total_pnl": 909.18, "win_rate": 0.6, "profit_factor": 2.78},
            metrics,
            [],
        )
    )

    close_link_ok = close_wo_link == 0
    claims.append(
        _claim(
            "SUPPORTED" if close_link_ok else "UNFOUNDED",
            "CLOSE_WITHOUT_ENTRY_LINK is resolved to 0.",
            0,
            close_wo_link,
            [],
        )
    )

    ci_orphan_zero = not any(v["check_name"] == "ORPHANED_POSITION" for v in violations)
    raw_orphan_expected = raw_orphans["count"] == 8
    claims.append(
        _claim(
            "SUPPORTED" if (ci_orphan_zero and raw_orphan_expected) else "UNFOUNDED",
            "ORPHANED_POSITION is 0 at gate level while raw orphan inventory is 8 accepted entries.",
            {"gate_orphan_violations": 0, "raw_orphans": 8},
            {"gate_orphan_violations": 0 if ci_orphan_zero else 1, "raw_orphans": raw_orphans["count"]},
            [],
        )
    )

    overall_healthy = enforcer_state["overall_status"] == "HEALTHY"
    claims.append(
        _claim(
            "SUPPORTED" if overall_healthy else "UNFOUNDED",
            "Integrity overall status is HEALTHY (integrity gate does not force CRITICAL_FAIL).",
            "HEALTHY",
            enforcer_state["overall_status"],
            [],
        )
    )

    links_ok = baseline["entry_link_mapping"] == EXPECTED_LINKS
    claims.append(
        _claim(
            "SUPPORTED" if links_ok else "UNFOUNDED",
            "Backfill links match SELL->BUY mappings: 9->7, 10->8, 15->18, 23->21.",
            EXPECTED_LINKS,
            baseline["entry_link_mapping"],
            [],
        )
    )

    files_to_check = {
        "scripts/repair_unlinked_closes.py",
        "scripts/migrate_allow_entry_id_backfill.py",
        "Documentation/INTEGRITY_STATUS_20260212.md",
        "Documentation/SESSION_SUMMARY_20260212.md",
    }
    existing = {p for p in files_to_check if (ROOT / p).exists()}
    files_ok = existing == files_to_check
    claims.append(
        _claim(
            "SUPPORTED" if files_ok else "UNFOUNDED",
            "Claimed repair/migration/documentation files exist.",
            sorted(files_to_check),
            sorted(existing),
            [],
        )
    )

    immutable_ok = attack_results["immutable_core_fields"]["pass"]
    backfill_ok = attack_results["entry_id_backfill"]["pass"]
    claims.append(
        _claim(
            "SUPPORTED" if (immutable_ok and backfill_ok) else "UNFOUNDED",
            "Trigger blocks closed-trade core field tampering while allowing entry_trade_id NULL->value backfill.",
            {"core_fields_blocked": True, "entry_id_backfill_allowed": True},
            {
                "core_fields_blocked": attack_results["immutable_core_fields"]["pass"],
                "entry_id_backfill_allowed": attack_results["entry_id_backfill"]["pass"],
            },
            [],
        )
    )

    stale_orphan_fails = attack_results["stale_orphan"]["pass"]
    claims.append(
        _claim(
            "SUPPORTED" if stale_orphan_fails else "UNFOUNDED",
            "CI logic fails on NEW stale orphaned positions (>3 days old).",
            True,
            stale_orphan_fails,
            [],
        )
    )

    pragma_bypass = attack_results["pragma_bypass"]["bypassed"]
    claims.append(
        _claim(
            "INCONCLUSIVE",
            "PRAGMA bypass requires admin-level control; hardening is still needed if arbitrary SQL execution is possible.",
            "context dependent",
            {"pragma_bypassed_constraints": pragma_bypass},
            [],
        )
    )

    daily_exit_claim = _claim(
        "INCONCLUSIVE",
        "Daily trader pipeline exit code is now always 0.",
        0,
        "not executed by this verifier (to avoid mutating live trading state)",
        [],
    )
    claims.append(daily_exit_claim)

    if existing_suite.get("available"):
        blocked = existing_suite.get("attacks_blocked")
        bypassed = existing_suite.get("attacks_bypassed")
        suite_match = blocked == 8 and bypassed == 2
        claims.append(
            _claim(
                "SUPPORTED" if suite_match else "UNFOUNDED",
                "Legacy adversarial suite reports 8 blocked attacks and 2 bypassed (view manipulation + PRAGMA).",
                {"attacks_blocked": 8, "attacks_bypassed": 2},
                {"attacks_blocked": blocked, "attacks_bypassed": bypassed},
                [],
            )
        )
    else:
        claims.append(
            _claim(
                "INCONCLUSIVE",
                "Legacy adversarial suite result is available.",
                "suite available",
                existing_suite.get("reason", "unavailable"),
                [],
            )
        )

    return claims


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial verifier for integrity claims.")
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "portfolio_maximizer.db"),
        help="Path to SQLite database.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "logs" / "audit_gate" / "integrity_claims_verification_latest.json"),
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--fail-on-unfounded",
        action="store_true",
        help="Exit 1 if any claim is UNFOUNDED.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 2

    baseline = _query_baseline(db_path)
    enforcer_state = _run_enforcer(db_path)

    attack_results = {
        "stale_orphan": _run_on_snapshot(db_path, _attack_stale_orphan_detection),
        "unlinked_close": _run_on_snapshot(db_path, _attack_unlinked_close_detection),
        "immutable_core_fields": _run_on_snapshot(db_path, _attack_core_field_immutability),
        "entry_id_backfill": _run_on_snapshot(db_path, _attack_entry_id_backfill_allowed),
        "opening_pnl_direct_insert": _run_on_snapshot(db_path, _attack_opening_pnl_direct_blocked),
        "pragma_bypass": _run_on_snapshot(db_path, _attack_pragma_bypass),
    }
    existing_suite = _run_existing_adversarial_suite(db_path)

    claims = _evaluate_claims(
        baseline=baseline,
        enforcer_state=enforcer_state,
        attack_results=attack_results,
        existing_suite=existing_suite,
    )

    supported = sum(1 for c in claims if c["status"] == "SUPPORTED")
    unfounded = sum(1 for c in claims if c["status"] == "UNFOUNDED")
    inconclusive = sum(1 for c in claims if c["status"] == "INCONCLUSIVE")

    report = {
        "timestamp_utc": _utc_now(),
        "db_path": str(db_path),
        "baseline": baseline,
        "enforcer_state": enforcer_state,
        "adversarial_attacks": attack_results,
        "legacy_adversarial_suite": existing_suite,
        "claims": claims,
        "summary": {
            "supported": supported,
            "unfounded": unfounded,
            "inconclusive": inconclusive,
        },
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Integrity Claims Adversarial Verification ===")
    print(f"DB           : {db_path}")
    print(f"Report       : {out_path}")
    print(
        f"Claims       : supported={supported}, unfounded={unfounded}, inconclusive={inconclusive}"
    )
    print(f"Overall      : {enforcer_state['overall_status']}")
    print(
        "Attack checks: "
        f"stale_orphan={attack_results['stale_orphan']['pass']} "
        f"unlinked_close={attack_results['unlinked_close']['pass']} "
        f"immutable={attack_results['immutable_core_fields']['pass']} "
        f"backfill={attack_results['entry_id_backfill']['pass']} "
        f"opening_pnl_blocked={attack_results['opening_pnl_direct_insert']['pass']} "
        f"pragma_bypass={attack_results['pragma_bypass']['bypassed']}"
    )

    if args.fail_on_unfounded and unfounded > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
