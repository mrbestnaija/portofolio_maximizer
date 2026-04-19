#!/usr/bin/env python3
"""emit_canonical_snapshot.py

Single source of truth for all PMX performance reporting.
Writes logs/canonical_snapshot_latest.json and prints a human-readable summary.

This is the ONLY file any plan, gate, or documentation is allowed to quote for:
  - Closed PnL / WR / PF          → production_closed_trades view
  - Capital base                   → portfolio_cash_state.initial_capital
  - Time-weighted deployment KPI   → scripts/compute_capital_utilization.py
  - Open risk / open lots          → trade_executions WHERE is_close=0
  - Gate readiness                 → logs/production_gate_latest.json (if present)
  - Unattended readiness           → scripts/institutional_unattended_gate.py --json (if present)

SCHEMA_VERSION = 2. Any consumer must assert schema_version >= 2.
metrics_summary.json is deprecated as a source-of-truth — it is a UI artifact only.

CLI:
    python scripts/emit_canonical_snapshot.py [--db PATH] [--output PATH] [--json]
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_OUTPUT = ROOT / "logs" / "canonical_snapshot_latest.json"
SCHEMA_VERSION = 2


def _gate_artifact_candidates() -> tuple[Path, Path]:
    return (
        ROOT / "logs" / "audit_gate" / "production_gate_latest.json",
        ROOT / "logs" / "production_gate_latest.json",
    )


def _ui_metrics_summary_path() -> Path:
    return ROOT / "visualizations" / "performance" / "metrics_summary.json"


def _query_closed_pnl(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Closed PnL metrics from production_closed_trades (canonical view)."""
    row = conn.execute("""
        SELECT
            COUNT(*)                                          AS n_trips,
            SUM(realized_pnl)                                AS total_pnl,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS n_wins,
            SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) AS gross_profit,
            SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) AS gross_loss,
            MIN(trade_date) AS first_close,
            MAX(trade_date) AS last_close
        FROM production_closed_trades
    """).fetchone()
    if row is None or row[0] == 0:
        return {"n_trips": 0, "total_pnl": 0.0, "win_rate": None, "profit_factor": None,
                "first_close": None, "last_close": None, "source": "production_closed_trades"}
    n, pnl, wins, gp, gl, fc, lc = row
    win_rate = wins / n if n else None
    pf = (gp / gl) if gl and gl > 0 else None
    return {
        "n_trips": n,
        "total_pnl": round(float(pnl or 0), 2),
        "n_wins": wins,
        "gross_profit": round(float(gp or 0), 2),
        "gross_loss": round(float(gl or 0), 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(pf, 3) if pf is not None else None,
        "first_close": fc,
        "last_close": lc,
        "source": "production_closed_trades",
    }


def _query_capital(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Capital base from portfolio_cash_state.initial_capital (single authoritative source)."""
    row = conn.execute(
        "SELECT initial_capital, cash FROM portfolio_cash_state ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"capital": None, "cash": None, "source": "portfolio_cash_state"}
    return {
        "capital": float(row[0]),
        "cash": round(float(row[1]), 2),
        "source": "portfolio_cash_state",
    }


def _query_open_risk(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Open positions risk from trade_executions WHERE is_close=0."""
    rows = conn.execute("""
        SELECT ticker, COUNT(*) AS lots, SUM(shares * price) AS notional
        FROM trade_executions
        WHERE is_close = 0 AND COALESCE(is_synthetic, 0) = 0
        GROUP BY ticker
        ORDER BY notional DESC
    """).fetchall()
    positions = {r[0]: {"lots": r[1], "notional": round(float(r[2] or 0), 2)} for r in rows}
    total_notional = sum(p["notional"] for p in positions.values())
    return {
        "open_positions": positions,
        "total_open_notional": round(total_notional, 2),
        "source": "trade_executions_is_close_0",
    }


def _read_gate_artifact() -> Optional[Dict[str, Any]]:
    """Read production_gate_latest.json if present (not always up-to-date)."""
    for gate_path in _gate_artifact_candidates():
        if not gate_path.exists():
            continue
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("_artifact_path", str(gate_path))
            return payload
    return None


def _run_utilization(db_path: Path, capital: float) -> Optional[Dict[str, Any]]:
    """Delegate to compute_capital_utilization.compute_utilization()."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.compute_capital_utilization import compute_utilization
        return compute_utilization(db_path, capital=capital)
    except Exception as exc:
        return {"error": str(exc)}


def emit_snapshot(db_path: Path) -> Dict[str, Any]:
    """Build and return the canonical snapshot dict."""
    conn = sqlite3.connect(str(db_path))
    try:
        pnl = _query_closed_pnl(conn)
        cap = _query_capital(conn)
        risk = _query_open_risk(conn)
    finally:
        conn.close()

    capital = cap.get("capital")
    util = _run_utilization(db_path, capital) if capital else {"error": "no capital"}

    gate = _read_gate_artifact()
    gate_summary = None
    if gate:
        gate_path = gate.get("_artifact_path") or str(_gate_artifact_candidates()[0])
        gate_summary = {
            "phase3_ready": gate.get("phase3_ready"),
            "posture": gate.get("posture"),
            "phase3_reason": gate.get("phase3_reason"),
            "matched": gate.get("readiness", {}).get("outcome_matched"),
            "eligible": gate.get("readiness", {}).get("outcome_eligible"),
            "artifact_path": str(gate_path),
        }

    # Unattended gate status
    unattended_status = "NOT_CHECKED"
    unattended_path = ROOT / "scripts" / "institutional_unattended_gate.py"
    if unattended_path.exists():
        try:
            res = subprocess.run(
                [sys.executable, str(unattended_path), "--json"],
                capture_output=True, text=True, timeout=30,
                cwd=str(ROOT),
            )
            if res.returncode == 0:
                unattended_status = "PASS"
            else:
                unattended_status = "FAIL"
        except Exception:
            unattended_status = "ERROR"

    ann_roi = util.get("roi_ann_pct") if isinstance(util, dict) else None
    ngn_gap_pp = round(28.0 - ann_roi, 2) if ann_roi is not None else None

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "source_contract": {
            "canonical": {
                "closed_pnl": "production_closed_trades",
                "capital": "portfolio_cash_state.initial_capital",
                "utilization": "scripts.compute_capital_utilization.compute_utilization",
                "open_risk": "trade_executions WHERE is_close=0",
                "gate_artifact": str(_gate_artifact_candidates()[0]),
                "unattended_gate": "scripts/institutional_unattended_gate.py --json",
            },
            "ui_only": {
                "metrics_summary": str(_ui_metrics_summary_path()),
            },
        },
        # ── Canonical metric sections ──
        "closed_pnl": pnl,
        "capital": cap,
        "open_risk": risk,
        "utilization": util,
        "gate": gate_summary,
        # ── Derived summary ──
        "summary": {
            "ann_roi_pct": ann_roi,
            "ngn_hurdle_pct": 28.0,
            "gap_to_hurdle_pp": ngn_gap_pp,
            "unattended_gate": unattended_status,
            "unattended_ready": unattended_status == "PASS",
        },
        # ── Deprecation notice ──
        "_note": (
            "metrics_summary.json is deprecated as a source-of-truth (UI artifact only). "
            "All plans must reference this file for measured metrics."
        ),
    }


@click.command()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="SQLite DB path")
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True, help="Output JSON path")
@click.option("--json", "as_json", is_flag=True, help="Print JSON to stdout only")
def main(db: str, output: str, as_json: bool) -> None:
    snapshot = emit_snapshot(Path(db))

    if as_json:
        print(json.dumps(snapshot, indent=2))
        return

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    s = snapshot["summary"]
    pnl = snapshot["closed_pnl"]
    cap = snapshot["capital"]
    util = snapshot.get("utilization") or {}
    gate = snapshot.get("gate") or {}

    print("Canonical PMX Snapshot")
    print(f"  Schema version     : {snapshot['schema_version']}")
    print(f"  Capital base       : ${cap.get('capital', 'N/A'):,.0f}  (cash: ${cap.get('cash', 0):,.0f})")
    print(f"  Closed trades      : {pnl['n_trips']}  WR={pnl['win_rate'] or 0:.1%}  "
          f"PF={pnl['profit_factor'] or 0:.2f}  PnL=${pnl['total_pnl']:+,.2f}")
    if isinstance(util, dict) and "ann_roi_pct" in util:
        print(f"  Ann ROI            : {util['ann_roi_pct']:.1f}%  "
              f"({util.get('trades_per_day', 0):.2f} trades/day, "
              f"{util.get('deployment_pct', 0):.1f}% capital deployed/day)")
    print(f"  NGN hurdle gap     : {s['gap_to_hurdle_pp']:+.1f}pp  (hurdle=28%)")
    print(f"  Unattended gate    : {s['unattended_gate']}")
    if gate:
        print(f"  Phase3 posture     : {gate.get('posture')}  "
              f"matched={gate.get('matched')}/{gate.get('eligible')}")
    print(f"\nArtifact: {output}")


if __name__ == "__main__":
    main()
