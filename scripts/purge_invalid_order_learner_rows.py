#!/usr/bin/env python3
"""
purge_invalid_order_learner_rows.py
----------------------------------
Remove legacy model_order_stats rows that use invalid cache identities
such as generic series labels ("Close", "price", "returns").

Default mode is dry-run. Use --apply to actually delete rows.

Usage:
    python scripts/purge_invalid_order_learner_rows.py
    python scripts/purge_invalid_order_learner_rows.py --apply
    python scripts/purge_invalid_order_learner_rows.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forcester_ts.order_learner import _clean_ticker_key
from integrity.sqlite_guardrails import guarded_sqlite_connect

DB_PATH = ROOT / "data" / "portfolio_maximizer.db"


def _connect(db_path: Path):
    conn = guarded_sqlite_connect(str(db_path), timeout=5.0, enable_guardrails=False)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _find_invalid_rows(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, ticker, model_type, regime, order_params, n_fits, best_aic, last_used
        FROM model_order_stats
        ORDER BY id ASC
        """
    ).fetchall()
    invalid: list[dict[str, Any]] = []
    for row in rows:
        if _clean_ticker_key(row[1]):
            continue
        invalid.append(
            {
                "id": int(row[0]),
                "ticker": str(row[1] or ""),
                "model_type": str(row[2] or ""),
                "regime": str(row[3] or ""),
                "order_params": str(row[4] or ""),
                "n_fits": int(row[5] or 0),
                "best_aic": row[6],
                "last_used": str(row[7] or ""),
            }
        )
    return invalid


def purge_invalid_rows(db_path: Path, *, apply: bool = False) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = _connect(db_path)
    try:
        invalid_rows = _find_invalid_rows(conn)
        deleted = 0
        if apply and invalid_rows:
            cur = conn.executemany(
                "DELETE FROM model_order_stats WHERE id = ?",
                [(row["id"],) for row in invalid_rows],
            )
            deleted = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(invalid_rows))
            conn.commit()

        tickers: dict[str, int] = {}
        for row in invalid_rows:
            ticker = row["ticker"]
            tickers[ticker] = tickers.get(ticker, 0) + 1

        return {
            "db_path": str(db_path),
            "apply": bool(apply),
            "invalid_rows": len(invalid_rows),
            "deleted_rows": deleted,
            "invalid_tickers": tickers,
            "preview": invalid_rows[:10],
        }
    finally:
        conn.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Purge invalid model_order_stats cache rows.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to portfolio_maximizer.db")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete invalid rows. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.db)

    try:
        result = purge_invalid_rows(db_path, apply=bool(args.apply))
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"[ERROR] {exc}")
        return 1

    if args.json:
        print(json.dumps({"ok": True, **result}, default=str))
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[purge_invalid_order_learner_rows] Mode: {mode}")
    print(f"[purge_invalid_order_learner_rows] DB: {result['db_path']}")
    print(f"  invalid_rows: {result['invalid_rows']}")
    print(f"  deleted_rows: {result['deleted_rows']}")
    print(f"  invalid_tickers: {json.dumps(result['invalid_tickers'], sort_keys=True)}")
    if result["preview"]:
        print("  preview:")
        for row in result["preview"]:
            print(
                "    id={id} ticker={ticker} model={model_type} regime={regime} n_fits={n_fits} best_aic={best_aic}".format(
                    **row
                )
            )
    else:
        print("  preview: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
