#!/usr/bin/env python3
"""
summarize_ts_candidates.py
--------------------------

Helper to inspect the TS hyper-parameter search cache.

- Reads ts_model_candidates from the SQLite DB.
- Aggregates best candidates per (ticker, regime) by score.
- Prints a compact table to stdout and optionally emits JSON.

This is intended as a read-only analysis tool to support institutional-grade
TS model selection and documentation, not as an automated config mutator.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from integrity.sqlite_guardrails import guarded_sqlite_connect


@dataclass
class CandidateSummary:
    ticker: str
    regime: Optional[str]
    candidate_name: str
    score: float
    stability: Optional[float]
    dm_better_model: Optional[str]
    dm_p_value: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "regime": self.regime,
            "candidate_name": self.candidate_name,
            "score": self.score,
            "stability": self.stability,
            "dm_better_model": self.dm_better_model,
            "dm_p_value": self.dm_p_value,
        }


def _load_candidates(db_path: Path) -> List[sqlite3.Row]:
    conn = guarded_sqlite_connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                ticker,
                regime,
                candidate_name,
                score,
                stability,
                metrics
            FROM ts_model_candidates
            ORDER BY ticker, regime, score DESC, created_at DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return rows


def _summarise(rows: List[sqlite3.Row]) -> List[CandidateSummary]:
    best: Dict[Tuple[str, Optional[str]], CandidateSummary] = {}
    for row in rows:
        ticker = str(row["ticker"])
        regime = row["regime"]
        key = (ticker, regime)
        score_val = float(row["score"] or 0.0)
        if key in best and best[key].score >= score_val:
            continue

        stability = row["stability"]
        stability_val = float(stability) if stability is not None else None

        dm_better: Optional[str] = None
        dm_p: Optional[float] = None
        try:
            metrics_payload = json.loads(row["metrics"] or "{}")
            dm_block = metrics_payload.get("dm_vs_baseline") or {}
            dm_better = dm_block.get("better_model")
            dm_p_raw = dm_block.get("p_value")
            if dm_p_raw is not None:
                dm_p = float(dm_p_raw)
        except Exception:
            dm_better = None
            dm_p = None

        best[key] = CandidateSummary(
            ticker=ticker,
            regime=regime,
            candidate_name=str(row["candidate_name"]),
            score=score_val,
            stability=stability_val,
            dm_better_model=dm_better,
            dm_p_value=dm_p,
        )
    return list(best.values())


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite DB path containing ts_model_candidates.",
)
@click.option(
    "--output",
    default=None,
    help="Optional JSON output path for the best-per-(ticker,regime) summary.",
)
def main(db_path: str, output: Optional[str]) -> None:
    """Summarise TS model candidates per (ticker, regime)."""
    path = ROOT_PATH / db_path
    if not path.exists():
        raise SystemExit(f"DB not found: {path}")

    rows = _load_candidates(path)
    if not rows:
        print("No TS model candidates found in ts_model_candidates.")
        return

    summaries = _summarise(rows)
    # Print a simple table.
    header = f"{'Ticker':<10} {'Regime':<12} {'Candidate':<18} {'Score':>10} {'Stab':>6} {'DMBetter':>10} {'DMp':>8}"
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda x: (x.ticker, x.regime or "")):
        regime = s.regime or "default"
        stab_str = f"{s.stability:.3f}" if s.stability is not None else "-"
        dm_better = s.dm_better_model or "-"
        dm_p = f"{s.dm_p_value:.3f}" if s.dm_p_value is not None else "-"
        print(
            f"{s.ticker:<10} {regime:<12} {s.candidate_name:<18} "
            f"{s.score:10.4f} {stab_str:>6} {dm_better:>10} {dm_p:>8}"
        )

    if output:
        out_path = ROOT_PATH / output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": Path().stat().st_mtime,
            "db_path": str(path),
            "candidates": [s.to_dict() for s in summaries],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {out_path}")


if __name__ == "__main__":
    main()
