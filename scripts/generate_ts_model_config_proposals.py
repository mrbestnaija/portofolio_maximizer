#!/usr/bin/env python3
"""
generate_ts_model_config_proposals.py
-------------------------------------

Generate *proposed* TS forecaster/model-profile config updates from the
ts_model_candidates cache.

- Reads best candidates per (ticker, regime) from ts_model_candidates
  (aggregated via scripts/summarize_ts_candidates.py or directly from DB).
- Emits a small JSON proposal describing preferred candidates and any
  relevant diagnostics (stability, DM vs baseline).

This mirrors the pattern used by scripts/generate_config_proposals.py for
thresholds and costs: proposals are advisory and must be reviewed before
mutating YAML configs (e.g., forecasting_config.yml or model_profiles.yml).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


ROOT_PATH = Path(__file__).resolve().parent.parent


@dataclass
class TSCandidate:
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


def _load_best_candidates(db_path: Path) -> List[TSCandidate]:
    conn = sqlite3.connect(db_path)
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

    best: Dict[Tuple[str, Optional[str]], TSCandidate] = {}
    for row in rows:
        ticker = str(row["ticker"])
        regime = row["regime"]
        key = (ticker, regime)
        score = float(row["score"] or 0.0)
        if key in best and best[key].score >= score:
            continue

        stability_raw = row["stability"]
        stability_val = float(stability_raw) if stability_raw is not None else None

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

        best[key] = TSCandidate(
            ticker=ticker,
            regime=regime,
            candidate_name=str(row["candidate_name"]),
            score=score,
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
    "--min-stability",
    default=0.0,
    show_default=True,
    help="Minimum stability score (0–1) required to emit a proposal.",
)
@click.option(
    "--max-dm-pvalue",
    default=0.10,
    show_default=True,
    help="Maximum DM p-value allowed when preferring a non-baseline candidate.",
)
@click.option(
    "--output",
    default="logs/automation/ts_model_config_proposals.json",
    show_default=True,
    help="Path to write TS model config proposal JSON.",
)
def main(
    db_path: str,
    min_stability: float,
    max_dm_pvalue: float,
    output: str,
) -> None:
    """
    Generate TS model config proposals from ts_model_candidates.

    Proposal semantics:
    - For each (ticker, regime), pick the best candidate by score.
    - Only emit a proposal when:
        * stability >= min_stability (if stability is available), and
        * either DM p-value is None (baseline) or DM p-value <= max_dm_pvalue.
    - Output is advisory; it does not edit any config files.
    """
    path = ROOT_PATH / db_path
    if not path.exists():
        raise SystemExit(f"DB not found: {path}")

    candidates = _load_best_candidates(path)
    if not candidates:
        print("No TS model candidates found; nothing to propose.")
        return

    proposals: List[Dict[str, Any]] = []
    for c in candidates:
        if c.stability is not None and c.stability < min_stability:
            continue
        if c.dm_p_value is not None and c.dm_p_value > max_dm_pvalue:
            continue
        proposals.append(
            {
                "ticker": c.ticker,
                "regime": c.regime or "default",
                "candidate_name": c.candidate_name,
                "score": c.score,
                "stability": c.stability,
                "dm_better_model": c.dm_better_model,
                "dm_p_value": c.dm_p_value,
                "action": "suggest_profile_update",
                "notes": "Map this (ticker, regime) to a model profile or forecaster config consistent with this candidate.",
            }
        )

    payload = {
        "generated_at": Path().stat().st_mtime,
        "db_path": str(path),
        "min_stability": float(min_stability),
        "max_dm_pvalue": float(max_dm_pvalue),
        "proposals": proposals,
    }

    out_path = ROOT_PATH / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"TS model config proposals written to {out_path} ({len(proposals)} entries).")


if __name__ == "__main__":
    main()

