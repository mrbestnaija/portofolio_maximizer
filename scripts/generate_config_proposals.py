#!/usr/bin/env python3
"""
generate_config_proposals.py
----------------------------

Read-only helper that ingests:
  - TS threshold sweep output (ts_threshold_sweep.json), and
  - Transaction cost estimates (transaction_costs.json),
and emits a small "proposal" document with suggested config deltas.

Design goals:
- Do NOT modify any YAML configs directly.
- Keep logic transparent and conservative: proposals are suggestions
  for human review, not automated mutations.
- Produce a compact JSON/YAML artifact that other tools (or agents)
  can turn into config diffs or pull requests.

CURRENT SCOPE (SCAFFOLD):
- Picks, per ticker, the gridpoint with the highest total_profit that
  also satisfies simple PF / win_rate / min_trades constraints.
- Derives default round-trip cost priors (bps) per asset class from
  transaction cost medians so configs can be updated via
  signal_routing.time_series.cost_model.default_roundtrip_cost_bps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").modules["sys"].path:
    __import__("sys").modules["sys"].path.insert(0, str(ROOT_PATH))

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class TickerProposal:
    ticker: str
    confidence_threshold: float
    min_expected_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    annualized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "confidence_threshold": self.confidence_threshold,
            "min_expected_return": self.min_expected_return,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_profit": self.total_profit,
            "annualized_pnl": self.annualized_pnl,
        }


@dataclass
class CostProposal:
    group: str
    trades: int
    roundtrip_cost_median_bps: float
    suggested_roundtrip_cost_bps: float
    commission_median: float = 0.0
    commission_median_bps: float = 0.0
    slippage_median_bps: float = 0.0
    total_cost_median_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "trades": self.trades,
            "roundtrip_cost_median_bps": self.roundtrip_cost_median_bps,
            "suggested_roundtrip_cost_bps": self.suggested_roundtrip_cost_bps,
            "commission_median": self.commission_median,
            "commission_median_bps": self.commission_median_bps,
            "slippage_median_bps": self.slippage_median_bps,
            "total_cost_median_bps": self.total_cost_median_bps,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Input JSON not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc
    return payload or {}


def _select_best_thresholds(
    sweep_payload: Dict[str, Any],
    min_trades: int,
    min_profit_factor: float,
    min_win_rate: float,
) -> List[TickerProposal]:
    # If the sweeper already produced a selection block, honour it.
    selection = sweep_payload.get("selection") or {}
    proposals: List[TickerProposal] = []
    if selection:
        for ticker, row in selection.items():
            proposals.append(
                TickerProposal(
                    ticker=ticker,
                    confidence_threshold=float(row.get("confidence_threshold")),
                    min_expected_return=float(row.get("min_expected_return")),
                    total_trades=int(row.get("total_trades") or 0),
                    win_rate=float(row.get("win_rate") or 0.0),
                    profit_factor=float(row.get("profit_factor") or 0.0),
                    total_profit=float(row.get("total_profit") or 0.0),
                    annualized_pnl=float(row.get("annualized_pnl") or 0.0),
                )
            )
        return proposals

    results = sweep_payload.get("results") or []
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        ticker = str(row.get("ticker") or "UNKNOWN")
        by_ticker.setdefault(ticker, []).append(row)

    proposals = []
    for ticker, rows in by_ticker.items():
        # Filter by constraints.
        candidates = [
            r
            for r in rows
            if int(r.get("total_trades") or 0) >= min_trades
            and float(r.get("profit_factor") or 0.0) >= min_profit_factor
            and float(r.get("win_rate") or 0.0) >= min_win_rate
        ]
        if not candidates:
            continue
        # Choose candidate with highest annualized_pnl (fallback to total_profit), tie-break on PF.
        best = max(
            candidates,
            key=lambda r: (
                float(r.get("annualized_pnl") or 0.0),
                float(r.get("total_profit") or 0.0),
                float(r.get("profit_factor") or 0.0),
            ),
        )
        proposals.append(
            TickerProposal(
                ticker=ticker,
                confidence_threshold=float(best.get("confidence_threshold")),
                min_expected_return=float(best.get("min_expected_return")),
                total_trades=int(best.get("total_trades") or 0),
                win_rate=float(best.get("win_rate") or 0.0),
                profit_factor=float(best.get("profit_factor") or 0.0),
                total_profit=float(best.get("total_profit") or 0.0),
                annualized_pnl=float(best.get("annualized_pnl") or 0.0),
            )
        )
    return proposals


def _derive_cost_proposals(
    cost_payload: Dict[str, Any],
    buffer_bps: float,
) -> List[CostProposal]:
    groups = cost_payload.get("groups") or []
    proposals: List[CostProposal] = []
    for row in groups:
        group = str(row.get("group") or "UNKNOWN")
        trades = int(row.get("trades") or 0)
        commission_median = float(row.get("commission_median") or 0.0)
        commission_median_bps = float(row.get("commission_median_bps") or 0.0)
        slippage_median_bps = float(row.get("slippage_median_bps") or 0.0)
        total_cost_median_bps = float(row.get("total_cost_median_bps") or 0.0)

        roundtrip_cost_median_bps = row.get("roundtrip_cost_median_bps")
        if roundtrip_cost_median_bps is None:
            if total_cost_median_bps:
                roundtrip_cost_median_bps = 2.0 * total_cost_median_bps
            elif commission_median_bps:
                roundtrip_cost_median_bps = 2.0 * commission_median_bps
            else:
                roundtrip_cost_median_bps = 0.0
        roundtrip_cost_median_bps = float(roundtrip_cost_median_bps or 0.0)

        # Apply a small additional safety buffer in bps.
        suggested_roundtrip_cost_bps = max(0.0, roundtrip_cost_median_bps + float(buffer_bps or 0.0))
        proposals.append(
            CostProposal(
                group=group,
                trades=trades,
                roundtrip_cost_median_bps=roundtrip_cost_median_bps,
                suggested_roundtrip_cost_bps=suggested_roundtrip_cost_bps,
                commission_median=commission_median,
                commission_median_bps=commission_median_bps,
                slippage_median_bps=slippage_median_bps,
                total_cost_median_bps=total_cost_median_bps,
            )
        )
    return proposals


@click.command()
@click.option(
    "--ts-sweep-path",
    default="logs/automation/ts_threshold_sweep.json",
    show_default=True,
    help="Path to ts_threshold_sweep.json produced by sweep_ts_thresholds.py.",
)
@click.option(
    "--costs-path",
    default="logs/automation/transaction_costs.json",
    show_default=True,
    help="Path to transaction_costs.json produced by estimate_transaction_costs.py.",
)
@click.option(
    "--min-trades",
    default=10,
    show_default=True,
    help="Minimum trades per ticker required to emit a TS threshold proposal.",
)
@click.option(
    "--min-profit-factor",
    default=1.1,
    show_default=True,
    help="Minimum profit factor for TS threshold proposals.",
)
@click.option(
    "--min-win-rate",
    default=0.5,
    show_default=True,
    help="Minimum win rate (0–1) for TS threshold proposals.",
)
@click.option(
    "--buffer-bps",
    default=5.0,
    show_default=True,
    help="Extra basis points buffer above median commission when deriving "
    "min_expected_return suggestions from transaction costs.",
)
@click.option(
    "--output",
    default="logs/automation/config_proposals.json",
    show_default=True,
    help="Path to write combined proposal JSON.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    ts_sweep_path: str,
    costs_path: str,
    min_trades: int,
    min_profit_factor: float,
    min_win_rate: float,
    buffer_bps: float,
    output: str,
    verbose: bool,
) -> None:
    """
    Generate config proposals from TS threshold sweeps and transaction cost stats.

    The output JSON is intentionally small and descriptive. It is meant to be
    reviewed and then mapped onto actual YAML config diffs (e.g., adjustments
    to config/signal_routing_config.yml and config/quant_success_config.yml).
    """
    _configure_logging(verbose)

    ts_payload = _load_json(Path(ts_sweep_path))
    cost_payload = _load_json(Path(costs_path))

    ticker_proposals = _select_best_thresholds(
        ts_payload,
        min_trades=min_trades,
        min_profit_factor=min_profit_factor,
        min_win_rate=min_win_rate,
    )
    cost_proposals = _derive_cost_proposals(cost_payload, buffer_bps=buffer_bps)

    payload = {
        "meta": {
            "ts_sweep_source": ts_sweep_path,
            "costs_source": costs_path,
            "min_trades": min_trades,
            "min_profit_factor": min_profit_factor,
            "min_win_rate": min_win_rate,
            "buffer_bps": buffer_bps,
        },
        "time_series_thresholds": [p.to_dict() for p in ticker_proposals],
        "transaction_costs": [p.to_dict() for p in cost_proposals],
        "notes": [
            "This file is a proposal only. No configs were modified.",
            "Use these values to craft edits to config/signal_routing_config.yml "
            "(signal_routing.time_series.per_ticker + cost_model.default_roundtrip_cost_bps) "
            "and config/quant_success_config.yml after manual review.",
        ],
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Config proposals written to %s (tickers=%s, cost_groups=%s)",
        out_path,
        len(ticker_proposals),
        len(cost_proposals),
    )


if __name__ == "__main__":
    main()
