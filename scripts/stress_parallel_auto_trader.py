#!/usr/bin/env python3
"""
stress_parallel_auto_trader.py
------------------------------

Run the auto-trader for a single cycle in sequential vs parallel mode and compare
aggregated execution outputs. Intended for manual/opt-in stress validation with
real market data access.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from scripts import run_auto_trader


@dataclass
class RunSummary:
    label: str
    exec_log: Path
    counts: Dict[str, int]
    pairs: List[Tuple[str, str]]


def _read_execution_log(path: Path) -> RunSummary:
    counts: Dict[str, int] = {}
    pairs: List[Tuple[str, str]] = []
    if not path.exists():
        return RunSummary(label=path.stem, exec_log=path, counts={}, pairs=[])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = str(payload.get("status") or "UNKNOWN")
        ticker = str(payload.get("ticker") or "UNKNOWN").upper()
        counts[status] = counts.get(status, 0) + 1
        pairs.append((ticker, status))
    return RunSummary(label=path.stem, exec_log=path, counts=counts, pairs=pairs)


def _run_once(
    *,
    label: str,
    tickers: str,
    lookback_days: int,
    forecast_horizon: int,
    initial_capital: float,
    parallel: bool,
    workers: int | None,
    out_dir: Path,
) -> RunSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    exec_log = out_dir / f"execution_{label}.jsonl"
    run_summary = out_dir / f"run_summary_{label}.jsonl"
    dashboard_json = out_dir / f"dashboard_{label}.json"
    dashboard_png = out_dir / f"dashboard_{label}.png"

    # Redirect outputs to the stress-run directory.
    run_auto_trader.EXECUTION_LOG_PATH = exec_log
    run_auto_trader.RUN_SUMMARY_LOG_PATH = run_summary
    run_auto_trader.DASHBOARD_DATA_PATH = dashboard_json
    run_auto_trader._emit_dashboard_png = lambda *args, **kwargs: None  # type: ignore

    # Isolate DB + parallel flags.
    os.environ["PORTFOLIO_DB_PATH"] = str(out_dir / f"portfolio_{label}.db")
    os.environ["ENABLE_PARALLEL_TICKER_PROCESSING"] = "1" if parallel else "0"
    os.environ["ENABLE_PARALLEL_FORECASTS"] = "1" if parallel else "0"
    if workers:
        os.environ["PARALLEL_TICKER_WORKERS"] = str(workers)
    else:
        os.environ.pop("PARALLEL_TICKER_WORKERS", None)

    # Call the click callback directly.
    callback = getattr(run_auto_trader.main, "callback", None)
    if callback is None:
        raise RuntimeError("run_auto_trader.main.callback not found; click signature changed")

    t0 = time.perf_counter()
    callback(
        tickers=tickers,
        include_frontier_tickers=False,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
        initial_capital=initial_capital,
        cycles=1,
        sleep_seconds=0,
        bar_aware=True,
        persist_bar_state=False,
        bar_state_path=str(out_dir / "bar_state.json"),
        enable_llm=False,
        llm_model="",
        verbose=False,
        yfinance_interval=None,
    )
    elapsed = time.perf_counter() - t0

    summary = _read_execution_log(exec_log)
    summary.label = label
    summary.counts["elapsed_seconds"] = round(elapsed, 4)
    return summary


def _compare_outputs(seq: RunSummary, par: RunSummary) -> Dict[str, object]:
    seq_pairs = sorted(seq.pairs)
    par_pairs = sorted(par.pairs)
    same = seq_pairs == par_pairs and seq.counts == par.counts
    return {
        "matches": bool(same),
        "sequential_counts": seq.counts,
        "parallel_counts": par.counts,
        "sequential_pairs": seq_pairs,
        "parallel_pairs": par_pairs,
        "sequential_elapsed_seconds": seq.counts.get("elapsed_seconds"),
        "parallel_elapsed_seconds": par.counts.get("elapsed_seconds"),
    }


def main() -> int:
    tickers = os.getenv("STRESS_TICKERS", "AAPL,MSFT,GOOG,NVDA,AMZN,TSLA")
    lookback_days = int(os.getenv("STRESS_LOOKBACK_DAYS", "120"))
    forecast_horizon = int(os.getenv("STRESS_FORECAST_HORIZON", "7"))
    initial_capital = float(os.getenv("STRESS_INITIAL_CAPITAL", "25000"))
    workers = os.getenv("PARALLEL_TICKER_WORKERS")
    workers_int = int(workers) if workers and workers.isdigit() else None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("logs") / "automation" / f"stress_parallel_{ts}"

    t0 = time.perf_counter()
    seq = _run_once(
        label="sequential",
        tickers=tickers,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
        initial_capital=initial_capital,
        parallel=False,
        workers=workers_int,
        out_dir=out_dir,
    )
    par = _run_once(
        label="parallel",
        tickers=tickers,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
        initial_capital=initial_capital,
        parallel=True,
        workers=workers_int,
        out_dir=out_dir,
    )
    t1 = time.perf_counter()

    comparison = _compare_outputs(seq, par)
    comparison["elapsed_seconds"] = t1 - t0
    comparison["tickers"] = tickers
    comparison["lookback_days"] = lookback_days
    comparison["forecast_horizon"] = forecast_horizon

    out_path = out_dir / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Wrote comparison to {out_path}")
    return 0 if comparison["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
