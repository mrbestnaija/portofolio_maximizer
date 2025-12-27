# MVS Reporting Notes – Portfolio Maximizer v45

**Purpose:** Explain how the Minimum Viable System (MVS) summaries are computed and how to interpret them while iterating toward a real PASS on live/paper data.

## 1. What MVS Summaries Show

The launchers `bash/run_end_to_end.sh` and `bash/run_pipeline_live.sh` call:

```python
from etl.database_manager import DatabaseManager
perf = DatabaseManager().get_performance_summary(start_date=..., end_date=...)
```

and print:

- `Total trades` – realised trades in `trade_executions`.
- `Total profit` – sum of `realized_pnl` (USD).
- `Win rate` – `winning_trades / total_trades`.
- `Profit factor` – `gross_profit / gross_loss`.
- `MVS Status` – PASS only when all are true:
  - `total_profit > 0`
  - `win_rate > 0.45`
  - `profit_factor > 1.0`
  - `total_trades >= 30`

These thresholds mirror `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md` and `Documentation/QUICK_REFERENCE_OPTIMIZED_SYSTEM.md`.

## 2. Controlling the Time Window

By default, the summary is over **full history**. You can narrow it to a recent window via env vars:

- `MVS_START_DATE` – ISO date `YYYY-MM-DD` (optional).
- `MVS_END_DATE` – ISO date `YYYY-MM-DD` (optional).
- `MVS_WINDOW_DAYS` – integer lookback; if set and `MVS_START_DATE` is empty, the scripts compute:
  - `end_date = today (UTC)`
  - `start_date = end_date - MVS_WINDOW_DAYS`

Examples:

```bash
# Last 90 days only
export MVS_WINDOW_DAYS=90
bash/run_end_to_end.sh

# Explicit calendar range
export MVS_START_DATE=2025-01-01
export MVS_END_DATE=2025-12-31
bash/run_pipeline_live.sh
```

The printed `Window` line will show the exact range used.

## 3. How to Use MVS During Iteration

- Early on, expect **MVS = FAIL** – with few trades and/or negative PnL, the system is still in validation.
- Use shorter windows (e.g. `MVS_WINDOW_DAYS=60`) to check whether **recent** changes are improving PnL, even if long-run history is still flat or negative.
- Treat **MVS PASS** as a *minimum* bar before:
  - Increasing allocation or frequency.
  - Relaxing any experimental guards.
  - Considering live capital beyond small paper-trading experiments.

MVS summaries are purely reporting; they do **not** gate execution by themselves. Hard gating remains in:

- `models/time_series_signal_generator.TimeSeriesSignalGenerator` (quant-success gate).
- `scripts/run_auto_trader.py` (LLM readiness gate).

Use the MVS outputs as a concise “should we trust this configuration yet?” signal, aligned with the quantitative success criteria.

## 4. Fast MVS Paper Window Replay (Deterministic)

When you need to quickly generate ≥30 **realised** trades for MVS validation (without waiting weeks of wall-clock paper trading), use:

```bash
python scripts/run_mvs_paper_window.py \
  --tickers AAPL,MSFT,GOOGL \
  --window-days 365 \
  --max-holding-days 2 \
  --entry-momentum-threshold 0.003 \
  --reset-window-trades
```

Notes:
- The runner replays OHLCV already stored in `ohlcv_data` and executes via `PaperTradingEngine`, so rows land in `trade_executions`.
- `PaperTradingEngine` now prefers `signal_timestamp` when present, so replayed trades have historical `trade_date` values (window filters work as expected).
- `SignalValidator` no longer blocks **risk-reducing exits** (SELL closing a long / BUY covering a short), so liquidation and time-based exits are not prevented by trend/regime guardrails.
- A Markdown artifact is written under `reports/` (example: `reports/mvs_paper_window_20251226_183023.md`).

## 5. What Changed for MVS PASS Reliability (Recent)

To drive MVS PASS on *recent paper/live windows* (not just full-history or replay), the system now includes:

- **Stop/target/time exits**: `execution/paper_trading_engine.PaperTradingEngine` stores `stop_loss`, `target_price`, and `forecast_horizon` when opening a position and can close positions even when the incoming signal is `HOLD` (lifecycle exit triggers).
- **Edge-aware cost validation**: `ai_llm/signal_validator.SignalValidator` prefers TS provenance (`signal.provenance.decision_context.net_trade_return` + `roundtrip_cost_*`) for transaction-cost feasibility; it falls back to drift *only* when no edge context is available.
- **bps-accurate cost priors**: `trade_executions` now persists `mid_price` and `mid_slippage_bps` so `scripts/estimate_transaction_costs.py` can emit bps cost stats (commission + mid-slippage) and propose round-trip cost priors for `signal_routing.time_series.cost_model.default_roundtrip_cost_bps`.
