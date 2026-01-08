# Baseline Snapshot Diff

- A: reports/baselines/20260107_055707_pre_phase10
- B: reports/baselines/20260107_161604_post_phase10

## File Changes
### configs
- changed: 0
- added: 0
- removed: 0

### code
- changed: 0
- added: 0
- removed: 0

## Run Metrics (run_summary_last.json)

| metric | A | B | Δ |
|---|---:|---:|---:|
| `forecaster.rmse.baseline` | n/a | n/a | n/a |
| `forecaster.rmse.ensemble` | 15.7319 | 15.7319 | 0 |
| `forecaster.rmse.ratio` | n/a | n/a | n/a |
| `liquidity.cash` | 24737.1 | 24737.5 | 0.373503 |
| `liquidity.cash_ratio` | 0.989505 | 0.989506 | 1.56792e-07 |
| `liquidity.open_positions` | 1 | 1 | 0 |
| `liquidity.total_value` | 24999.5 | 24999.8 | 0.373503 |
| `profitability.pnl_dollars` | -0.525671 | -0.152169 | 0.373503 |
| `profitability.pnl_pct` | -2.10269e-05 | -6.08675e-06 | 1.49401e-05 |
| `profitability.profit_factor` | 0 | 0 | 0 |
| `profitability.realized_trades` | 0 | 0 | 0 |
| `profitability.trades` | 1 | 1 | 0 |
| `profitability.win_rate` | 0 | 0 | 0 |
| `quant.fail_fraction` | 0.793103 | 0.70245 | -0.090653 |
| `quant.negative_expected_profit_fraction` | 0 | 0 | 0 |

## Backtest Metrics (horizon_backtest_latest.json)

| metric | A | B | Δ |
|---|---:|---:|---:|
| `backtest.max_drawdown` | n/a | 0.00313619 | n/a |
| `backtest.profit_factor` | n/a | 0.312723 | n/a |
| `backtest.total_return` | n/a | -83.398 | n/a |
| `backtest.total_trades` | n/a | 8 | n/a |
| `backtest.win_rate` | n/a | 0.375 | n/a |

