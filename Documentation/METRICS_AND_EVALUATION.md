# Metrics & Evaluation (Definitions + Minimal Math)

**Purpose**: Provide unambiguous, auditable metric definitions for papers/thesis, internal reports, dashboards, and automated gates.

Ground truth implementations live in:

- `etl/portfolio_math.py` (portfolio-level metrics and risk)
- `etl/database_manager.py` (`get_performance_summary()` and trade aggregates)
- `etl/statistical_tests.py` (Diebold–Mariano-style tests and stability metrics)

---

## 1. Returns and Annualization

### 1.1 Log returns (canonical in code)

`etl.portfolio_math.calculate_returns()` defines log returns:

- `r_t = ln(P_t / P_{t-1})`

If you report arithmetic returns, state it explicitly; otherwise assume log returns.

### 1.2 Annualized volatility

For daily returns `r_t`:

- `σ_annual = std(r_t) * sqrt(252)`

### 1.3 Annualized return (geometric)

For portfolio returns series `r_t`:

- `R_total = Π_t (1 + r_t) - 1`
- `R_annual = (1 + R_total)^(252 / T) - 1`

This matches the approach used in `etl/portfolio_math.calculate_enhanced_portfolio_metrics()`.

---

## 2. Core Trading Metrics (Realized Trades)

The canonical trade summary is `DatabaseManager.get_performance_summary()` over `trade_executions` where `realized_pnl IS NOT NULL`.

### 2.1 Win rate

- `win_rate = winning_trades / total_trades`

where `winning_trades = count(realized_pnl > 0)`.

### 2.2 Profit factor (PF)

Profit factor must be computed using **gross** realized PnL:

- `gross_profit = Σ realized_pnl_i over i where realized_pnl_i > 0`
- `gross_loss = |Σ realized_pnl_i over i where realized_pnl_i < 0|`
- `profit_factor = gross_profit / gross_loss` (if `gross_loss > 0`)

If `gross_loss == 0` and `gross_profit > 0`, PF is treated as `+∞` in `etl/database_manager.py`.

---

## 3. Risk-Adjusted Metrics (Portfolio Returns)

Definitions follow `etl/portfolio_math.py`.

### 3.1 Sharpe ratio

- `Sharpe = (R_annual - R_f) / σ_annual`

where `R_f` is the annual risk-free rate (default in code: `DEFAULT_RISK_FREE_RATE = 0.02`).

### 3.2 Sortino ratio

Compute downside deviation using only negative returns:

- `σ_down_annual = std(r_t | r_t < 0) * sqrt(252)`
- `Sortino = (R_annual - R_f) / σ_down_annual`

### 3.3 Max drawdown (MDD)

Let `C_t = Π_{i<=t} (1 + r_i)` be the cumulative curve and `M_t = max_{s<=t}(C_s)` the running peak:

- `DD_t = 1 - C_t / M_t`
- `MDD = max_t(DD_t)`

### 3.4 Calmar ratio

- `Calmar = R_annual / MDD` (when `MDD > 0`)

---

## 4. Tail Risk (VaR / CVaR)

`etl/portfolio_math.calculate_enhanced_portfolio_metrics()` computes a percentile VaR and tail averages:

### 4.1 Value at Risk (VaR)

For daily returns `r_t`:

- `VaR_95 = percentile(r_t, 5)`
- `VaR_99 = percentile(r_t, 1)`

### 4.2 Conditional VaR (CVaR) / Expected Shortfall

Tail mean below the VaR threshold:

- `CVaR_95 = mean(r_t | r_t <= VaR_95)`
- `CVaR_99 = mean(r_t | r_t <= VaR_99)`

The module also reports `expected_shortfall = mean(r_t | r_t < 0)` as a simple negative-return average.

---

## 5. Forecast Evaluation Metrics

When comparing forecasts across horizons, keep definitions consistent:

### 5.1 RMSE

- `RMSE = sqrt(mean((y_t - ŷ_t)^2))`

### 5.2 sMAPE

A robust symmetric mean absolute percentage error:

- `sMAPE = mean( 2 * |y_t - ŷ_t| / (|y_t| + |ŷ_t| + ε) )`

Choose `ε` explicitly to avoid division by zero.

---

## 6. Statistical Comparisons (Model Selection)

### 6.1 Diebold–Mariano (Newey–West variance)

`etl.statistical_tests.diebold_mariano()`:

- Defines loss differential `d_t = L(e1_t) - L(e2_t)` (squared or absolute loss).
- Uses a Newey–West variance estimator on `d_t` with lag `floor(sqrt(T))`.
- Computes a t-statistic and p-value under a t distribution.

Reporting guidance:

- State the loss used (squared/absolute) and the lag heuristic (default `floor(sqrt(T))`).
- Note that assumptions are small-sample NW; if you change lag selection, report it alongside the result.

### 6.2 Rank stability (cross-validation robustness)

`etl.statistical_tests.rank_stability_score()` produces:

- average rank per model across folds,
- a stability score in `[0, 1]` indicating whether model ordering is preserved across folds.

---

## 7. Verification Windows (Avoiding Recency Bias)

For institutional-grade reporting, always report at least:

- full-history realized-trade metrics,
- a recent window (e.g., last 60 days) with trade count,
- a hold-out or walk-forward evaluation window (CV protocol).

When results disagree across windows, treat the discrepancy as the primary research signal (regime dependence, costs, gating strictness), not as something to average away.
