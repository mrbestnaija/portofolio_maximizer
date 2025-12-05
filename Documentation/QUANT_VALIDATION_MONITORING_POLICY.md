# Quant Validation Monitoring & Threshold Policy

**Version**: 0.1  
**Scope**: Time-series forecaster / signal quant validation (`logs/signals/quant_validation.jsonl`) and brutal/CI health checks.  
**Status**: Draft calibration memo – codifies current behaviour and outlines how to tighten it.

This document explains how the current quant validation monitoring thresholds work, how they map to GREEN / YELLOW / RED states, and how they should be interpreted in the context of Portfolio Maximizer’s “evidence before complexity” philosophy.

The goal is twofold:

- Make the behaviour of `config/forecaster_monitoring.yml`, `scripts/check_quant_validation_health.py`, and `scripts/summarize_quant_validation.py` fully explicit.
- Provide a roadmap for tightening thresholds from “research hygiene” to “production gates” as empirical data accumulates.

---

## 1. Current Monitoring Configuration

The relevant section of `config/forecaster_monitoring.yml` is:

```yaml
forecaster_monitoring:
  quant_validation:
    # Production thresholds used for per-ticker alerts and
    # GREEN tier classification in summarize_quant_validation.py.
    min_profit_factor: 1.1
    min_win_rate: 0.52
    min_annual_return: 0.0
    min_pass_rate: 0.05

    # Global ceilings used by check_quant_validation_health.py
    # to gate CI/brutal runs. With these values, a global
    # FAIL fraction above 90% or more than half of entries
    # having negative expected_profit will trip the hard gate
    # and mark the run RED.
    max_fail_fraction: 0.90
    max_negative_expected_profit_fraction: 0.5

    # Softer global warning band used to classify runs as
    # YELLOW when they are better than the hard RED gate but
    # still outside the comfortable GREEN zone. These do not
    # cause CI/brutal to fail, but should be treated as
    # "research / needs attention" in dashboards.
    warn_fail_fraction: 0.80
    warn_negative_expected_profit_fraction: 0.40

    # Softer "research-ok" thresholds used for YELLOW tier.
    # Regimes that clear these but fail the production bars
    # can be treated as promising but not yet production ready.
    warn_profit_factor: 1.0
    warn_win_rate: 0.48
    warn_annual_return: -0.05
    warn_pass_rate: 0.02

  per_ticker:
    AAPL:
      min_profit_factor: 1.15
      min_win_rate: 0.53
      min_annual_return: 0.0
      min_pass_rate: 0.02
      warn_profit_factor: 1.05
      warn_win_rate: 0.50
      warn_annual_return: -0.02
      warn_pass_rate: 0.03
    MSFT:
      # Same pattern as AAPL (slightly stricter than global)
      ...
```

Interpretation assumptions:

- `min_*` thresholds represent **production bars** for a single ticker/model:
  - If median metrics and pass rate are below `min_*`, the regime is **not production-ready** (RED at the ticker level).
- `warn_*` thresholds represent a softer **research / warning band**:
  - Between warn and min = YELLOW – allowed for research and regression, but not yet production-grade.
- `min_pass_rate` and `warn_pass_rate` refer to the fraction of CV windows/regimes where the model satisfies success criteria.
- Global `max_*` / `warn_*` ceilings (fail fraction, negative expected profit fraction) are used for **brutal / CI gating**, not per-ticker status.

This separation gives three conceptual layers:

1. **Per-ticker** GREEN/YELLOW/RED (summaries & routing hints).  
2. **Global** GREEN/YELLOW/RED (brutal/CI health).  
3. Execution/routing rules that decide which tiers are allowed to drive trades.

---

## 2. Per-Ticker Classification (summarize_quant_validation.py)

`scripts/summarize_quant_validation.py` reads `logs/signals/quant_validation.jsonl` and prints:

- Global counts of PASS / FAIL for quant validation.
- A breakdown of the most common failed criteria.
- Per-ticker aggregates:
  - Count, PASS, FAIL.
  - Median profit factor, win rate, annual return.
  - A **Tier** field: `GREEN`, `YELLOW`, `RED`.
  - An **Alerts** field showing which production thresholds are blocking GREEN.

For each ticker:

- Global thresholds (`min_*`, `warn_*`) are read from `forecaster_monitoring.quant_validation`.
- Per-ticker overrides (e.g. for AAPL, MSFT, BTC-USD) are applied on top.
- Median metrics are computed from the quant log for that ticker:
  - `median_profit_factor`, `median_win_rate`, `median_annual_return`.
  - `pass_rate = PASS / (PASS + FAIL)`.

Tier classification logic:

- Let `(min_pf, min_wr, min_ar, min_pass_rate)` be the production thresholds.
- Let `(warn_pf, warn_wr, warn_ar, warn_pass_rate)` be the warning thresholds.
- If median metrics and pass_rate meet **all** production thresholds:
  - `Tier = GREEN` (production-ready, from a quant perspective).
- Else, if they fail production but meet all warn thresholds:
  - `Tier = YELLOW` (promising, research / pre-production).
- Else:
  - `Tier = RED` (quantitatively weak; treat as research-only or disabled).

Alerts are keyed only off production thresholds (`min_*`) so operators can quickly see what is blocking GREEN (e.g. `PF<min`, `AnnRet<min`, `PASS_rate<min`).

### 2.1 Suitability of current per-ticker thresholds

At current values:

- `min_profit_factor = 1.1` is just above break-even:
  - Suitable as a **research “doesn’t obviously suck”** floor.
  - Too weak as a sole gate for real capital allocation; most systematic desks would want PF nearer 1.5–1.75 in backtest for strategies that will face live slippage and degradation.
- `min_win_rate = 0.52` is a mild sanity check:
  - Without conditioning on payoff asymmetry (average win vs average loss), WR alone is not a strong edge test.
  - A 52% win rate only becomes statistically convincing if each fold has hundreds of trades; otherwise confidence intervals overlap 50% heavily.
- `min_annual_return = 0.0` allows non-negative annualised returns:
  - Aligns with “keep only non-negative strategies in research”.
  - Conflicts with the project’s broader guardrail that **new complexity should only be justified by clearly profitable baselines** (e.g. >8–10% annualised).
- `min_pass_rate = 0.05` (with `min_pass_rate` defined as fraction of windows that pass success criteria):
  - With 20 CV windows, this requires only 1 window to pass.
  - This is extremely tolerant and should be treated as a **research-only robustness floor**, not a production bar.

For per-ticker overrides like AAPL and MSFT, the pattern is:

- Slightly stricter `min_profit_factor` and `min_win_rate` than global (e.g. 1.15 and 0.53 respectively).
- `min_annual_return` still 0.0.
- `min_pass_rate` even lower than some `warn_pass_rate` values (e.g. AAPL: `min_pass_rate=0.02`, `warn_pass_rate=0.03`), which is logically confusing and should be normalised (see section 4).

The upshot: the current thresholds are **good as research hygiene filters** (remove obviously bad regimes) but **too lenient to be treated as final production gates** for NAV-critical systems.

---

## 3. Global Health Classification (check_quant_validation_health.py)

`scripts/check_quant_validation_health.py` is the CI/brutal helper. It inspects `logs/signals/quant_validation.jsonl` and enforces global health thresholds derived from `forecaster_monitoring.quant_validation`:

1. Computes global totals:
   - `total`, `pass_count`, `fail_count`.
   - `fail_fraction = FAIL / total`.
   - `negative_expected_profit_fraction` = fraction of entries with `expected_profit < 0`.
2. Reads ceilings and warning band from config or CLI flags:
   - Hard ceilings:
     - `max_fail_fraction`.
     - `max_negative_expected_profit_fraction`.
   - Warning band:
     - `warn_fail_fraction`.
     - `warn_negative_expected_profit_fraction`.
3. Prints a summary and classifies the run globally as GREEN / YELLOW / RED:
   - **RED**:
     - Any hard ceiling violated (e.g. `fail_fraction > max_fail_fraction` or `negative_expected_profit_fraction > max_negative_expected_profit_fraction`).
     - Script prints `Global health classification   : RED` and exits with code `1` – CI/brutal must treat this as a failure.
   - **YELLOW**:
     - No hard ceiling violated, but either `fail_fraction > warn_fail_fraction` or `negative_expected_profit_fraction > warn_negative_expected_profit_fraction`.
     - Script prints `Global health classification   : YELLOW` and a warning, but exits with `0` – advisory only.
   - **GREEN**:
     - Both metrics below the warning band.
     - Script prints `Global health classification   : GREEN` and exits with `0`.

The brutal harness (`bash/comprehensive_brutal_test.sh`) now captures this classification and includes a line in `final_report.md`:

- `**Quant validation health (global)**: GREEN | YELLOW | RED`

This makes each brutal run self-describing with respect to quant health.

### 3.1 Interpretation of current global ceilings

With the current configuration:

- `max_fail_fraction = 0.90`:
  - Brutal/CI will mark runs **RED** when more than 90% of quant regimes fail.
  - Example: a run with 8.3% PASS and 91.7% FAIL is RED (current state).
  - This is intentionally strict: the global system should not be considered healthy when almost all quant validations fail.
- `max_negative_expected_profit_fraction = 0.50`:
  - At most half of the entries may have negative expected profit; otherwise the run is RED.
  - This is a baseline sanity check for positive expectation in most windows.
- Warning band (`warn_fail_fraction = 0.80`, `warn_negative_expected_profit_fraction = 0.40`):
  - A run where ~85% of regimes fail is YELLOW – better than catastrophic, but still “research / needs attention”.
  - A run with ~75% fail but only 30% negative expected profit might be GREEN, indicating overall acceptable health with localized weaknesses.

This design separates:

- **Hard CI/brutal gate (RED)** – strict, failure conditions.  
- **Advisory monitoring band (YELLOW)** – flags that the ensemble is not yet comfortable but improving.  
- **Healthy operation (GREEN)** – global metrics are robustly within limits.

---

## 4. Calibration Guidance & Future Tightening

The current thresholds are intentionally conservative in logic but lenient in numerics to avoid overfitting to early data. As more cross-validated and live results accumulate, they should be tightened using empirical calibration rather than ad hoc guesses.

Key calibration ideas:

1. **Profit Factor (PF)**:
   - Treat `min_profit_factor ≈ 1.1` as a **research hard floor** (exclude obviously bad strategies).
   - For production-grade regimes, aim to gradually move toward `min_profit_factor_prod ≈ 1.4–1.7` once you have enough historical evidence.
   - For per-ticker overrides on core tickers (AAPL, MSFT), make PF thresholds strictly higher than global.
2. **Win Rate (WR)**:
   - Maintain WR as a secondary check; don’t overinterpret small improvements (e.g. 0.52 vs 0.5) unless there are hundreds of trades per fold.
   - Consider adding a binomial confidence-based rule (e.g. lower 95% CI of WR > 0.5) for a more statistically grounded gate.
3. **Annual Return**:
   - Split research vs production explicitly in config:
     - `min_annual_return_research: 0.0` (keep strategies with non-negative annualised return).
     - `min_annual_return_prod: 0.08–0.10` (only promote regimes with clearly positive edge, consistent with project guardrails).
4. **Robustness via pass_rate**:
   - Redefine `min_pass_rate` conceptually as a robustness metric (“fraction of windows where all production thresholds are met”).
   - Tighten:
     - Research: `min_pass_rate_research ≈ 0.3`.
     - Production: `min_pass_rate_prod ≈ 0.6–0.7`.
   - Avoid contradictory configurations where `warn_pass_rate > min_pass_rate`; enforce monotonicity at load time if possible.
5. **Negative expected profit fraction**:
   - For research, `max_negative_expected_profit_fraction ≈ 0.5` is acceptable.
   - For production, aim to move toward ~0.3 (i.e. at least 70% of windows show positive expectation).
6. **Risk-adjusted metrics**:
   - Consider extending monitoring config to include:
     - `min_sharpe_ratio` or t-stat on returns.
     - `max_drawdown` or related drawdown limits.
   - Time-series quant validation already logs Sharpe, Sortino, max drawdown per regime; wiring these into monitoring thresholds will align the gates with the broader success criteria in `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md`.

These calibrations can be added incrementally, always anchored to **empirical distributions from your own backtests** (calibrate quantiles for “good” vs “bad” strategies in hindsight) rather than borrowed rules of thumb.

---

## 5. Practical Next Steps

Short-term, without changing any routing logic, the system already supports a useful semantic split:

- **Per-ticker**:
  - Use the existing `GREEN / YELLOW / RED` tiers from `summarize_quant_validation.py` to label regimes as production-ready (GREEN) vs. still-research (YELLOW/RED).
  - Ensure execution/routing only ever uses GREEN-tier TS regimes for live trades, with YELLOW/RED contributing only to research and monitoring.
- **Global**:
  - Treat `check_quant_validation_health.py`’s global classification as the brutal/CI gate:
    - RED = fail the suite; do not consider the system ready.
    - YELLOW = CI technically passes but overall quant health is still in a research/transition state.
    - GREEN = ensemble is robust enough to consider widening live usage (subject to other checks).

Medium-term, as more data accumulates:

- Introduce explicit `*_research` vs `*_prod` fields in `forecaster_monitoring.yml` and wire them into both the per-ticker summariser and the global health check.
- Enforce monotonic relationships between warn, research, and production thresholds (no “warn > min” inversions).
- Backfill documentation references:
  - `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md` for portfolio-level KPIs.
  - `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` for how TS forecaster metrics are logged and interpreted.

This keeps quant validation monitoring strictly configuration-driven, visible in brutal/CI outputs, and aligned with the project’s overall “no complexity without proven profitability” philosophy.

