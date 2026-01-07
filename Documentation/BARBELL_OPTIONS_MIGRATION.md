# Migrating From Spot Assets to Options/Derivatives for Barbell Strategy Use

**Status**: Design + Initial Config/Policy (Phase O1 in progress)  
**Scope**: Portfolio Maximizer v45 – migration path from pure spot (stocks/commodities) to options / derivative / synthetic exposures suitable for Taleb-style barbell strategies.

This document describes how to extend the current spot‑only stack to:

- Store and process options/derivative data,
- Implement a barbell allocation (safe leg + convex risk leg),
- Use cheap OTM options or synthetic convexity as the primary risk‑bucket instrument.

It is implementation-oriented but does **not** change any guardrail: risk remains configuration-driven and quant-validated. The concrete implementation is gated behind:

- `options_trading.enabled: true` in `config/options_config.yml`, and
- `barbell.enable_barbell_allocation: true` in `config/barbell.yml`.

When these flags are left `false` (current default), the options/barbell paths remain inert and the spot-only system behaves as before. For **NAV-centric barbell risk budgeting and bucket-level NAV caps**, see `Documentation/NAV_BAR_BELL_TODO.md`.
For how diagnostic mark-to-market and liquidation behave for options and synthetic structures during research runs, see `Documentation/MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md` (and the current pricing_policy assumptions in `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md`).

**Project-wide sequencing**: See `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` for the TS-first execution + reporting improvements that should be validated before enabling options/barbell features in any live-like environment.

---

## 1. OTM Options – Clear Primer for This Project

### 1.1 Simple definition

An **Out‑of‑The‑Money (OTM) option** is an option that has **no intrinsic value right now**. If you exercised it immediately, the trade would not be profitable.

- For OTM options, the **strike price is in an unfavourable position** relative to the current underlying price.
- Their price (premium) is therefore **purely time value / extrinsic value** – a bet that the underlying will move enough before expiry.

### 1.2 When is an option OTM?

It depends on whether it is a **Call** or a **Put**:

| Option Type | When is it OTM? | Quick intuition | Example (Stock @ $100) |
| :---------- | :-------------- | :-------------- | :---------------------- |
| **Call**    | Stock price **below** strike | You would not buy at strike because the market is cheaper. | $110 Call is OTM (100 < 110). |
| **Put**     | Stock price **above** strike | You would not sell at strike because the market pays more. | $90 Put is OTM (100 > 90). |

### 1.3 Key characteristics of OTM options

1. **Cheap premium**  
   - OTM options cost far less than ATM/ITM because there is no intrinsic value.
   - You can control large notional with small capital.

2. **Purely directional / volatility bets**  
   - Their entire value comes from **time + volatility + probability** of moving ITM before expiry.
   - You buy OTM **Calls** when you expect **strong upside moves**.
   - You buy OTM **Puts** when you expect **strong downside moves**.

3. **Lower probability of profit**  
   - The underlying must move a larger distance to make the option profitable.
   - Statistically, OTM options have a **lower chance of finishing ITM**.

4. **High leverage, high risk**  
   - Maximum loss for a long option is the **entire premium**, which can easily go to zero.
   - Upside, especially for Calls, can be very large relative to the cost if a big move happens.

### 1.4 Example in project terms

Assume `AAPL` spot in our database trades at **$150**.

- **OTM Call** – `AAPL 155C` costing $2.00:
  - Spot ($150) < Strike ($155) ⇒ OTM.
  - Break‑even at expiry ≈ $157 ($155 + $2 premium).
  - In the risk leg of the barbell, this is a **cheap convex bet**: small known loss if nothing happens, big payoff if AAPL explodes upward.

- **OTM Put** – `AAPL 145P` costing $1.50:
  - Spot ($150) > Strike ($145) ⇒ OTM.
  - Break‑even at expiry ≈ $143.50 ($145 − $1.50 premium).
  - In the barbell, this is a **cheap downside tail hedge** against a crash.

For Portfolio Maximizer, these become **canonical instruments in the risk bucket**: tiny capital allocations with strongly convex payoffs.

---

## 2. Barbell Strategy Mapping for Options/Derivatives

We adopt the Taleb barbell structure:

- **Safe leg** (70–95% of capital):
  - Cash / T‑bill proxies, short‑duration sovereigns, very low‑vol ETFs.
  - Potentially some large, stable spot positions (e.g. broad index ETFs) with tight risk caps.

- **Risk leg** (5–30% of capital):
  - Long OTM options (calls and puts),
  - Long volatility structures (e.g. straddles/strangles in a later phase),
  - Synthetic convex exposures where listed options are unavailable (e.g. frontier/NGX).

At system level this means:

1. Add an **options/derivatives data layer** alongside existing OHLCV spot data.
2. Tag instruments as `safe` or `risk` in configuration (`config/barbell.yml`, universe metadata).
3. Route **all long OTM option strategies into the risk leg** and enforce barbell caps via policy:
   - Global barbell shell via `risk/barbell_policy.BarbellConstraint` + `config/barbell.yml`.
   - Options-specific caps via `config/options_config.yml` (`max_options_weight`, `max_premium_pct_nav`).

---

## 3. Data Model Migration: From Spot OHLCV to Options/Derivatives

### 3.1 New instrument dimensions

Extend the current universe and database schema to represent option contracts explicitly:

- Core fields:
  - `underlying_symbol` (e.g. `AAPL`),
  - `option_symbol` / `contract_id`,
  - `option_type` (`CALL` / `PUT`),
  - `strike_price`,
  - `expiry_date`,
  - `exercise_style` (`American` / `European`),
  - `multiplier` (e.g. 100 per contract).

- Market data fields (per contract per day or per bar):
  - `bid`, `ask`, `last`, `volume`, `open_interest`,
  - `implied_volatility` (if provided or derived),
  - Greeks where available: `delta`, `gamma`, `theta`, `vega`, `rho`.

These sit logically beside existing OHLCV tables:

- `ohlcv_spot` – current table (stocks/commodities/FX).
- `options_quotes` – new table keyed by `(underlying, contract_id, date)`.

### 3.2 ETL implications

1. **Data source configuration**
   - Extend `config/data_sources_config.yml` to mark providers that support options (e.g. yfinance).
   - Add options‑related endpoints or flags for each provider.

2. **Extractors**
   - Add an `OptionsExtractor` or extend `yfinance_extractor` with:
     - `fetch_option_chain(symbol, expiry)` capability.
     - Normalisation into the shared schema above.

3. **Storage**
   - New writer in `etl/data_storage.py` (e.g. `save_options_quotes`) that writes to `data/options/` and into `options_quotes` in SQLite.

4. **Preprocessing**
   - Handle:
     - Missing IV/Greeks (compute from prices where feasible, or flag as unavailable),
     - Corporate actions affecting options chains (splits, special dividends).

5. **Validation**
   - Extend `DataValidator` with checks:
     - No negative prices / volume,
     - Strike/expiry sanity (expiry > trade date, strike > 0),
     - Bid ≤ Ask, spreads within configured bounds.

---

## 4. Synthetic & Derivative Exposures When Listed Options Are Limited

In some markets (e.g. NGX / certain frontier exchanges) listed options may be absent or illiquid. For those cases:

1. **Synthetic options using spot + risk rules**
   - Use leverage, stop‑loss, and take‑profit bands to approximate option‑like payoffs:
     - E.g. small notional leveraged long with hard stop‑loss at a fixed % (bounded downside).
   - Track these trades using **virtual strikes/expiries** in metadata so they can still be analysed like options.

2. **Volatility proxies**
   - Use high‑volatility ETFs, leveraged products, or baskets as **crude long‑vol instruments** where explicit options aren’t available.

3. **Configuration**
   - Mark these synthetic instruments as `risk_class: risk` in the universe config and route them into the risk leg just like true options.

---

## 5. Barbell Allocation Logic with Options

The existing higher‑order hyperopt driver (`bash/run_post_eval.sh`) and barbell policy can be extended to options with the following rules:

1. **Instrument tagging**
   - In a universe config (or barbell config), tag:
     - Safe leg: cash/T‑bill proxies, short‑duration bond ETFs, large index ETFs.
     - Risk leg: all long OTM options, synthetic convex exposures, high‑volatility derivatives.

2. **Weight constraints**
   - Enforce:
     - `safe_weight ∈ [α_min, α_max]` (e.g. 0.75–0.95),
     - `risk_weight ≤ 1 − α_min` (e.g. ≤ 0.25),
     - Optional per‑market caps (e.g. NGX options ≤ 5% NAV).

3. **Risk leg sizing**
   - For options specifically, position sizing uses **premium at risk**, not notional:
     - Risk per trade = premium / NAV,
     - Sum of option premiums ≤ configured `risk_premium_budget` (e.g. 2–5% of NAV over a horizon).

4. **Hyper‑parameter tuning**
   - The higher‑order hyperopt loop can include:
     - Bands for **option moneyness** (e.g. 5–20% OTM),
     - Expiry windows (e.g. 30–90 days),
     - Risk budget per underlying or per asset class.

All of these constraints remain **configuration‑driven** and can be tuned by the existing hyperopt machinery without changing core trading logic.

---

## 6. Pipeline & Module Impact Summary

**ETL**
- Add options extraction to `etl/data_source_manager.py` and an `OptionsExtractor` wrapper.
- New Parquet/SQLite storage paths under `data/options/` and `options_quotes` table.

**Signal generation & routing**
- Extend `models/time_series_signal_generator` and/or a new `options_signal_generator` to:
  - Take forecasts on underlying assets,
  - Map them to OTM call/put selections (strike, expiry, side),
  - Respect barbell risk budgets.
- Route options signals through the same `SignalRouter`, but with:
  - Hard mapping to the risk bucket,
  - Additional quant checks on leverage and probability of profit.

**Quant success criteria**
- Extend `config/quant_success_config.yml` to:
  - Track metrics that make sense for options:
    - Premium at risk, gamma/vega exposure, frequency of full premium loss.
  - Accept lower win rates but require:
    - Strong positive skew and tail ratio,
    - Acceptable drawdown at the **portfolio** level once barbell is applied.

**Optimization**
- Use `strategy_optimization_config.yml` to:
  - Include option‑specific knobs in the search space:
    - Moneyness bands, expiry buckets, risk budget per trade.
  - Keep guardrails (max drawdown, profit factor) enforced at the **total portfolio** level.

---

## 7. Migration Phases

1. **Phase O1 – Data & Schema**
   - Implement option chain ETL and `options_quotes` storage.
   - Validate data quality and integrate basic visualizations (IV surfaces, moneyness ladders).

2. **Phase O2 – Synthetic & Risk Bucket Pilot**
   - Define a small set of long OTM call/put strategies on liquid underlyings.
   - Plug them into the existing barbell + hyperopt stack with small risk budgets.

3. **Phase O3 – Frontier / NGX Extension**
   - Design synthetic convexity for markets without listed options.
   - Encode them as risk instruments with clear risk caps in config.

4. **Phase O4 – Full Barbell with Options**
   - Safe leg: cash/bonds/large index ETFs.
   - Risk leg: portfolio of long OTM options and synthetic convex structures, sized and tuned via higher‑order hyper‑parameter search.

Each phase should go through the brutal/test harness with:

- Spot‑only baseline,
- Spot + options comparison,
- Tail‑event stress testing (volatility amplification and jump scenarios).

Only after options/derivatives show acceptable behaviour under the barbell and quant‑success criteria should they be promoted to live/paper trading profiles.
