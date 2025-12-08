# Mark-to-Market & Liquidation Implementation Plan

**Last updated**: 2025-11-20  
**Scope**: Enhancing `scripts/liquidate_open_trades.py` from a spot-only, yfinance-dependent helper into a configurable, asset-class-aware MTM tool that can support equities, crypto, options, and synthetic exposures under the barbell architecture.

This plan is sequenced and intended to be kept in sync with the codebase. It builds on the design discussion captured in the prompt text and aligns with:

- `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md`
- `Documentation/NAV_RISK_BUDGET_ARCH.md`
- `Documentation/BARBELL_INTEGRATION_TODO.md`

---

## Phase 1 – Robust Spot-Only MTM (DONE)

**Goal**: Make liquidation robust and offline-friendly for existing spot trades without changing schema.

**Implementation (completed)**:

- [x] Replace the single yfinance call with a **multi-step MTM hierarchy** in `scripts/liquidate_open_trades.py`:
  - Step 1: Try latest close from local `ohlcv_data` per ticker.
  - Step 2: Fall back to yfinance (if available).
  - Step 3: Fall back to entry price when neither source is available.
- [x] Introduce a `TradeRow` dataclass capturing:
  - `id`, `ticker`, `action`, `shares`, `price`, `commission`
  - Optional hints: `asset_class`, `instrument_type`, `underlying_ticker`, `strike`, `expiry`, `multiplier` (used only when present in schema).
- [x] Add a pluggable **pricing policy** via CLI:
  - `--pricing-policy` with choices `neutral`, `conservative`, `intrinsic`, `bs_model`.
  - Spot (equity/ETF/crypto):  
    - `neutral`: MTM = latest spot price.  
    - `conservative`: clamp MTM toward entry (no unrealised gains) depending on long/short side.
  - Options (call/put):  
    - `intrinsic` / `bs_model`: currently intrinsic-only (BS reserved for later), using underlying spot and strike.
  - Synthetic/unknown: MTM = entry price (no fantasy PnL).
- [x] Keep **schema compatibility**:
  - Script discovers optional columns via `PRAGMA table_info(trade_executions)` and only selects them when present.
  - Defaults to `asset_class="equity"`, `instrument_type="spot"` when hints are absent.

**Resulting behaviour**:

- Existing spot-only DBs continue to work, but MTM is more resilient (DB + vendor + entry fallback).
- The script now exposes a CLI knob for conservative MTM without touching code.

---

## Phase 2 – Asset-Class Hints & Options Support

**Goal**: Allow the liquidation script to handle basic options and crypto more realistically using minimal schema extensions.

**Planned tasks**:

- [x] Extend DB schema (non-breaking) for richer trade metadata:

  ```sql
  ALTER TABLE trade_executions
    ADD COLUMN asset_class TEXT DEFAULT 'equity',
    ADD COLUMN instrument_type TEXT DEFAULT 'spot',
    ADD COLUMN underlying_ticker TEXT,
    ADD COLUMN strike REAL,
    ADD COLUMN expiry TEXT,
    ADD COLUMN multiplier REAL DEFAULT 1.0;
  ```

- [x] Keep this change in `etl/database_manager.py`’s schema management, using `ALTER TABLE` guarded by `PRAGMA table_info` so it is idempotent (`trade_executions` now exposes these columns with defensive migrations).

- [x] Ensure order/trading paths populate these fields where applicable:
  - Current status:
    - Paper trades: `execution/paper_trading_engine.PaperTradingEngine._store_trade_execution` populates `asset_class` heuristically (equity vs crypto) and keeps `instrument_type="spot"`; the Trade dataclass carries optional option/synthetic metadata for future use.
    - Live orders: `execution.order_manager.OrderManager._record_execution` reads optional `asset_class` / `instrument_type` / `underlying_ticker` / `strike` / `expiry` / `multiplier` from `OrderRequest.metadata` and persists them, falling back to ticker heuristics and `"spot"` when absent.
  - Next: once options execution paths are wired under `options_trading.enabled`, they should set these metadata fields explicitly for options/synthetic instruments.

- [ ] Tighten option MTM logic in `liquidate_open_trades.py`:
  - For intrinsic pricing:
    - Calls: `max(S − K, 0)`  
    - Puts: `max(K − S, 0)`  
    - Decide whether entry price and MTM are *per contract* or *per underlying unit*, and document this explicitly.
  - Ensure PnL formula is consistent with how options are stored (contracts × premium vs underlying units).

**Success criteria**:

- [ ] Options trades with populated metadata can be liquidated via intrinsic-only MTM.
- [ ] Crypto trades follow the spot hierarchy (DB → vendor → entry) correctly.
- [ ] No regressions for legacy spot-only DBs.

---

## Phase 3 – Black–Scholes & Realised Volatility (Options)

**Goal**: Add a neutral-ish valuation mode for options using a simple Black–Scholes model with realised volatility as a proxy for implied vol.

**Planned tasks**:

- [ ] Extend `liquidate_open_trades.py` with a true `bs_model` branch:
  - Use underlying close series from `ohlcv_data` to compute realised volatility (e.g. 30–60 day log-return std).
  - Compute time-to-expiry from `expiry` minus valuation date.
  - Use a flat risk-free rate from config (`config/analysis_config.yml` or `config/quant_success_config.yml`).
  - Implement Black–Scholes call/put pricing (no greeks needed for this script).

- [ ] Make the `bs_model` policy:
  - The default for options when all required inputs are available.
  - Fall back to intrinsic-only when vol or expiry is missing.

- [ ] Document limitations:
  - No smile/surface model.
  - Using realised volatility as a proxy for implied volatility.
  - Intended for research/diagnostic MTM, not production option desk valuation.

**Success criteria**:

- [ ] Options MTM under `--pricing-policy bs_model` yields reasonable values vs intrinsic-only for ITM/ATM contracts.
- [ ] Fallbacks are explicit and logged when inputs are missing.

---

## Phase 4 – Synthetic Instruments & Barbell Legs

**Goal**: Handle synthetic trades (multi-leg barbell structures, spreads, etc.) via leg decomposition and per-leg MTM, instead of treating them as opaque.

**Planned tasks**:

- [x] Design and create a `synthetic_legs` table:

  Implemented in `etl/database_manager.DatabaseManager._initialize_schema` as:

  - `synthetic_legs` table with the same columns as the design snippet (id, synthetic_trade_id, leg_type, ticker, underlying_ticker, direction, quantity, strike, expiry, multiplier).
  - Index `idx_synthetic_legs_trade` on `(synthetic_trade_id)` for fast lookup.

- [x] Add helper(s) in a small module (e.g. `etl/synthetic_pricer.py`) to:
  - Load legs for a given synthetic trade (`etl/synthetic_pricer.load_synthetic_legs`).
  - Invoke neutral MTM rules per leg (spot/cash via spot, options via intrinsic with multiplier).
  - Sum leg-level MTM to obtain the synthetic MTM (`etl/synthetic_pricer.compute_synthetic_mtm`).

- [x] Wire synthetic MTM into `liquidate_open_trades.py`:
  - When `instrument_type='synthetic'` and legs exist, `scripts/liquidate_open_trades._mark_to_market`:
    - Rebuilds a spot map for the leg underlyings using the same DB→vendor→fallback hierarchy as spot MTM.
    - Calls `compute_synthetic_mtm` to obtain a neutral synthetic MTM value.
  - When legs are missing or any error occurs, the script falls back to entry price (explicitly non‑fantasy) and keeps the diagnostic-only contract.

**Success criteria**:

- [ ] Synthetic trades with leg definitions are liquidated as the sum of leg-level MTM.
- [ ] No synthetic trade is accidentally “fantasy‑marked” without a clear log message.

---

## Phase 5 – Integration with Risk & Reporting

**Goal**: Make MTM and liquidation behaviour consistent with risk monitoring, barbell policy, and reporting, while keeping `liquidate_open_trades.py` clearly diagnostic.

**Planned tasks**:

- [ ] Cross-link MTM policies with barbell/risk docs:
  - Ensure `NAV_RISK_BUDGET_ARCH.md` and `BARBELL_INTEGRATION_TODO.md` describe how liquidation scripts treat safe vs risk buckets.
  - Consider a barbell-aware pricing policy (e.g. intrinsic-only for risk bucket options by default).

- [ ] Add a short section to `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md` summarising:
  - How diagnostic liquidation is used in experiments.
  - How MTM policy choices (neutral vs conservative vs intrinsic vs BS) affect PnL distributions and risk metrics.

- [ ] Keep `liquidate_open_trades.py` clearly marked as **non-production**:
  - Consider adding a dedicated environment or CLI guard (e.g. `--confirm-diagnostic`) to avoid accidental use in production pipelines.

**Success criteria**:

- [ ] MTM policy is documented and consistent across risk docs and code.
- [ ] Research experiments explicitly state which pricing policy was used.
- [ ] No production process depends on this script for authoritative PnL.

---

## Quick Reference

- **Current script**: `scripts/liquidate_open_trades.py`  
  - CLI: `--db-path`, `--pricing-policy {neutral, conservative, intrinsic, bs_model}`  
  - Behaviour: robust spot MTM; basic hooks for options and future expansion.

- **Next steps**:
  - Implement Phase 2 schema hints and option metadata.
  - Add Black–Scholes pricing (Phase 3).
  - Introduce synthetic leg tables and decomposition (Phase 4).

Keep this document updated as phases complete so it remains a faithful, high-level guide to the MTM and liquidation behaviour of the system.
