# Production Deployment Roadmap — Phase 7.9

**Date**: 2026-02-03
**Basis**: Verified against live codebase; reconciled with [PNL_ROOT_CAUSE_AUDIT.md](PNL_ROOT_CAUSE_AUDIT.md) and [REVIEW_VERIFICATION_VERDICT.md](REVIEW_VERIFICATION_VERDICT.md)
**Go/No-Go Gates**: 5 phases, each with measurable pass criteria

---

## What Was Dropped From the Proposed Roadmap — and Why

The user-supplied roadmap contained several phases built on premises the verification proved incorrect. These are excluded to avoid introducing regressions:

| Dropped Section | Reason | Verdict Ref |
|-----------------|--------|-------------|
| Phase 0.1 "Create holding\_bars migration" | `holding_bars` is a runtime bar-count in `_evaluate_exit_reason`, not a DB column. Attribution migration (8 columns) is already committed. Adding a `holding_bars` column would be dead code. | C3: FALSE |
| Phase 0.2 "Emergency relaxed thresholds" (confidence→0.45, min\_expected\_return→8 bps) | Lowers `min_expected_return` from 5 bps to 8 bps. The audit proves it must go UP to 30 bps to cover 15 bps roundtrip friction. The proposed change would deepen the -0.14% per-trade loss. | C4, M1 |
| Phase 0.3 "Fix ensemble policy gate" (line ~847, add PRODUCTION status) | (a) Actual policy logic is at lines 1250-1312, not 847. (b) The gate already shows `Decision: KEEP`. (c) "PRODUCTION" is not a valid status — the system uses KEEP / RESEARCH\_ONLY / DISABLE\_DEFAULT. Applying this patch would break policy semantics. | C2: OUTDATED |
| Phase 1 "DQN Agent" (entire section: network, agent, replay buffer) | DQN is explicitly deferred to Phase 12+ per `Documentation/AGENT_INSTRUCTION.md:215`. Prerequisites include $100k+ live capital. The section also references `agents/` directory, `main.py`, `agents.forecasting.samossa_core.SAMoSSAEngine` — none of which exist. RL is already handled by MSSA-RL (tabular Q-learning). | C1: FALSE |
| Phase 1.2 "SAMoSSA-DQN Coordinator" | Same as Phase 1. References non-existent modules. | C1: FALSE |
| Phase 1.3 "Integration with main.py" | `main.py` does not exist. Entry points are `scripts/run_auto_trader.py` and `scripts/run_etl_pipeline.py`. | Structural |
| All DQN test files | Test non-existent modules. | Structural |

---

## Corrected Roadmap

The roadmap below uses only existing infrastructure. No new modules or directories are required. Each phase maps to specific file + line changes, uses the existing test suite, and gates on metrics observable in current log output.

---

## PHASE 0: P0 Cost Math Corrections (Week 1)

**Objective**: Close the 5x cost-assumption gap that causes systematic -0.14% per-trade loss.
**Gate**: Re-run 2-audit sprint; average PnL per trade must be >= -0.05% (i.e., no longer systematically negative).

### 0.1 Raise min\_expected\_return

**File**: [config/signal_routing_config.yml](../config/signal_routing_config.yml)
**Current** (line 21): `min_expected_return: 0.0005` (5 bps)
**Target**: `min_expected_return: 0.0030` (30 bps = 2x roundtrip cost buffer)

**Rationale**: With ~15 bps actual roundtrip friction, 5 bps threshold admits trades guaranteed to lose money. 30 bps provides 2x safety margin and matches the signal generator's own default ([models/time_series_signal_generator.py](../models/time_series_signal_generator.py) line 148: `min_expected_return: float = 0.003`).

**Expected impact**: ~75% reduction in trade count; surviving trades have positive expected edge.

### 0.2 Enable transaction cost in execution

**File**: [execution/paper_trading_engine.py](../execution/paper_trading_engine.py)
**Current** (line 153): `transaction_cost_pct: float = 0.0`
**Target**: `transaction_cost_pct: float = 0.00015` (1.5 bps, matching US\_EQUITY prior in signal\_routing\_config.yml)

**Rationale**: PnL calculation currently ignores transaction costs entirely. Signal routing assumes 1.5 bps; execution must match. See [config/signal_routing_config.yml](../config/signal_routing_config.yml) lines 29-36 for the cost model.

### 0.3 Align edge-gate roundtrip cost estimate

**File**: [execution/paper_trading_engine.py](../execution/paper_trading_engine.py) line 454

**Current**:
```python
est_roundtrip_cost = float(spread_pct) + 2.0 * float(self.slippage_pct or 0.0) + 2.0 * float(self.transaction_cost_pct or 0.0)
```

**Target**: Include a market-impact floor so the estimate is not dominated by spread alone when spread data is missing:
```python
est_roundtrip_cost = (
    float(spread_pct)
    + 2.0 * float(self.slippage_pct or 0.0)
    + 2.0 * float(self.transaction_cost_pct or 0.0)
    + 2.0 * float(self.impact_pct or 0.00005)  # 0.5 bps each way market impact floor
)
```

**Rationale**: Actual roundtrip impact is ~10 bps (5 bps each way) per audit evidence. Without an impact floor, the gate underestimates costs when bid/ask data is missing.

### 0.4 Phase 0 Gate Validation

```bash
# Activate venv
simpleTrader_env\Scripts\activate

# Run 2-audit sprint with P0 fixes applied
PROOF_MODE=1 ENABLE_DATA_CACHE=0 bash/run_20_audit_sprint.sh --audits 2 --tickers AAPL

# Pass criteria (from gate_1 log):
#   Average PnL per trade >= -0.05%  (was -0.14%)
#   At least 1 trade executed (signal quality filter may reduce count)
```

**Gate decision**: If average PnL is still < -0.05%, investigate whether slippage estimate needs adjustment before proceeding.

---

## PHASE 1: Proof-Mode Validation Sprint (Week 2)

**Objective**: Accumulate 20 audits with P0 fixes active; validate holding\_bars fix eliminates premature TIME\_EXIT.
**Gate**: 20 effective audits complete; violation rate <= 25%; zero premature TIME\_EXIT on positions held < 3 calendar days.

### 1.1 Run 20-audit sprint

```bash
# Full 20-audit sprint with proof-mode enabled
export PROOF_MODE=1
export ENABLE_DATA_CACHE=0
export PMX_LONG_ONLY=1
export PMX_EDGE_COST_GATE=1
export PMX_EDGE_COST_MULTIPLIER=1.25

bash/run_20_audit_sprint.sh --tickers AAPL
```

### 1.2 Validate holding\_bars fix

After sprint completes, check intraday pass logs for TIME\_EXIT:

```bash
# Search for TIME_EXIT in all intraday passes
grep "TIME_EXIT" logs/audit_sprint/*/gate_2_intraday*.log

# Expected: If TIME_EXIT appears, verify holding_period_days >= 3
# A TIME_EXIT at holding_period_days=0 or 1 indicates the frequency-mismatch
# fix in commit 668ee26 did not take effect for that run.
```

### 1.3 Validate close-trade attribution

```bash
# Query the audit DB for attribution fields
sqlite3 logs/audit_sprint/<run_id>/portfolio_maximizer_<run_id>_audit.db \
  "SELECT ticker, action, entry_price, exit_price, is_close, exit_reason FROM trade_executions WHERE is_close = 1 LIMIT 20;"

# All closed trades should have non-NULL entry_price, exit_price, exit_reason
```

### 1.4 Phase 1 Gate

| Metric | Pass Threshold |
|--------|----------------|
| Effective audits completed | 20 |
| Forecast gate violation rate | <= 25% |
| Premature TIME\_EXIT (< 3 calendar days) | 0 |
| Close-trade attribution populated | >= 90% of closed trades |

---

## PHASE 2: P1 Tuning (Week 3-4)

**Objective**: Tighten signal quality and position sizing to improve win rate.
**Gate**: Win rate >= 45% on closed trades (up from ~0% under P0 cost assumptions).

### 2.1 Increase directional accuracy requirement

**File**: [config/quant_success_config.yml](../config/quant_success_config.yml)
**Current** (line 38): `min_directional_accuracy: 0.42`
**Target**: `min_directional_accuracy: 0.50`

**Rationale**: 42% accuracy is below coin-flip for a binary direction call. Raising to 50% ensures only forecasts with demonstrated edge pass the quant gate.

### 2.2 Increase position sizing multipliers

**File**: [execution/paper_trading_engine.py](../execution/paper_trading_engine.py) lines 768-774

| Regime | Current Multiplier | Target Multiplier | Rationale |
|--------|-------------------|-------------------|-----------|
| Exploration | 0.25 | 0.50 | Spreads fixed costs over 2x notional |
| Red regime | 0.30 | 0.60 | Same logic; avoids sub-$250 positions |
| Green regime | 1.20 | 1.20 | No change needed |

### 2.3 Increase edge weight in confidence scoring

**File**: [models/time_series_signal_generator.py](../models/time_series_signal_generator.py) line 973

Current edge weight in confidence composite: 0.25
Target: 0.40

**Rationale**: Edge (expected return vs cost) should be the dominant confidence factor. Agreement and diagnostic signals are secondary.

### 2.4 Phase 2 Gate

```bash
# Run 5-audit sprint with P1 tuning active
PROOF_MODE=1 ENABLE_DATA_CACHE=0 bash/run_20_audit_sprint.sh --audits 5 --tickers AAPL MSFT
```

| Metric | Pass Threshold |
|--------|----------------|
| Win rate on closed trades | >= 45% |
| Profit factor | >= 0.80 |
| Avg PnL per trade | >= 0.0% (break-even or better) |

---

## PHASE 3: Feature Activation Gates (Week 5-6)

**Objective**: Validate opt-in features (barbell, frontier markets) in isolated sprints before enabling broadly.
**Gate**: Each feature passes its own 3-audit validation with no regressions vs Phase 2 baseline.

### 3.1 Barbell Activation

Infrastructure is complete ([risk/barbell_policy.py](../risk/barbell_policy.py), 177 lines). Currently disabled via [config/barbell.yml](../config/barbell.yml) line 5: `enable_barbell_allocation: false`.

**Activation sequence**:
1. Set `enable_barbell_allocation: true` in [config/barbell.yml](../config/barbell.yml)
2. Run 3-audit sprint with barbell enabled
3. Verify `barbell_bucket` and `barbell_multiplier` columns populate in trade\_executions:
```bash
sqlite3 <db_path> "SELECT ticker, barbell_bucket, barbell_multiplier FROM trade_executions WHERE barbell_bucket IS NOT NULL LIMIT 10;"
```
4. Compare PnL metrics vs Phase 2 baseline. Barbell should not worsen win rate by > 5 percentage points.

**Current bucket structure** (from [config/barbell.yml](../config/barbell.yml)):
- Safe (75-95%): SHY, BIL, IEF (Treasury ballast)
- Core (max 20%): MSFT, CL=F, MTN
- Speculative (max 10%): AAPL, BTC-USD

### 3.2 Frontier Market Activation

Infrastructure is complete ([etl/frontier_markets.py](../etl/frontier_markets.py), imported by [etl/data_universe.py](../etl/data_universe.py)). Currently disabled: `include_frontier_tickers` defaults to `False`.

**Activation sequence**:
1. Run a single ETL extraction with `--include-frontier-tickers` flag to verify data availability:
```bash
python scripts/run_etl_pipeline.py --tickers AAPL --include-frontier-tickers --cycles 1 --execution-mode synthetic
```
2. Verify NGX/NSE/JSE tickers appear in extracted data (check logs for `merge_frontier_tickers` output)
3. If extraction succeeds, run 3-audit sprint with frontier tickers included alongside US equities
4. Note: NGX cap is 5% ([config/barbell.yml](../config/barbell.yml) line 40: `ngx_equity_max: 0.05`) — enforce this even if barbell is not yet enabled

### 3.3 Phase 3 Gate

| Feature | Pass Criteria |
|---------|--------------|
| Barbell | Attribution columns populate; PnL regression < 5pp vs Phase 2 |
| Frontier | Data extraction succeeds for >= 2 frontier tickers; no pipeline errors |

---

## PHASE 4: Extended Audit Accumulation & Regime Evaluation (Week 7-9)

**Objective**: Run 100+ audits to stabilize ensemble gate; evaluate whether regime detection improves or hurts PnL.
**Gate**: 100 effective audits accumulated; regime detection either validated or disabled based on PnL evidence.

### 4.1 Extended audit sprint

```bash
# Run continuous 100-audit sprint (may take multiple sessions)
PROOF_MODE=1 ENABLE_DATA_CACHE=0 bash/run_20_audit_sprint.sh --audits 100 --tickers AAPL MSFT NVDA
```

Gate check after every 20-audit block:
```bash
python scripts/check_forecast_audits.py
# Monitor: violation rate, Decision status (KEEP vs RESEARCH_ONLY)
```

### 4.2 Regime detection A/B evaluation

Run two parallel 20-audit sprints — one with regime detection enabled, one disabled:

```bash
# Arm A: regime enabled (current default)
REGIME_ENABLED=1 bash/run_20_audit_sprint.sh --audits 20 --tickers AAPL

# Arm B: regime disabled
# Temporarily set regime_detection.enabled: false in config/forecasting_config.yml
# (revert after sprint)
REGIME_ENABLED=0 bash/run_20_audit_sprint.sh --audits 20 --tickers AAPL
```

**Decision rule**: If Arm B (regime off) PnL is better by > 10% relative, disable regime detection in production config until 100+ audits are available for weight re-optimization. Current evidence ([Documentation/PHASE_7.5_VALIDATION.md](PHASE_7.5_VALIDATION.md)) shows +42% RMSE regression with regime detection on.

### 4.3 Phase 4 Gate

| Metric | Pass Threshold |
|--------|----------------|
| Effective audits accumulated | >= 100 |
| Forecast gate status | KEEP (not RESEARCH\_ONLY) |
| Regime A/B verdict | Clear winner identified; config updated accordingly |
| Win rate (rolling 20-trade window) | >= 50% |

---

## PHASE 5: Production Readiness Checklist (Week 10+)

**Objective**: Final checks before declaring production-ready.

### 5.1 Checklist

- [ ] P0 cost fixes applied and validated (Phase 0)
- [ ] 100+ audits accumulated with KEEP status (Phase 4)
- [ ] Close-trade attribution fully populated (Phase 1)
- [ ] Barbell activated and validated OR explicitly deferred with rationale (Phase 3)
- [ ] Frontier markets activated and validated OR explicitly deferred (Phase 3)
- [ ] Regime detection decision made and config updated (Phase 4)
- [ ] `min_directional_accuracy >= 0.50` (Phase 2)
- [ ] Win rate >= 50% on last 50 closed trades
- [ ] Profit factor >= 1.0 on last 50 closed trades
- [ ] All tests pass: `pytest tests/ -m "not slow"` (720+ tests)
- [ ] UTC timestamps verified: all `trade_date` and `entry_timestamp` fields contain `+00:00`
- [ ] Cron job path fixed (currently `/mnt/c/.../bin/python` — needs Windows path)

### 5.2 DQN / Deep RL — Future Consideration

DQN portfolio optimization is documented as a Phase 12+ objective. Prerequisites per [Documentation/AGENT_INSTRUCTION.md](AGENT_INSTRUCTION.md):
- $100k+ live capital deployed
- Existing ML models (GARCH, SAMoSSA, MSSA-RL) demonstrating proven value in live trading
- Multi-month training timeline accepted
- PyTorch dependency added to requirements.txt

The current RL component (MSSA-RL, tabular Q-learning in [forcester_ts/mssa_rl.py](../forcester_ts/mssa_rl.py)) is lightweight by design and does not require PyTorch or gymnasium. Do not introduce deep RL dependencies until Phase 12+ prerequisites are met.

---

## Summary: Phases and Expected Outcome

| Phase | Focus | Key Changes | Expected Outcome |
|-------|-------|-------------|------------------|
| 0 | Cost math | 3 config/code edits | Eliminate systematic -0.14% loss |
| 1 | Validation sprint | 20-audit sprint | Confirm fixes, accumulate audits |
| 2 | Signal quality | Accuracy + sizing tuning | Win rate >= 45% |
| 3 | Feature gates | Barbell + frontier opt-in | Diversification path validated |
| 4 | Audit accumulation | 100+ audits, regime A/B | Stable gate; regime decision |
| 5 | Production checklist | Final validation | Go/no-go for live |

---

*Reconciled against codebase on 2026-02-03. Commits in scope: up through cda18ac.*
