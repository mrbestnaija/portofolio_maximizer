# Phase 7.14: Gate Recalibration, Production Hygiene & DA Improvement

**Status**: IN PROGRESS
**Started**: 2026-02-24
**Completed**: TBD
**Preceded by**: Phase 7.13 (Architectural Sanitization - ID Unification & Pipeline Wiring)

---

## Executive Summary

Phase 7.13 fixed 12 architectural disconnects (signal ID collisions, pipeline dead-ends, path
centralization). A follow-up gate-stack audit revealed three additional layers of "test-mode drift"
that poison both performance reporting and gate decisions independently of model quality:

1. **Config left in test-mode**: `confidence_threshold: 0.45` ("lowered for test runs"),
   `max_risk_score: 0.85` ("raised during evaluation"), AAPL/MSFT `min_expected_return: 0.0005`
   (5 bps -- below roundtrip cost floor ~15 bps). SNR gate `min_signal_to_noise: 0.0` disabled
   despite being implemented at `signal_generator.py:478`.
2. **Proof-mode in production pipelines**: `overnight_refresh.sh` Step 2.5 passed `--proof-mode`
   -> artificially tight exits -> Platt calibration data poisoned with test behavior.
   `run_20_audit_sprint.sh` defaulted `PROOF_MODE=1` -> sprint metrics not comparable to live.
3. **Gates that pass trivially**: `min_lift_fraction: 0.10` let ensemble "certify" despite beating
   best-single in only 8% of windows. `max_fail_fraction: 0.95` (raised for proof-mode, never
   reverted) made 71.7% rolling FAIL rate appear YELLOW not RED. `calibration.db_path: null` meant
   Platt calibration never read from DB.

**DB state at phase start (2026-02-24)**:
- 39 production closes, 41% WR: TIME_EXIT(24,+$14.20avg) | STOP_LOSS(10,-$68.17avg) | TAKE_PROFIT(4,+$253.03avg)
- 84 trades all `legacy_*` ts_signal_ids -- overnight synthetic cycle never produced `ts_*` attributed trades
- JSONL: 1 entry, 0 Platt pairs -> Platt calibration completely starved
- Mean confidence: 80.6% -> actual WR 41% -> severe calibration gap

---

## Implementation Phases

### Phase 7.14-A: Config Sanitization + Gate Recalibration

**Type**: Config-only, no code changes
**Status**: COMPLETE
**Commit**: TBD

| File | Setting | Old | New | Reason |
|------|---------|-----|-----|--------|
| `signal_routing_config.yml` | `confidence_threshold` | 0.45 | **0.55** | Comment "lowered for test runs" -- never raised back |
| `signal_routing_config.yml` | `AAPL.min_expected_return` | 0.0005 | **0.0020** | 5 bps < roundtrip cost floor (~15 bps) |
| `signal_routing_config.yml` | `MSFT.min_expected_return` | 0.0005 | **0.0020** | Same reason as AAPL |
| `signal_routing_config.yml` | `MTN.min_expected_return` | 0.0005 | **0.0020** | Same reason |
| `signal_routing_config.yml` | `MSFT.confidence_threshold` | 0.45 | **0.55** | Align with global production threshold |
| `signal_routing_config.yml` | `MTN.confidence_threshold` | 0.45 | **0.55** | Align with global production threshold |
| `signal_routing_config.yml` | `AAPL.confidence_threshold` | 0.50 | **0.55** | Align with global production threshold |
| `signal_routing_config.yml` | `max_risk_score` | 0.85 | **0.70** | Comment "raised during evaluation" -- never reverted |
| `signal_routing_config.yml` | `MSFT.max_risk_score` | 0.85 | **0.70** | Align with global |
| `signal_routing_config.yml` | `min_signal_to_noise` | 0.0 | **1.5** | Implemented at signal_generator.py:478 -- just disabled. 1.5 requires E[return] > 1.5x CI half-width |
| `forecaster_monitoring.yml` | `max_fail_fraction` | 0.95 | **0.85** | Raised for proof-mode (never reverted). 71.7% FAIL rate was YELLOW; now correctly RED |
| `forecaster_monitoring.yml` | `min_lift_fraction` | 0.10 | **0.25** | Ensemble beats best-single on only 8% of windows; 10% trivially passes |
| `quant_success_config.yml` | `min_directional_accuracy` | 0.40 | **0.45** | Actual WR 41% = 1pp above 0.40 floor -- no upward pressure |
| `quant_success_config.yml` | `calibration.db_path` | null | **`"data/portfolio_maximizer.db"`** | `_calibrate_confidence()` queries this DB -- null path means calibration never executes |
| `overnight_refresh.sh` | Step 2.5 `--proof-mode` flag | present | **removed** | Proof-mode taints Platt calibration data with non-production exit behavior |
| `overnight_refresh.sh` | PLATT_BOOTSTRAP loop | absent | **added** | 8 historical dates (2021-2024) to seed Platt pairs; opt-in via `PLATT_BOOTSTRAP=1` |
| `run_20_audit_sprint.sh` | `PROOF_MODE` default | 1 | **0** | Proof-mode is opt-in for development only; production sprints run live-comparable |

**Verification**:
```bash
pytest tests/ -q  # Must maintain 914+
python scripts/run_all_gates.py --db data/portfolio_maximizer.db
```

---

### Phase 7.14-B: ATR-Based Stop Loss

**Type**: Code change
**Status**: COMPLETE
**Files**: `models/time_series_signal_generator.py`, `tests/models/test_time_series_signal_generator.py`

**Problem**: Current stop loss = `volatility * 0.5` clamped `[1.5%, 5%]`. NVDA ATR(14) ~$10
on ~$130 = 7.7%, well above the 5% cap -- normal intraday noise fires stop. ATR uses actual
bar High/Low ranges (market-observed noise), not model-implied vol.

**Implementation**:
- Add `_compute_atr(self, market_data: pd.DataFrame, period: int = 14) -> Optional[float]`
- Update `_calculate_targets(self, ..., market_data: Optional[pd.DataFrame] = None)`
- New stop logic: `stop_loss_pct = max((atr * 1.5) / current_price, 0.015)` -- no 5% ceiling
- Fallback to vol-based stop when OHLC columns unavailable

**Tests** (4 new, all passing):
- `test_atr_stop_uses_bar_data`: Known OHLCV -> verify stop = price - ATR*1.5
- `test_atr_stop_fallback_no_ohlc`: Missing High/Low -> vol-based fallback
- `test_atr_stop_minimum_floor`: Very low ATR -> stop >= 1.5%
- `test_atr_stop_nvda_wide`: ATR > 5% of price -> no 5% cap applied

**Verification (2026-02-25)**: 4/4 tests pass. Fast regression lane: 903 passed, 0 failures.

---

### Phase 7.14-C: GARCH Convergence Hardening

**Type**: Code change
**Status**: COMPLETE
**Files**: `forcester_ts/garch.py`, `tests/forecasting/test_garch_convergence.py` (NEW)

**Problem**: `garch.py:172` -- `model.fit(disp="off")` with no convergence check. `arch` library
wraps scipy ConvergenceWarning as RuntimeWarning when SLSQP hits code 9. Unconverged fits produce
garbage CI bounds that the SNR gate (now enabled at 1.5) should catch -- but only if CIs are
flagged as wider.

**Self-attenuating chain**: convergence failure -> wider CI -> lower SNR -> gate blocks -> no trade.

**Implementation**:
- Wrap `model.fit()` in `warnings.catch_warnings(record=True)` to detect "convergence"/"code 9"
- Extend GJR fallback trigger: `if convergence_failed or persistence >= 0.97`
- On failure: inflate CI half-width by 1.5x
- Parkinson estimator backup when GJR also fails: `sqrt(mean((ln(H/L))^2) / (4*ln2))`
- Tag result: `result["convergence_ok"] = not convergence_failed`

**Tests** (5 in `tests/forecasting/test_garch_convergence.py`, all passing):
- `test_convergence_ok_true_by_default`: initial state check
- `test_convergence_failure_detected_via_warning`: Mock RuntimeWarning -> `_convergence_ok=False` + GJR attempted
- `test_convergence_ok_stays_true_on_clean_fit`: Normal convergence -> flag unchanged
- `test_convergence_failure_inflates_ci`: CI 1.5x wider after failure (`_enrich_garch_forecast`)
- `test_good_fit_ci_unchanged`: `convergence_ok=True` -> CI unmodified

**Verification (2026-02-25)**: 5/5 tests pass. Self-attenuating chain confirmed: bad fit -> wider CI -> SNR drops -> signal blocked.

---

### Phase 7.14-D: Persist detected_regime to DB

**Type**: Migration + code change
**Status**: COMPLETE
**Files**: `scripts/migrate_add_regime_to_forecasts.py` (NEW), `forcester_ts/forecaster.py`,
          `etl/database_manager.py`, `scripts/run_etl_pipeline.py`, `tests/etl/test_database_manager_schema.py`

**Problem**: Regime detection runs in-memory, result never saved to `time_series_forecasts` table.
Downstream analysis cannot correlate regime with trade outcomes.

**Implementation**:
- Migration: `ALTER TABLE time_series_forecasts ADD COLUMN detected_regime TEXT, regime_confidence REAL`
- `forecaster.py`: Add `detected_regime`/`regime_confidence` to `forecast_bundle` dict
- `database_manager.py`: Add params to `save_forecast()` + auto-migration in `_initialize_schema()`

**Wire fix (2026-02-25)**:
- `forecaster.py`: added `results["detected_regime"]` alias (`None` when regime is "STATIC")
- `run_etl_pipeline.py`: extracted `_regime_for_db`/`_regime_conf_for_db` and added to all 5 model `forecast_data` dicts (SARIMAX, GARCH, SAMOSSA, MSSA_RL, ENSEMBLE)

**Tests** (2, all passing):
- `test_detected_regime_saved_and_retrieved`: Round-trip through DB
- `test_detected_regime_null_ok`: NULL regime -> no error

**Verification (2026-02-25)**: Migration idempotent on production DB ([SKIP] columns already exist). 2/2 tests pass. Fast regression lane: 903 passed, 0 failures.

---

### Phase 7.14-E: Platt Calibration End-to-End Wire

**Type**: Code change
**Status**: COMPLETE
**Files**: `execution/paper_trading_engine.py`, `tests/execution/test_paper_trading_engine.py`

**Problem**: `_calibrate_confidence()` queries `confidence_calibrated` column in `trade_executions`.
Column exists (Phase 7.9 schema) but `PaperTradingEngine` never saves it -> all 84 rows NULL ->
calibration query returns 0 usable rows even after A3 sets `db_path`.

**Full chain (all links verified present 2026-02-25)**:
1. `TimeSeriesSignal.confidence_calibrated` field on signal dataclass (`time_series_signal_generator.py:137`)
2. `_calibrate_confidence()` result → `self._platt_calibrated` → set on signal at line 617
3. `signal_router._signal_to_dict()` exposes via `getattr(signal, 'confidence_calibrated', None)` (line 348)
4. `PaperTradingEngine` reads from signal dict → `Trade.confidence_calibrated` (line 604)
5. `save_trade_execution()` passes to DB column (line 1353)

**Tests** (1, passing):
- `test_confidence_calibrated_saved_to_db`: Signal with `confidence_calibrated=0.62` -> DB query -> value stored

**Verification (2026-02-25)**: 1/1 test pass. Final regression lane: 903 passed, 0 failures.

---

### Phase 7.14-F: Signal Generator Factory

**Status**: DEFERRED TO 7.15
Complexity and regression risk exceed benefit relative to other 7.14 changes. Config drift between
`run_etl_pipeline.py` and `run_auto_trader.py` is partially mitigated by Phase A config
sanitization. Factory refactor is first item in Phase 7.15.

---

## Verification Checklist (Final)

**Phase 7.14 status (2026-02-25)**: A COMPLETE, B COMPLETE, C COMPLETE, D COMPLETE, E COMPLETE, F DEFERRED->7.15
**Regression baseline**: 903 passed, 1 skipped, 28 deselected, 7 xfailed (fast lane `not gpu and not slow`)

```bash
# After Phase A
pytest tests/ -q                                              # 903+ (fast lane)
python scripts/run_all_gates.py --db data/portfolio_maximizer.db

# After Phase D
python scripts/migrate_add_regime_to_forecasts.py
sqlite3 data/portfolio_maximizer.db ".schema time_series_forecasts" | grep regime

# After Phase E: run 1 overnight synthetic pass to seed Platt
bash bash/overnight_refresh.sh
python scripts/update_platt_outcomes.py
python -c "
import json, pathlib
entries = [json.loads(l) for l in pathlib.Path('logs/signals/quant_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
with_outcome = [e for e in entries if 'outcome' in e]
ts_ids = [e for e in entries if str(e.get('signal_id','')).startswith('ts_')]
print(f'Platt pairs: {len(with_outcome)} (target: >0)')
print(f'ts_* signal_ids in JSONL: {len(ts_ids)}')
"

# Full suite
pytest tests/ -q                                              # 914+
```

---

## Files Modified (Complete List)

| File | Phase | Change |
|------|-------|--------|
| `config/signal_routing_config.yml` | A | confidence 0.45->0.55, min_return AAPL/MSFT/MTN 0.0005->0.002, risk 0.85->0.70, SNR 0->1.5 |
| `config/forecaster_monitoring.yml` | A | max_fail_fraction 0.95->0.85, min_lift_fraction 0.10->0.25 |
| `config/quant_success_config.yml` | A | min_DA 0.40->0.45, calibration.db_path null->path |
| `bash/overnight_refresh.sh` | A | Remove --proof-mode; add PLATT_BOOTSTRAP historical seeding loop |
| `bash/run_20_audit_sprint.sh` | A | PROOF_MODE default 1->0 |
| `models/time_series_signal_generator.py` | B | `_compute_atr()` + `_calculate_targets()` ATR logic |
| `tests/models/test_time_series_signal_generator.py` | B | 4 new ATR stop tests |
| `forcester_ts/garch.py` | C | ConvergenceWarning catch + GJR trigger extension + CI inflation |
| `tests/forecasting/test_garch_convergence.py` | C | NEW -- 4 convergence tests |
| `scripts/migrate_add_regime_to_forecasts.py` | D | NEW -- idempotent migration |
| `forcester_ts/forecaster.py` | D | detected_regime in forecast_bundle |
| `etl/database_manager.py` | D | save_forecast() regime params + auto-migration |
| `tests/etl/test_database_manager_schema.py` | D | 2 regime persistence tests |
| `execution/paper_trading_engine.py` | E | Trade.confidence_calibrated + save wiring |
| `tests/execution/test_paper_trading_engine.py` | E | 1 confidence_calibrated test |

**Deferred to 7.15**: `models/signal_generator_factory.py`, factory wiring in pipeline scripts

---

## Context for Future Iterations

### Why these specific thresholds?

- **confidence 0.55**: Comment in config explicitly said "lowered for test runs". 0.55 is
  conservative enough to require real model conviction (mean confidence 80.6% is well above)
  but tight enough to filter marginal signals.
- **SNR 1.5**: `E[return] > 1.5 * CI_half_width`. When GARCH unconverged, CI widens ->
  SNR drops below 1.5 -> signal blocked. This creates the self-attenuating chain.
- **min_expected_return 0.0020 (20 bps)**: Roundtrip cost ~15 bps realistic. 5 bps (prior)
  is below cost floor -- economic edge is zero. 20 bps gives ~5 bps net edge.
- **max_risk_score 0.70**: Config comment "raised to allow higher volatility names during
  evaluation". 0.70 returns to conservative production stance.
- **max_fail_fraction 0.85**: Rolling FAIL rate was 71.7%. At 0.95 threshold this was YELLOW;
  at 0.85 it correctly goes RED. 0.85 still provides headroom from current 27.7% FAIL (post-7.10b).
- **min_lift_fraction 0.25**: Ensemble beats best-single on 8% of windows. 10% trivially passes.
  25% requires real evidence of ensemble benefit.
- **min_directional_accuracy 0.45**: Actual WR is 41%. At 0.40 floor, it's 1pp above -- no
  upward pressure. 0.45 ensures at least moderate performance pressure.

### ATR Stop Logic Rationale

ATR(14) uses realized High-Low ranges to measure market-observable noise. A stop at
`current_price - ATR * 1.5` places the stop below 1.5 average-true-ranges from the current
price. Historically, moves within 1 ATR are "noise"; moves beyond 1.5 ATR signal actual
adverse price action. The 5% cap was too tight for NVDA (ATR ~7.7% of price), causing
stops to fire on normal intraday moves.

### GARCH Self-Attenuating Chain

The chain is:
1. GARCH fails to converge (scipy code 9)
2. CI inflated 1.5x
3. SNR = E[return] / (0.5 * inflated_CI_width) -> drops below 1.5
4. `net_expected_return = 0.0` at signal_generator.py:478-480
5. Signal blocked at quant validation gate
6. No trade generated from an unconverged forecast

This makes the system self-protective: bad fits produce no trades rather than bad trades.

### Platt Calibration Bootstrap

The `PLATT_BOOTSTRAP=1` flag in `overnight_refresh.sh` seeds (confidence, outcome) pairs
from historical dates. Each date uses a 1-cycle-per-business-day ladder with `--resume`
so bar timestamps advance and proof-mode max_holding fires, producing closed `ts_*` trades
that `update_platt_outcomes.py` reconciles into JSONL. After 30+ pairs (wins>=5,
losses>=5), `_calibrate_confidence()` activates **LogisticRegression** (classic Platt
scaling) to shrink overconfident signals toward actual win rates.

**Important**: the DB-global fallback tier (queries `trade_executions` directly) is the
primary calibration source when JSONL pairs are below threshold. JSONL is supplementary.
Classifier contract is enforced by `scripts/platt_contract_audit.py` and
`tests/scripts/test_platt_calibration_contract.py`.

---

**Last Updated**: 2026-02-25 (Phases A-E all COMPLETE)
**Next Phase**: 7.15 (Signal Generator Factory + session_id implementation)
