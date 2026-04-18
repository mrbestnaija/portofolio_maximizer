# NGN Hurdle Plan — $25k Capital Base

Date: 2026-04-18  
Capital: $25,000 (confirmed from `portfolio_cash_state.initial_capital`)  
Current state: $567.30 closed PnL / 42 trips / 79 days

---

## Critical Review of Prior Analysis

### Error 1: Wrong ROI denominator (fixes everything downstream)

Every prior calculation divided PnL by *sum of notionals* ($65,416) instead of
*portfolio capital* ($25,000). This overstates the gap by 2.6×.

| Metric | Prior (wrong) | Corrected |
|--------|---------------|-----------|
| Cumulative ROI | 0.87% | **2.27%** |
| Annualised ROI | 3.8% | **10.5%** |
| Gap multiplier to 28% | 7.4× | **2.67×** |
| Dollar gap (79-day window) | $3,603 | **$948** |

The system is not 7.4× below the hurdle. It is 2.67× below. That is a different
problem with a different solution.

### Error 2: "Notional increase" framing misapplied

Current position sizing: avg $1,558 per trade = **6.2% of $25k capital**.  
Kelly fraction at current edge (WR=38%, PF=1.73): 14.6% of capital.  
Current sizing = 6.2% / 14.6% = **42% of Kelly** — already between quarter and half-Kelly.

The system is not undersized. The reviewed text's prescription "apply fractional Kelly
at 3.5–4% of capital" would be a sizing *reduction* from where we already are. The
correct prescription is: maintain current sizing, fix the quality of what is entered.

### Error 3: `rmsevalues` NameError not verified

The reviewed text cites a live bug: "forecasting failed (`name rmsevalues is not
defined`)". `_rmse_values` is properly scoped in `ensemble.py:488`. No such NameError
exists in the current codebase. This claim cannot be acted on without verified log
evidence. It should not be in the operating plan.

### Error 4: GARCH EWMA CI inflation mischaracterised

The text says CI is inflated "specifically to *force* SNR failures." This is wrong.
The 1.5× inflation is a convergence guard — statistically correct behaviour when
GARCH fails to converge. It becomes a persistent blocker because the market is in
a sustained high-vol regime where EWMA is the universal fallback, not because the
code was designed to block entries.

The fix is regime-aware SNR calibration, not removing the guard.

### Error 5: Two levers already close the gap — no Kelly change needed

At $25k capital the dollar gap is $948.

| Lever | Conservative estimate | Cumulative |
|-------|----------------------|------------|
| Cull AAPL/GS (prevent recurrence of AAPL 7 losses + GS 5 losses) | +$467 | $1,034 |
| Fix exit geometry (3 more TIME_EXIT → TAKE_PROFIT conversions) | +$717 | $1,751 |
| **Total** | **+$1,184** | **+$1,751** |

$1,751 on $25k over 79 days = **7.0% cumulative = 32.4% annualised** — above the
28% hurdle. Without touching position sizing.

---

## What the Reviewed Text Gets Right

- **Kelly as budget not target** — correct. The CI on 42 trades is ±15pp; full Kelly
  is not safe.
- **Exit geometry is a shared lever for WR, avg win, and Kelly fraction** — correct
  and should be P0.
- **Ticker culling unlocks Kelly headroom on survivors** — correct mechanism.
- **Capacity/covariance defer until >$50k single-ticker notional** — correct, and
  at current scale (7–8 shares NVDA) even 6× is invisible to the tape.
- **Evidence-gated step promotions** — the right governance pattern.

---

## Phased Concrete Plan

The plan is ordered by what is currently the hard wall, not by conceptual priority.

### Phase 0 — Unblock the entry pipeline (Days 1–7)

**Problem**: New BUYs are fully blocked. Confidence 0.23–0.38 vs gate 0.55, SNR 0.11–0.84
vs gate 1.5. The GARCH EWMA universal fallback (high-vol regime) inflates CI 1.5×,
which halves realised SNR before it reaches the gate.

**Actions**:

| # | Action | File | Specifics |
|---|--------|------|-----------|
| 0a | Verify `_rmse_values` error claim against actual logs | `logs/` | Run one live cycle with `LOG_LEVEL=DEBUG`; confirm or dismiss |
| 0b | Add regime-aware SNR floor | `models/time_series_signal_generator.py` | When `garch_fallback=EWMA`, reduce SNR gate effective threshold: `adj_snr_gate = snr_gate / ci_inflation_factor` where `ci_inflation_factor` is read from audit artifact |
| 0c | Surface CI inflation factor in routing diagnostic | `scripts/run_auto_trader.py` | Log `garch_convergence_ok`, `ci_inflation_factor`, `effective_snr_gate` per cycle |

**Evidence gate to proceed**: ≥ 2 new qualified entries generated across 10 consecutive cycles.

**Do NOT**: lower confidence gate (0.55) or SNR gate (1.5) unconditionally. Adjust
only relative to the known inflation factor.

---

### Phase 1 — Fix exit geometry (Days 5–21)

**Problem**: TAKE_PROFIT hit rate is 9.5% (4/42 trades) despite delivering 78% of
gross profit at +$253 avg. TIME_EXIT delivers +$14 avg. AAPL had 0.03% target vs
4.87% stop — a 0.006:1 R:R that should never clear the quant gate.

**Actions**:

| # | Action | File | Specifics |
|---|--------|------|-----------|
| 1a | Enforce R:R ≥ 2:1 at quant gate | `models/time_series_signal_generator.py` | In `_calculate_targets()`: reject signal if `(target_pct / stop_pct) < 2.0`; emit `INSUFFICIENT_RR` reason code |
| 1b | Add trailing stop | `execution/paper_trading_engine.py` | Once position unrealised PnL ≥ +1× ATR: ratchet stop to break-even. Once ≥ +2× ATR: ratchet stop to +0.5× ATR (locks partial gain) |
| 1c | Extend max_hold for high-conviction | `config/signal_routing_config.yml` | `max_holding_days: 15` when entry SNR ≥ 2.0; default stays 10 bars |
| 1d | Add `rr_ratio` to routing diagnostic | `scripts/run_auto_trader.py` | Log R:R ratio on every signal accepted/rejected for audit |

**Expected outcome**: TAKE_PROFIT rate improves from 9.5% → 20%+. Conservative
estimate: 3 additional TIME_EXIT → TAKE_PROFIT conversions per 42-trade window.

**Evidence gate to proceed**: R:R gate confirmed on ≥ 5 signals in logs; trailing
stop ratchet fires at least once in live cycle.

---

### Phase 2 — Concentrate capital on positive-EV tickers (Days 14–35)

**Problem**: AAPL destroys 66% of gross profit (1W/7L, -$376). GS is 0W/5L (-$91).
Together they absorb $17,630 in notional that would generate higher returns in NVDA/GOOG/MSFT.

**Actions**:

| # | Action | File | Specifics |
|---|--------|------|-----------|
| 2a | Demote AAPL and GS to LAB_ONLY | `logs/automation/nav_rebalance_plan_latest.json` | Run `build_nav_rebalance_plan.py`; verify AAPL/GS classified WEAK; after 2 consecutive green weekly cycles, enable `live_apply_allowed=True` in `apply_nav_reallocation.py` |
| 2b | Concentrate on NVDA/GOOG/MSFT | barbell config | These already annualise at 4.7–18.5% individually at current sizing |
| 2c | Maintain current position sizing | no change | 6.2% of capital per trade = 42% of Kelly; do NOT reduce to 3.5–4% |

**Evidence gate for AAPL/GS demotion** (already statistically sound):
- AAPL: P(WR ≤ 12.5% | true WR = 45%) ≈ 2.6% — demotion justified
- GS: 0/5 wins — demotion justified

**Do NOT**: demote based on this snapshot permanently. The rolling window check in
`build_nav_rebalance_plan.py` governs re-admission automatically when rolling PF/WR
recovers.

---

### Phase 3 — Accumulate evidence (Days 35–90)

**Problem**: System is in warmup-covered state. Platt calibration inactive (0 pairs, needs 43).
Directional classifier at DA=0.562 (not gate-lifting). THIN_LINKAGE at 2/10 (needs 8 more closes).

**Actions** (passive — no code changes):

| Metric | Current | Target | Mechanism |
|--------|---------|--------|-----------|
| THIN_LINKAGE matched | 2 | ≥ 10 | Close 4 open positions + 6 new round-trips |
| Platt pairs | 0 | ≥ 43 | Every live close adds a pair |
| Classifier labels (AAPL) | ~290 | ≥ 400 | Run `run_etl_pipeline.py --tickers AAPL,MSFT,NVDA,GOOG` backfill |
| OOS coverage (AAPL) | thin | ≥ 5 fresh audit files | Run 5 `--as-of-date` ETL passes for AAPL |

**Monitoring gate to proceed to Phase 4**: over the next 20 new trades (post Phase 1/2):
- WR ≥ 42% AND PF ≥ 1.8 AND max drawdown < 15% of capital

---

### Phase 4 — Expand universe and sizing ratchet (Day 90+)

Only enter this phase if Phase 3 evidence gates pass.

**Actions**:

| # | Action | Specifics |
|---|--------|-----------|
| 4a | Expand ticker universe to 6–8 names | Post THIN_LINKAGE ≥ 10; candidates: NVDA, GOOG, MSFT, JPM, TSLA (accumulate evidence first) |
| 4b | Ratchet position sizing to half-Kelly | 8% of capital per trade (~$2,000) on core tickers with ≥ 20-trade rolling evidence |
| 4c | Covariance heuristic | Max 30% combined capital in correlated mega-cap tech (NVDA + GOOG + MSFT at any time); no factor model needed yet |
| 4d | Platt calibration activates | Confidence routing becomes calibrated; sizing scales with effective_confidence |

**Hard cap**: never exceed half-Kelly (8% of capital) until 3+ months of post-Phase-2 evidence at WR ≥ 45%, PF ≥ 1.8.

---

## The Math at Each Phase Exit

| Phase exit | Estimated cumulative PnL / 79 days | Ann ROI |
|------------|-------------------------------------|---------|
| Baseline now | $567 | 10.5% |
| After Phase 0 (entries unblocked) | $567 + new trades | 10.5%+ |
| After Phase 1 (exit fix, 3 more TAKE_PROFITs) | $567 + $717 = $1,284 | 23.7% |
| After Phase 2 (AAPL/GS culled, drag removed) | $1,284 + $467 = $1,751 | **32.4%** |
| After Phase 4 (half-Kelly, 8 tickers) | $2,000–$2,500 | 37–46% |

**NGN hurdle is cleared at Phase 2 exit without any position sizing change.**

---

## What Not To Do

| Anti-pattern | Why |
|---|---|
| Reduce position sizing to "apply quarter-Kelly" | Already at 42% of Kelly; reduction lowers returns |
| Remove GARCH EWMA CI inflation guard unconditionally | Statistically correct; fix via regime-aware SNR adjustment only |
| Target 79% win rate to hit hurdle via WR alone | Requires WR of ~79% at current sizing — not achievable |
| Add covariance model before $50k+ single-ticker notional | Premature engineering; concentration caps are sufficient |
| Force Platt calibration before 43 pairs | Degrades calibration; let it activate naturally |
| Lower SNR gate below 1.5 unconditionally | Fix the CI inflation input, not the gate |

---

## Summary: Why Two Levers Are Sufficient

At $25k capital the gap is $948 — not $3,603. That changes the plan entirely.

The prior analysis prescribed Kelly sizing ratchets, covariance modeling, and universe
expansion because it was solving for a 7.4× improvement against a wrong denominator.
The real problem is a 2.67× improvement against a $25k capital base. That is solved by:

1. **Phase 1** — Fix exits so winners reach TAKE_PROFIT. Estimated +$717 (conservative).
2. **Phase 2** — Demote AAPL/GS so capital stops being destroyed. Estimated +$467.

Those two changes, both code or governance (not capital deployment), close $1,184 of
the $948 gap. Everything else (Kelly ratchets, covariance, universe expansion) is Phase 4
and later — important for sustaining and scaling, but not necessary to first cross the hurdle.
