# Alpha Gap Decomposition — NGN Hurdle Analysis

Date: 2026-04-18  
Baseline snapshot: 42 production round-trips, 2026-01-16 → 2026-04-10 (84 days)

---

## The Gap

| Metric | Current | NGN Hurdle |
|--------|---------|-----------|
| Cumulative ROI | 0.87% / 84 days | 6.4% / 84 days |
| Annualised | 3.8% | 28% |
| Total PnL on $65k notional | +$567 | +$4,170 |
| EV per trade (42 trips) | +$13.50 | +$99.30 |
| Gap multiplier | 1× | **7.4×** |

The system is not alpha-positive against a meaningful hurdle. It survives entirely on
magnitude asymmetry: avg win $91.59 vs avg loss $34.54 (2.65× ratio) despite a 38% win
rate. The NGN hurdle (28% ann) requires either a 7.4× increase in EV/trade or a
proportional improvement spread across multiple levers.

---

## Ticker-Level Truth

| Ticker | Trips | PnL | W/L | Ann ROI on deployed | Verdict |
|--------|-------|-----|-----|---------------------|---------|
| NVDA   | 9  | +$682 | 5/4 | +18.5% | Core — scale up |
| MSFT   | 6  | +$188 | 4/2 | +6.6%  | Core — maintain |
| GOOG   | 10 | +$180 | 5/5 | +4.7%  | Core — maintain |
| JPM    | 2  | +$13  | 1/1 | +3.6%  | Neutral — accumulate evidence |
| TSLA   | 1  | -$1   | 0/1 | n/a    | Insufficient data |
| AMZN   | 1  | -$28  | 0/1 | n/a    | Insufficient data |
| GS     | 5  | -$91  | 0/5 | negative | **Demote — 0 wins in 5 trips** |
| AAPL   | 8  | -$376 | 1/7 | negative | **Demote — primary PnL anchor** |

AAPL alone destroys 66% of the system's gross profit ($376 loss on $567 total PnL).
GS destroys a further 16%. Combined drag: -$467. Without them the system would have
made +$1,034 and annualised at ~9.4%.

---

## Exit Reason Truth

| Exit | N | Total PnL | Avg PnL | Interpretation |
|------|---|-----------|---------|----------------|
| TAKE_PROFIT | 4 | +$1,012 | +$253 | The only real alpha events. Hit rate: 9.5%. |
| TIME_EXIT   | 24 | +$341 | +$14 | Marginally positive carry. Volume driver. |
| STOP_LOSS   | 13 | -$785 | -$60 | Primary loss source. 31% of all exits. |
| legacy_close | 1 | -$0   | — | Negligible |

TAKE_PROFIT events deliver 78% of gross profit from only 9.5% of trades.
STOP_LOSS events cost $785 across 13 trades — larger absolute drag than the
TIME_EXIT gains.

The system currently runs like a breakeven carry strategy punctuated by rare
large wins. The lever to pull is: let winners run to TAKE_PROFIT more often,
not just to TIME_EXIT.

---

## Stack-Layer Decomposition

Each layer below is analysed for (a) current state, (b) target state,
(c) estimated PnL impact on 42-trade sample, and (d) annualised contribution.

---

### Layer 1 — Ticker Allocation (Lever size: ~+5pp ann)

**Problem**: $17,630 capital tied up in AAPL (8 trips, 12.5% WR) and GS (5 trips, 0% WR).
This is structurally misallocated capital.

**Fix**: Execute the NAV rebalance plan already in shadow mode.
- Move AAPL/GS to LAB_ONLY (no live allocation).
- Redeploy $17,630 into NVDA/GOOG/MSFT at their observed blended annualised rate (~10%).
- Additional PnL from redeployment: $17,630 × (10% × 84/365) ≈ +$405.
- Stop accumulating AAPL/GS losses: +$467 saved.
- Net 84-day gain: ~+$870.
- Ann contribution: +$870/$65k × 365/84 ≈ **+5.8pp ann**.

**Status**: `build_nav_rebalance_plan.py` already classifies AAPL/GS as WEAK.
`live_apply_allowed=False` — needs 2 consecutive green weekly cycles to lift.
**Action**: Run weekly cycles, confirm classification stable, enable live apply.

---

### Layer 2 — Take-Profit Capture Rate (Lever size: ~+9pp ann, highest impact)

**Problem**: Only 4/42 trades (9.5%) reach TAKE_PROFIT. Yet those 4 deliver $1,012 —
more than the entire system's net PnL. TIME_EXIT at 10 bars captures $14/trade avg,
leaving potential on the table.

**Diagnosis**:
- ATR-based targets are set at 1.5–2.0× ATR. With GARCH EWMA in high-vol regime,
  ATR is inflated → targets are placed further away → fewer hits.
- Max-hold = 10 bars. Winners that need 12–15 bars to develop get TIME_EXIT'd.

**Fixes**:
1. Extend max_hold from 10 to 15 bars on signals with SNR > 2.0 (high-conviction exits).
2. Trail stop once position is +1.5× ATR in profit (ratchet stop to entry, let winner run).
3. Review target distance: AAPL target was $0.09 away on a $270 price — 0.03% target on
   a 4.87% stop. This is asymmetric risk/reward in the wrong direction.

**Estimated impact**: Converting 5 additional TIME_EXITs to TAKE_PROFITs:
+5 × ($253 − $14) = +$1,195 incremental PnL.
Ann contribution: +$1,195/$65k × 365/84 ≈ **+8.0pp ann**.

**Status**: Code change required in `models/time_series_signal_generator.py`
(`_calculate_targets()`) and `execution/paper_trading_engine.py` (trailing stop logic).

---

### Layer 3 — Stop-Loss Reduction via Entry Quality (Lever size: ~+3pp ann)

**Problem**: 13 stop-loss exits totalling -$785. Avg loss -$60.42.
Stop-loss exits concentrate heavily in AAPL (primary) and GS.

**Diagnosis**:
- Stop-loss is not a tuning problem — it is an entry quality problem.
  Stops are firing because entries are wrong, not because stops are too tight.
- Directional classifier (DA=0.562) is slightly above random but not yet gate-lifting
  (needs more data). Gate-lift would block low-DA entries before they become stop-losses.
- OOS coverage thin for AAPL → RMSE-rank disabled → ensemble picks suboptimal model
  → directional forecast less reliable.

**Fixes**:
1. Block entries where classifier p_up < 0.52 (current threshold 0.55 — verify it's enforced).
2. Accumulate 100+ directional labels per ticker to push classifier DA toward 0.60+.
3. Run ETL pipeline for AAPL/GS OOS backfill to activate RMSE-rank weighting.

**Estimated impact**: -30% stop-loss frequency (4 fewer stops):
+4 × $60.42 = +$242 retained.
Ann contribution: +$242/$65k × 365/84 ≈ **+1.6pp ann**.

---

### Layer 4 — Opportunity Rate / Ticker Universe (Lever size: ~+3pp ann)

**Problem**: 42 trades / 84 days = 0.5 trades/day. Current live signals blocked by
low SNR (0.11–0.84 vs gate 1.5) — new BUYs are entirely blocked.

**Root cause**: GARCH EWMA universal fallback (high-vol regime) inflates CI 1.5×.
Wide CI → low SNR → signal blocked at routing gate. This is a market-conditions
constraint that will ease when volatility normalises.

**What can be controlled**:
1. Expand ticker universe from 5 to 8–10 names once THIN_LINKAGE ≥ 10.
   More tickers = more chances to find signals that clear the SNR gate.
2. Add 2 intraday cycles (currently 2×/day NYSE). With 3–4 cycles: 50% more
   opportunity capture at same gate quality.
3. When GARCH convergence improves in lower-vol regime, SNR recovers naturally
   (no code change needed — market condition).

**Estimated impact** (universe expansion only): 8 tickers at current 0.5 trades/ticker/84days
= 0.5 × 8/5 × $13.50/trade × 42 trades_equiv ≈ +$227 incremental.
Ann contribution: **+1.5pp ann** conservatively.

---

### Layer 5 — Win Rate via Confidence Calibration (Lever size: ~+3pp ann)

**Problem**: 38% win rate. Win rates below 45% indicate that roughly 2/5 entries are
directionally wrong at the threshold confidence level.

**Current calibration state**:
- Platt scaling: inactive (0 pairs accumulated, needs 43 minimum).
- Directional classifier: DA=0.562, not yet gate-lifting at p_up > 0.55.
- Edge score: calibrated to /10.0 (full credit at 2% return). Correct.
- SNR upper anchor: calibrated to /3.0 (full credit at SNR=3.5). Correct.

**Path to 45% WR**:
- Platt calibration activates naturally as live trades close (43 pairs needed).
  With 4 open positions + 4+ new round-trips: accumulates toward threshold.
- Classifier needs 100+ labeled samples per ticker for stable gate-lift.
- These are passive accumulation steps — no code changes needed.

**Estimated impact**: WR 38% → 45% on 42 trades:
+3 wins (× $91.59) + -3 losses saved (× $34.54) = +$274 + +$104 = +$378.
Ann contribution: +$378/$65k × 365/84 ≈ **+2.5pp ann**.

---

### Layer 6 — Position Sizing Efficiency (Lever size: +1–2pp ann)

**Problem**: Avg notional $1,558 per trade. This is consistent sizing regardless of
signal conviction level. High-conviction signals (conf ≥ 0.70, SNR ≥ 2.5) should
receive larger allocations within barbell constraints.

**Current state**:
- Platt-calibrated confidence is inactive (no pairs yet).
- Barbell max positions: speculative ≤ 10%, core ≤ 20%.
- No conviction-scaled sizing within those limits.

**Fix**: Add fractional-Kelly scaling tied to effective_confidence.
At effective_confidence ≥ 0.65 (post-Platt): allow up to 2× base notional.
At effective_confidence < 0.55: reduce to 0.5× base notional.

**Estimated impact**: +20% avg notional on winning trades, -20% on losers
(if confidence correlates with outcome, which Platt activation will verify).
Conservative ann contribution: **+1.0–2.0pp ann** once Platt is active.

---

## Combined Lever Estimate

| Lever | Est. +Ann | Priority | Blocker |
|-------|-----------|----------|---------|
| 1. Ticker culling (AAPL/GS) | +5.8pp | **P0** | 2 green weekly cycles |
| 2. Take-profit capture rate | +8.0pp | **P0** | Code change: trailing stop + max_hold extension |
| 3. Stop-loss reduction (entry quality) | +1.6pp | P1 | Data accumulation (classifier, OOS backfill) |
| 4. Opportunity rate / universe | +1.5pp | P1 | THIN_LINKAGE ≥ 10 |
| 5. Win rate (Platt + classifier) | +2.5pp | P1 | Passive accumulation (43 pairs, 100 labels) |
| 6. Position sizing (Kelly) | +1.5pp | P2 | Platt activation first |
| **Total** | **~20.9pp** | | |

Current baseline: 3.8pp ann  
Combined target: 3.8 + 20.9 = **24.7pp ann**

This falls short of the 28% NGN hurdle by ~3.3pp. The remaining gap closes with:
- Compound effect of better entries AND larger sizing (levers 3+6 interact).
- Market vol regime normalisation (GARCH EWMA fallback exits → SNR recovers → more entries).
- Each additional full year of operation accumulates Platt pairs, classifier labels, and
  OOS audit files, shifting all three calibration levers upward.

---

## The Critical Path

The single highest-ROI action is **Lever 2 (take-profit capture)**. This is a code
change, not a data accumulation problem. It can move the needle immediately:

1. Review `_calculate_targets()` — ensure target distance is ≥ 2× stop distance (minimum
   R:R of 2:1). Currently AAPL had 0.03% target vs 4.87% stop — that is 0.006:1 R:R.
   A signal with that geometry should never pass the quant gate.

2. Add trailing stop logic — once position is +1× ATR in profit, ratchet stop to break-even.
   This converts potential TIME_EXIT losers to TIME_EXIT breakeven or TAKE_PROFIT.

3. Extend max_hold to 15 bars for high-conviction entries (SNR ≥ 2.0).

**The second highest-ROI action is Lever 1 (ticker culling)** — already built and in
shadow mode. The only remaining step is 2 consecutive green weekly cycles, then enable.

---

## What the NGN Hurdle Actually Demands

28% annualised on $65k notional = +$18,200/year = +$4,200/84-day window.

To reach $4,200 in 84 days with 42 trades: $100/trade EV.
Current EV/trade: $13.50.

Decomposed path to $100 EV/trade:
- WR 45%, avg win $120 (trailing stop extends winners), avg loss $40 (tighter entries):
  EV = 0.45 × $120 − 0.55 × $40 = $54 − $22 = $32/trade. Still below $100.
- At $32/trade, need 84 trades in 84 days (1 trade/day) to hit $2,700 — 15% ann.

The honest conclusion: **$65k notional at paper trading scale cannot reach 28% ann
without either (a) compounding gains into larger positions, or (b) increasing trade
frequency to 1–2 per day**. The system structure is sound; the capital base and
opportunity rate are the binding constraints beyond the alpha improvements above.

At $200k deployed capital with 1 trade/day and $32 EV/trade:
$32 × 365 = $11,680/year on avg $1,558 notional/trade.
On $200k that is still only 5.8% ann — underlining that position sizing (Kelly scaling
to full conviction) is the lever that bridges the final gap.

---

## Open Actions (Prioritised)

| # | Action | Layer | File(s) | Status |
|---|--------|-------|---------|--------|
| A1 | Fix target geometry: ensure R:R ≥ 2:1 before signal passes quant gate | 2 | `time_series_signal_generator.py` | OPEN |
| A2 | Add trailing stop: ratchet to break-even after +1× ATR profit | 2 | `paper_trading_engine.py` | OPEN |
| A3 | Extend max_hold 10→15 bars for SNR ≥ 2.0 signals | 2 | `run_auto_trader.py` config | OPEN |
| A4 | Enable NAV rebalance live apply after 2 green weekly cycles | 1 | `apply_nav_reallocation.py` | BLOCKED (data) |
| A5 | OOS ETL backfill for AAPL (5 historical dates) | 3 | `run_etl_pipeline.py --tickers AAPL` | OPEN |
| A6 | THIN_LINKAGE accumulation: close 4 open positions | 4 | passive | OPEN (market) |
| A7 | Expand universe to 8 tickers post THIN_LINKAGE ≥ 10 | 4 | `signal_routing_config.yml` | BLOCKED (gate) |
| A8 | Kelly scaling tied to effective_confidence | 6 | `paper_trading_engine.py` | BLOCKED (Platt) |
