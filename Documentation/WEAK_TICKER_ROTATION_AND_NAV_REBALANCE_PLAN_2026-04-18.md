# Weak Ticker Rotation and NAV Rebalance Plan

Date: 2026-04-18

## Purpose

This document formalizes the highest-ROI alpha repair move currently available:

- demote or sharply reduce weak tickers such as `AAPL` and `GS` until their rolling PF / win-rate evidence recovers,
- concentrate capital toward the currently productive names such as `NVDA`, `MSFT`, and `GOOG`,
- keep the evidence-first gates as the admission rule so allocation stays governed, not discretionary.

The malformed `~/.openclaw/openclaw.json` runtime issue is intentionally out of scope here. That is an infrastructure unblock, not the alpha-repair path.

## Current snapshot

The latest performance summary already shows a clear asymmetry:

- `AAPL`: weak win rate and negative PnL
- `GS`: weak win rate and negative PnL
- `NVDA`, `MSFT`, `GOOG`: positive PnL and materially better rolling evidence

The repo also already has the right pieces to support automation:

- `scripts/compute_ticker_eligibility.py`
- `scripts/apply_ticker_eligibility_gates.py`
- `scripts/evaluate_sleeve_promotions.py`
- `risk/nav_allocator.py`
- `scripts/build_automation_dashboard.py`
- `scripts/run_quality_pipeline.py`

That means the next step is not a new gate. It is a narrow rotation governor that consumes the existing evidence outputs and turns them into NAV and eligibility actions.

## Target operating model

The future automation should run as a single governed pipeline:

1. Compute ticker health from realized trades.
2. Compute sleeve health from realized sleeve PnL.
3. Produce a rotation plan that classifies each symbol as:
   - `HEALTHY`
   - `WEAK`
   - `LAB_ONLY`
4. Convert the rotation plan into NAV budgets and symbol weights.
5. Expose the resulting plan and rationale in the dashboard.
6. Keep live routing fail-closed if evidence health or preprocessing health is weak.

## Evidence contract

Promotion is allowed only when the following are true on a rolling window:

- `win_rate` is above the promotion floor
- `profit_factor` is above the promotion floor
- trade count is high enough to be statistically meaningful
- evidence quality is acceptable:
  - OOS coverage is sufficient
  - missing-metric counts are bounded
  - lift CI is not stale or negative
  - heuristic fallback is either absent or explicitly guarded
- preprocessing health is acceptable:
  - imputed fraction is bounded
  - padding fraction is bounded
  - post-preprocess validation passes
- provenance is trusted:
  - data source is known
  - fallback or synthetic paths remain visible

Demotion is allowed when the rolling window erodes:

- `win_rate` falls below the demotion floor, or
- `profit_factor` falls below break-even / safety floor, or
- evidence health degrades, or
- preprocessing distortion increases, or
- provenance becomes untrusted.

## Recommended starting policy

Use staged thresholds rather than one-off overrides:

| Status | Condition | Live action |
|---|---|---|
| `HEALTHY` | Promotion thresholds passed on a rolling window | Eligible for full or increased NAV budget |
| `WEAK` | Enough evidence exists, but PF / WR are below promotion floors | NAV is reduced, not removed |
| `LAB_ONLY` | Missing evidence, poor provenance, or persistent failure | No live NAV allocation |

Recommended starting floors:

- promotion `win_rate >= 0.55`
- promotion `profit_factor >= 1.20`
- demotion `win_rate <= 0.45`
- demotion `profit_factor <= 0.90`
- minimum trades per decision window: `10`

These are starting points, not hardcoded forever. They should be versioned and reviewed with the evidence snapshots.

## Automation pipeline

### 1) Ticker health

Continue using `scripts/compute_ticker_eligibility.py` as the source of truth for per-ticker PnL health.

Enhance the output to preserve:

- rolling window size
- realized win rate
- realized profit factor
- total PnL
- reason codes

### 2) Eligibility gate

Continue using `scripts/apply_ticker_eligibility_gates.py` as the read-only translation layer.

Future automation should use that sidecar to decide whether a symbol remains:

- eligible for live routing,
- capped in size,
- or research-only.

### 3) Sleeve promotion / demotion

Extend `scripts/evaluate_sleeve_promotions.py` so it becomes a rolling sleeve governor instead of a one-shot summary.

It should emit:

- promotions
- demotions
- hold / no-action cases
- the reason each decision was made

### 4) NAV allocation

Wire the rotation plan into `risk/nav_allocator.py` so NAV is allocated by bucket and then by symbol.

The intended behavior is:

- healthy names get normal or increased budget
- weak names get reduced budget
- lab-only names receive no live budget

### 5) Dashboard visibility

Expose the rotation state in `scripts/build_automation_dashboard.py` and the automation dashboard artifact so the operator can see:

- who was demoted
- who was promoted
- why the decision happened
- whether the decision was evidence-driven or blocked by health contracts

## Implementation phases

### Phase 1: Plan only

- keep the current live lane unchanged
- generate `logs/automation/nav_rebalance_plan_latest.json` via `scripts/build_nav_rebalance_plan.py`
- show the artifact in the dashboard snapshot and weekly sleeve maintenance output
- verify that the plan correctly identifies `AAPL` and `GS` as weak

### Phase 2: Shadow mode

- compute NAV reallocations without applying them to live orders
- compare shadow allocations against realized PnL
- verify that `NVDA`, `MSFT`, and `GOOG` would receive more capital under the governed policy
- keep `live_apply_allowed=false` until the evidence gate is green for sustained weekly windows

### Phase 3: Controlled live enablement

- apply the allocation only to healthy or promoted symbols
- keep weak symbols constrained
- demote symbols automatically when rolling evidence erodes
- require at least two consecutive healthy weekly cycles before gate lift is eligible

## Guardrails

- No manual threshold dodge to force a promotion
- No hidden second gate
- No silent fallback that bypasses evidence contracts
- No live allocation for untrusted provenance
- No promotion when preprocessing distortion is excessive

## Test plan

The implementation should add tests for:

- ticker eligibility classification
- rolling promotion / demotion decisions
- NAV allocator application
- dashboard payload visibility
- demotion on evidence erosion
- promotion on recovered PF / win-rate evidence

Recommended regression coverage:

- `AAPL` and `GS` demote or stay constrained under current evidence
- `NVDA`, `MSFT`, and `GOOG` remain promotable only when their evidence passes
- evidence-health failures keep a symbol in research-only mode

## Outcome

The goal is to turn the current asymmetry into a governed allocation rule:

- weak names are automatically constrained
- strong names receive more capital
- the gate remains evidence-first
- the system stays explainable and auditable
