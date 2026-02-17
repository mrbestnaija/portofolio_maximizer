# Portfolio Maximizer - Quick Reference Card

**Last Updated**: 2026-02-17 (Phase 7.9 Complete)
**System Status**: Research Phase - Adversarial findings under review

**Ensemble status (canonical, current)**: `Documentation/ENSEMBLE_MODEL_STATUS.md` (per-forecast policy labels vs aggregate audit gate). Use this as the single source of truth for external-facing ensemble claims.

---

## Quick Start Commands

### Run Pipeline (Multi-Ticker)
```bash
python scripts/run_etl_pipeline.py \
  --tickers AAPL,MSFT,NVDA \
  --start 2024-07-01 \
  --end 2026-02-17 \
  --execution-mode auto
```

### Launch Autonomous Trading
```bash
python scripts/run_auto_trader.py \
  --tickers AAPL,MSFT,NVDA \
  --lookback-days 365 \
  --cycles 5 \
  --sleep-seconds 900
```

### Run PnL Integrity Audit
```bash
python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db
```

### Check Production Gate
```bash
python scripts/production_audit_gate.py
```

### Check Canonical Metrics (Correct Way)
```bash
python -c "
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as e:
    m = e.get_canonical_metrics()
    print(f'Round-trips: {m.total_trades}, PnL: \${m.total_realized_pnl:+,.2f}')
    print(f'Win rate: {m.win_rate:.1%}, Profit factor: {m.profit_factor:.2f}')
"
```

### Run Proof-Mode Audit Sprint
```bash
PROOF_MODE=1 RISK_MODE=research_production bash bash/run_20_audit_sprint.sh
```

---

## Current Performance (Phase 7.9)

### Production Metrics (2026-02-14)

| Metric | Value |
|--------|-------|
| Round-trips | 37 |
| Total PnL | $673.22 |
| Win Rate | 43.2% (16W/21L) |
| Profit Factor | 1.85 |
| Avg Win | $91.59 |
| Avg Loss | $34.54 |
| Win/Loss Ratio | 2.65x |
| Largest Win | $497.83 |
| Integrity | ALL PASSED (0 violations) |
| Forecast Gate | PASS (21.4%, threshold 25%) |

### Adversarial Findings (2026-02-16)

| Finding | Severity | Status |
|---------|----------|--------|
| 94.2% quant FAIL rate (0.8% from RED gate) | P0 | Open |
| Ensemble worse than best single 92% of time | P0 | Open |
| Directional accuracy below coin-flip (41% WR) | P0 | Open |
| Confidence calibration broken (0.9+ -> 41% WR) | P1 | Open |
| AAPL -$325 drag, GS 0-for-5 | P1 | Open |
| signal_id NULL for all trades | P2 | Open |

Full details: [ADVERSARIAL_AUDIT_20260216.md](Documentation/ADVERSARIAL_AUDIT_20260216.md)

---

## System Architecture

### Forecasting Models
- **GARCH**: Volatility forecasting (best for liquid, range-bound)
- **SARIMAX**: Linear time series (off by default, 15x speedup)
- **SAMoSSA**: Spectral decomposition (trending markets, dominates all regimes)
- **MSSA-RL**: Change-point detection + RL (under-weighted at 8.7%)

### LLM Models (3-Model Local Strategy)
- **deepseek-r1:8b**: Fast reasoning (chain-of-thought, math, code-gen)
- **deepseek-r1:32b**: Heavy reasoning (deep analysis, long-context)
- **qwen3:8b**: Tool orchestrator (function-calling, structured output)

### OpenClaw Cron Jobs (9 active)

| Job | Schedule | Announce When |
|-----|----------|---------------|
| [P0] PnL Integrity Audit | Every 4h | CRITICAL/HIGH violations |
| [P0] Production Gate Check | Daily 7 AM | Gate FAIL or RED |
| [P0] Quant Validation Health | Daily 7:30 AM | FAIL rate >= 90% |
| [P1] Signal Linkage Monitor | Daily 8 AM | Orphan opens/unlinked closes |
| [P1] Ticker Health Monitor | Daily 8:30 AM | 3+ consecutive losses or PnL < -$300 |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | Unit-root rate >= 35% |
| [P2] Overnight Hold Monitor | Weekly Fri 9 AM | Overnight drag > 25% |
| System Health Check | Every 6h | Model offline or errors |
| Weekly Session Cleanup | Sunday 3 AM | Never (silent) |

---

## Key File Locations

### Core Code
- `forcester_ts/forecaster.py` - Main forecasting engine
- `forcester_ts/ensemble.py` - Ensemble coordinator
- `integrity/pnl_integrity_enforcer.py` - PnL integrity (6 checks, CI gate)
- `execution/paper_trading_engine.py` - Paper trading engine
- `models/time_series_signal_generator.py` - Signal router
- `config/forecasting_config.yml` - Model parameters + ensemble config

### Operations
- `scripts/production_audit_gate.py` - Production readiness gate
- `scripts/ci_integrity_gate.py` - CI integrity gate
- `scripts/openclaw_models.py` - OpenClaw model management
- `scripts/pmx_interactions_api.py` - Interactions API (FastAPI)
- `scripts/llm_multi_model_orchestrator.py` - Multi-model orchestrator
- `bash/run_20_audit_sprint.sh` - Audit sprint with lockfile

### Documentation
- `Documentation/ADVERSARIAL_AUDIT_20260216.md` - Current adversarial findings
- `Documentation/OPENCLAW_INTEGRATION.md` - OpenClaw + LLM + Interactions API
- `Documentation/EXIT_ELIGIBILITY_AND_PROOF_MODE.md` - Proof-mode spec
- `Documentation/ENSEMBLE_MODEL_STATUS.md` - Ensemble governance labels
- `CLAUDE.md` - Agent guidance
- `AGENTS.md` - Agent guardrails + cron rules

---

## Troubleshooting

### PnL Integrity Violations
```bash
# Run full audit
python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db

# Fix opening legs with PnL (dry run)
python -m integrity.pnl_integrity_enforcer --fix-opening-pnl

# Apply all fixes
python -m integrity.pnl_integrity_enforcer --fix-all --apply
```

### OpenClaw Issues
```bash
# Check cron status
openclaw cron list

# Force-run a job
openclaw cron run <job-id> --timeout 120000

# Check gateway
openclaw gateway status

# Check models
python scripts/openclaw_models.py status --list-ollama-models
```

### Pipeline Failures
```bash
# Validate environment
python scripts/validate_environment.py

# Validate credentials (no values shown)
python scripts/validate_credentials.py

# Check LLM health
python scripts/llm_multi_model_orchestrator.py status
```

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Forecast Gate | 21.4% | <25% | PASS |
| PnL Integrity | 0 violations | 0 violations | PASS |
| Win Rate | 43.2% | >50% | Needs work |
| Profit Factor | 1.85 | >2.0 | Close |
| Quant FAIL Rate | 94.2% | <90% | P0 priority |
| Ensemble vs Best Single | Worse 92% | Worse <50% | P0 priority |

---

**Version**: Phase 7.9 Complete
**Next**: Phase 7.10 (Production Hardening - address adversarial findings)
