# AGENT A/B Overnight Hardening Handoff - 2026-03-09

## Scope Completed

This pass focused on production-gate integrity wiring and reporting truthfulness, not residual redesign.

1. `scripts/gate_failure_decomposition.py`
   - Added invocation-binding guard between gate artifact inputs and summary cache (`audit_dir`, `max_files`, `include_research`).
   - Added fail-safe reason decomposition when summary cache is mismatched/stale:
     - emits `UNATTRIBUTED_INVALID_CONTEXT` / `UNATTRIBUTED_NON_TRADE_CONTEXT` counts from readiness totals.
   - Removed hardcoded profitability thresholds in decomposition output:
     - now sourced from `scripts.robustness_thresholds.threshold_map()`.
   - Added markdown report output with waterfall visualization:
     - `--out-md` and default `logs/audit_gate/production_gate_decomposition_latest.md`.

2. `tests/scripts/test_gate_failure_decomposition.py`
   - Extended to validate:
     - markdown output creation,
     - invocation mismatch handling,
     - unattributed fallback reason buckets.

## Verification Evidence

### Targeted tests
- `python -m pytest tests/scripts/test_gate_failure_decomposition.py -q --basetemp C:\tmp\pytest-gfd3`
  - `3 passed`
- `python -m pytest tests/scripts/test_production_audit_gate.py -q --basetemp C:\tmp\pytest-pag3`
  - `20 passed`
- `python -m pytest tests/scripts/test_run_all_gates.py -q --basetemp C:\tmp\pytest-rag3`
  - `6 passed`
- `python -m pytest tests/scripts/test_thresholds_canonical.py -q --basetemp C:\tmp\pytest-thresh3`
  - `2 passed`
- `python -m pytest tests/scripts/test_numerical_stability.py -q --basetemp C:\tmp\pytest-numstab4`
  - `12 passed`

### Fast lane
- `python -m pytest -m "not gpu and not slow" --tb=short -q`
  - `1842 passed, 8 skipped, 28 deselected, 7 xfailed`

### Fresh gate/decomposition artifacts
- `python scripts/run_all_gates.py --json`
  - `overall_passed=false`
  - `phase3_reason=GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL`
- `python scripts/gate_failure_decomposition.py --gate-artifact logs/audit_gate/production_gate_latest.json --summary-cache logs/forecast_audits_cache/latest_summary.json --out-json logs/audit_gate/production_gate_decomposition_latest.json --out-md logs/audit_gate/production_gate_decomposition_latest.md`
  - Waterfall preserved: `82 -> 49 -> 1 -> 19 -> 0`
  - Reason breakdown now explicit:
    - `INVALID_CONTEXT`: `MISSING_EXECUTION_METADATA=27`, `HORIZON_MISMATCH=3`
    - `NON_TRADE_CONTEXT`: `NON_TRADE_CONTEXT=30`, `MISSING_TICKER=3`

### Adversarial pass
- `python scripts/adversarial_diagnostic_runner.py --json`
  - `29/29 cleared`, `0 confirmed`

## Current Production Gate Reality (post-pass)

- Still blocked by real evidence/performance issues (not wiring now):
  - Performance under policy (`profit_factor~0.60`, `win_rate~39%`, negative total PnL)
  - Thin linkage (`matched=0`, `eligible=1`)
  - Hygiene load (`non_trade=33`, `invalid=30`)

## Agent Coordination Next

### Agent A (offline follow-up)
- Upstream hygiene/source fixes only:
  - reduce `NON_TRADE_CONTEXT`, `MISSING_EXECUTION_METADATA`, `HORIZON_MISMATCH`, `MISSING_TICKER` at source.
- Keep gate thresholds unchanged.

### Agent B
- Use:
  - `logs/audit_gate/production_gate_decomposition_latest.json`
  - `logs/audit_gate/production_gate_decomposition_latest.md`
- Track waterfall and reason-code deltas run-over-run.

### Agent C (online)
- Sync readiness/blocker docs from the new decomposition outputs.
- Keep reporting taxonomy explicit by blocker component (PERFORMANCE/LINKAGE/HYGIENE).
