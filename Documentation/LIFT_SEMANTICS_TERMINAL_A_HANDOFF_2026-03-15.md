# Lift Semantics Terminal A Handoff (2026-03-15)

Doc Type: `integration_note`

Role: Terminal A (`Lift Semantics / Measurement`)

This note prepares the reviewed lift-semantics patch intake for Terminal B on the local
integration branch `integration/readiness-20260315`.

## Scope

This handoff covers only the lift-semantics and measurement side of gate lifting:

- readiness denominator truth
- structured RMSE/lift summary truth
- structured summary consumption in the production audit gate
- baseline-definition parity for live signal measurement
- Layer 1 audit-source wiring for model-improvement measurement

It does not include evidence-quality writer tightening, clean-cohort promotion, OpenClaw
ops work, or dashboard integration.

## Patch Bundles

Preferred intake order:

1. Gate truth patch
   - `Documentation/patch_bundles/lift_semantics_gate_truth_2026-03-15.reviewed.patch`
   - `Documentation/patch_bundles/lift_semantics_gate_truth_2026-03-15.reviewed.files.txt`
   - `Documentation/patch_bundles/lift_semantics_gate_truth_2026-03-15.reviewed.sha256.txt`

2. Baseline parity patch
   - `Documentation/patch_bundles/lift_semantics_baseline_parity_2026-03-15.reviewed.patch`
   - `Documentation/patch_bundles/lift_semantics_baseline_parity_2026-03-15.reviewed.files.txt`
   - `Documentation/patch_bundles/lift_semantics_baseline_parity_2026-03-15.reviewed.sha256.txt`

Reference aggregate bundle:

- `Documentation/patch_bundles/lift_semantics_measurement_2026-03-15.reviewed.patch`

The split bundles are preferred because they align to the two measurement concerns and are
safer to validate one at a time.

## Files In Scope

Gate truth bundle:

- `scripts/check_forecast_audits.py`
- `scripts/production_audit_gate.py`
- `tests/scripts/test_check_forecast_audits.py`
- `tests/scripts/test_production_audit_gate.py`

Baseline parity bundle:

- `models/time_series_signal_generator.py`
- `scripts/check_model_improvement.py`
- `tests/models/test_time_series_signal_generator.py`
- `tests/scripts/test_check_model_improvement.py`

## Verification Evidence

Targeted tests:

```powershell
python -m pytest tests/scripts/test_check_forecast_audits.py tests/scripts/test_production_audit_gate.py tests/models/test_time_series_signal_generator.py tests/scripts/test_check_model_improvement.py -q --basetemp C:\tmp\pmx_lift_semantics_terminal_a
```

Result:

- `117 passed`

Live measurement checks:

```powershell
python scripts/check_model_improvement.py --layer 1 --json
python scripts/production_audit_gate.py --unattended-profile
```

Observed current live state:

- Layer 1 remains a real `FAIL`
  - `n_total_files=91`
  - `n_used_windows=28`
  - `coverage_ratio=0.3077`
  - `lift_fraction_global=0.0714`
  - `lift_ci=[-0.3672,-0.1846]`
- Production audit gate remains a real `FAIL`
  - `reason=GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL`
  - `matched=0/1`
  - `closed=41/30`
  - `days=11/21`

These are acceptable outcomes for this slice. The goal here is measurement truth, not a
policy change or synthetic green status.

## Important Diff Note

`origin/master...HEAD` has no merge base in this reference workspace, so these patch bundles
were generated using direct tree comparison:

```powershell
git diff origin/master HEAD -- <paths>
```

That is intentional for Terminal A patch production in this lane.

## Terminal B Apply Commands

Apply on `C:\Users\Bestman\personal_projects\pmx_readiness_integration`:

```powershell
git apply --index --3way C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\patch_bundles\lift_semantics_gate_truth_2026-03-15.reviewed.patch
python -m pytest tests/scripts/test_check_forecast_audits.py tests/scripts/test_production_audit_gate.py -q

git apply --index --3way C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\patch_bundles\lift_semantics_baseline_parity_2026-03-15.reviewed.patch
python -m pytest tests/models/test_time_series_signal_generator.py tests/scripts/test_check_model_improvement.py -q
```

Then run:

```powershell
python scripts/run_all_gates.py --json
python -m integrity.pnl_integrity_enforcer
python -m pytest -m "not gpu and not slow" --tb=short -q
```

## Decision Guidance

If the gate-truth patch applies cleanly and the targeted tests pass, Terminal B should take
it first because it improves the trustworthiness of lift failure reporting without changing
threshold policy.

If the baseline-parity patch also applies cleanly, Terminal B should take it second because
it reduces the mismatch between execution-side and audit-side lift judgment.
