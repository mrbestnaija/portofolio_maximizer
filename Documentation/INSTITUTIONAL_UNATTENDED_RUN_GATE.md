# Institutional Unattended-Run Gate

This document defines the hardening gate for unattended/autonomous operation claims.

Related navigator:
- [DOCS_INDEX.md](DOCS_INDEX.md)

## Scope

The gate enforces contracts that must not drift:

- Runtime package-install policy (supply-chain hardening).
- Prompt-injection blocking defaults.
- Checkpoint load safety (ID validation + safe unpickling).
- Forecast/audit default synchronization (`max_files` contract).
- Overnight refresh error accounting integrity.
- Platt contract audit standalone executability.
- Repo hygiene (no shadow duplicate runtime files).

## Gate Command

Human-readable:

```powershell
python scripts/institutional_unattended_gate.py
```

Machine-readable:

```powershell
python scripts/institutional_unattended_gate.py --json
```

Exit behavior:

- `0`: no `FAIL` findings.
- `1`: one or more `FAIL` findings.

## Phase Map

- `P0`: Security hardening contracts.
- `P1`: Operational repo-sync contracts.
- `P2`: Platt calibration contract executability.
- `P3`: Repo hygiene contracts.

## Mandatory Readiness Evidence

Use this full set before any "stable/safe for unattended runs" claim:

```powershell
python scripts/institutional_unattended_gate.py --json
python scripts/run_all_gates.py --json
python -m pytest tests/scripts/test_institutional_unattended_contract.py tests/scripts/test_institutional_unattended_gate.py tests/scripts/test_llm_runtime_install_policy.py tests/scripts/test_platt_calibration_contract.py tests/scripts/test_run_all_gates.py -q
python -m pytest -m "not gpu and not slow" --tb=short -q
```

Do not use skip flags (`--skip-forecast-gate`, `--skip-profitability-gate`, `--skip-institutional-gate`) as final evidence.

## Interpretation Rules

- `institutional_unattended_gate=PASS` means hardening contracts are intact.
- `run_all_gates=PASS` is required for unattended readiness.
- `run_all_gates=FAIL` with only `production_audit_gate` failing means:
  - Infrastructure hardening is intact.
  - Readiness is blocked by lift/profitability policy, not by security wiring.

## CI Wiring

Institutional checks are covered by:

- `tests/scripts/test_llm_runtime_install_policy.py`
- `tests/scripts/test_run_ensemble_diagnostics_security.py`
- `tests/scripts/test_institutional_unattended_contract.py`
- `tests/scripts/test_institutional_unattended_gate.py`
- `tests/scripts/test_platt_calibration_contract.py`
- `tests/scripts/test_run_all_gates.py`
- `tests/utils/test_openclaw_cli.py`
- `tests/etl/test_checkpoint_manager.py`

Blocking CI step:

```powershell
python scripts/institutional_unattended_gate.py --json
```
