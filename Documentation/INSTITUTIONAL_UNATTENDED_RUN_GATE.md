# Institutional Unattended-Run Gate

This repository now includes a phased institutional hardening gate for autonomous operation safety.

## Scope

The gate enforces controls that previously drifted:

- Runtime package-install policy (supply-chain hardening)
- Prompt-injection guard defaults
- Checkpoint load safety (ID/path validation + safe unpickling)
- Forecast gate default synchronization across scripts
- Overnight refresh error accounting integrity
- Platt contract audit standalone execution
- Repo hygiene (shadow duplicate runtime files)

## Phased Gate

Run:

```powershell
python scripts/institutional_unattended_gate.py
```

JSON output:

```powershell
python scripts/institutional_unattended_gate.py --json
```

Phases:

- `P0` Security hardening contracts
- `P1` Operational repo-sync contracts
- `P2` Platt calibration contract executability
- `P3` Repo hygiene contracts

Exit behavior:

- `0`: no `FAIL` findings
- `1`: one or more `FAIL` findings

## CI Test Coverage

Institutional contract coverage is enforced by:

- `tests/scripts/test_llm_runtime_install_policy.py`
- `tests/scripts/test_run_ensemble_diagnostics_security.py`
- `tests/scripts/test_institutional_unattended_contract.py`
- `tests/scripts/test_institutional_unattended_gate.py`
- `tests/utils/test_openclaw_cli.py` (default prompt-injection block)
- `tests/etl/test_checkpoint_manager.py` (checkpoint ID/unpickling safety)

Recommended CI lane:

```powershell
python -m pytest -m "not gpu and not slow"
```

