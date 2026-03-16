# OpenClaw Implementation Policy

Status: Mandatory  
Scope: repo-wide OpenClaw runtime + maintenance implementation

## Purpose

Prevent implementation drift that reintroduces:

- host/sandbox/ACP configuration mismatches,
- maintenance-path bypasses,
- false-green runtime health status.

This policy is enforced by anti-regression tests and must be treated as a
code contract, not a guideline.

## Non-Negotiable Contracts

1. Exec host allowlist is fixed:
- `tools.exec.host` must be one of `sandbox`, `gateway`, `node`.

2. Sandbox host compatibility is fixed:
- when `tools.exec.host=sandbox`, `agents.defaults.sandbox.mode` must be
  `non-main` or `all`.

3. ACP default agent is required when ACP is configured:
- `acp.defaultAgent` must be present and resolve to an agent id.

4. Pre-maintenance enforcement is mandatory:
- `scripts/run_openclaw_maintenance.ps1` must run
  `scripts/enforce_openclaw_exec_environment.py` before maintenance on both
  Windows and WSL branches.
- `scripts/start_openclaw_guardian.ps1` must run the same enforcement before
  launching long-running maintenance watch mode.

5. Runtime health signal contract is mandatory:
- `scripts/project_runtime_status.py` must expose `openclaw_exec_env` checks
  with explicit signals:
  - `invalid_exec_host`
  - `invalid_sandbox_mode`
  - `missing_acp_default_agent`
  - `exec_env_valid`

6. Constant parity is mandatory across OpenClaw validators:
- `VALID_EXEC_HOSTS` and sandbox-host mode allowlists must remain aligned in:
  - `scripts/enforce_openclaw_exec_environment.py`
  - `scripts/project_runtime_status.py`
  - `scripts/verify_openclaw_config.py`

7. Notification storm suppression is mandatory:
- `utils/openclaw_cli.py` must keep persistent cross-process storm protection
  enabled by default via `OPENCLAW_STORM_GUARD_ENABLED`.
- Repeated retryable transport failures must trigger adaptive cooldown using:
  - `OPENCLAW_STORM_BASE_COOLDOWN_SECONDS`
  - `OPENCLAW_STORM_MAX_COOLDOWN_SECONDS`
  - `OPENCLAW_STORM_BACKOFF_MULTIPLIER`
  - `OPENCLAW_STORM_RESET_WINDOW_SECONDS`
- Storm state must be persisted in the same guard state file so suppression
  survives process restarts.
- A successful send must clear prior storm state for that target.

8. Local-first Ollama agent configuration is mandatory:
- `scripts/openclaw_models.py` must write native Ollama provider settings
  (`api=ollama`) and must not route native Ollama through legacy `/v1`
  OpenAI-compatible paths.
- Preferred OpenClaw primary is the first tool-capable local qwen model in
  `OPENCLAW_OLLAMA_MODEL_ORDER` (`qwen3.5:27b` preferred,
  `qwen3:8b` compatible fallback).
- Non-tool delegate models such as `deepseek-r1:*` may remain available to PMX
  reasoning paths, but must not be inserted into the OpenClaw agent fallback
  chain.

## Anti-Regression Evidence

Run at minimum:

```bash
python -m pytest tests/scripts/test_openclaw_implementation_contract.py -q
python -m pytest tests/scripts/test_enforce_openclaw_exec_environment.py tests/scripts/test_project_runtime_status.py tests/scripts/test_run_openclaw_maintenance_wrapper.py -q
python -m pytest tests/scripts/test_openclaw_models.py tests/scripts/test_openclaw_production_readiness.py -q
python -m pytest tests/utils/test_openclaw_cli.py -k storm_guard -q
```

If any of these fail, OpenClaw implementation policy is considered broken and
must be fixed before merge.
