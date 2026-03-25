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
- exec-capable agent entries in `agents.list` must not override sandbox back to
  `off` or `main`, because that disables the sandbox runtime for embedded
  WhatsApp/cron sessions even when defaults are valid.
- when `tools.exec.host=sandbox`, the Docker-backed sandbox runtime must be
  available on the host; otherwise enforcement must fall back to `node` or
  `gateway` instead of leaving the config in a known-bad state.

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
  - `sandbox_runtime_unavailable`
  - `missing_acp_default_agent`
  - `exec_env_valid`

6. Constant parity is mandatory across OpenClaw validators:
- `VALID_EXEC_HOSTS` and sandbox-host mode allowlists must remain aligned in:
  - `scripts/enforce_openclaw_exec_environment.py`
  - `scripts/project_runtime_status.py`
  - `scripts/verify_openclaw_config.py`

7. OpenClaw config readers must tolerate UTF-8 BOM:
- Scripts that read `~/.openclaw/openclaw.json` must accept BOM-prefixed JSON
  (`utf-8-sig` or equivalent), because PowerShell/manual edits can introduce a
  BOM even when the JSON content is otherwise valid.

8. Cron-agent tool policy should avoid noisy profile expansion:
- `trading` and `training` should prefer explicit `tools.allow` / `tools.deny`
  policy over the built-in `coding` profile on affected OpenClaw builds, so
  embedded runs do not emit false `apply_patch` unknown-entry warnings.

9. Notification storm suppression is mandatory:
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

## Anti-Regression Evidence

Run at minimum:

```bash
python -m pytest tests/scripts/test_openclaw_implementation_contract.py -q
python -m pytest tests/scripts/test_enforce_openclaw_exec_environment.py tests/scripts/test_project_runtime_status.py tests/scripts/test_run_openclaw_maintenance_wrapper.py -q
python -m pytest tests/utils/test_openclaw_cli.py -k storm_guard -q
```

If any of these fail, OpenClaw implementation policy is considered broken and
must be fixed before merge.
