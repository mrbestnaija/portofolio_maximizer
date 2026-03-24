# OpenClaw Discord Token Audit 2026-03-24

## Scope

This note records the March 24, 2026 audit of PMX Discord channel readiness for the OpenClaw `ops` agent.

The goal was to verify whether Discord was fully callable end to end after:

- OpenClaw model/runtime sync
- plugin enablement
- channel account re-add
- gateway restart

No secret values are recorded here. Only presence, absence, and runtime behavior are documented.

## Sanitized `.env` Findings

The local `.env` file contains the Discord interactions app variables needed for interaction-mode flows:

- `DISCORD_APP_NAME` present at line 261
- `DISCORD_APPLICATION_ID` present at line 262
- `DISCORD_PUBLIC_KEY` present at line 263
- `INTERACTIONS_API_KEY` present at line 264
- `DISCORD_APP_INSTALL_LINK` present at line 265

The local `.env` file does **not** contain either Discord bot token key required for OpenClaw channel messaging:

- `DISCORD_BOT_TOKEN` missing
- `DISCORD_TOKEN` missing

## Runtime Evidence

Commands run during the audit:

```powershell
python scripts/validate_credentials.py
python scripts/llm_multi_model_orchestrator.py sync
python scripts/openclaw_env.py plugins enable discord
python scripts/openclaw_env.py channels add --channel discord --use-env
openclaw gateway restart
openclaw channels status --json
openclaw channels logs --channel discord --lines 80
openclaw status --deep
```

Observed results:

- `python scripts/validate_credentials.py` reported:
  - Discord interactions app configured
  - Discord bot token missing
- `openclaw status --deep` reported:
  - `Discord WARN failed (401) - getMe failed (401)`
- `openclaw channels logs --channel discord --lines 80` reported repeated:
  - `Failed to resolve Discord application id`

## Interpretation

The OpenClaw runtime is wired correctly enough to attempt startup, but Discord still fails before the provider can remain online.

The decisive blocker is credentials, not routing:

- PMX has Discord interactions app material
- PMX does not currently have a valid Discord bot token in `.env`
- OpenClaw Discord channel messaging requires a bot token, not just interactions app configuration

This matches the existing PMX guidance in:

- `Documentation/OPENCLAW_INTEGRATION.md`
- `scripts/openclaw_notify.py`

## Why The Channel Still Fails

Interaction-mode credentials and channel-bot credentials are not the same thing.

In the current PMX state:

- Discord app metadata is present
- OpenClaw Discord accounts are configured
- Gateway restart succeeds
- Discord provider startup reaches the authentication step
- Discord API validation still fails with `401`

That means Discord cannot be considered callable end to end yet.

## Required Remediation

1. Add a valid `DISCORD_BOT_TOKEN` or `DISCORD_TOKEN` to `.env`.
2. Re-run:

```powershell
python scripts/openclaw_env.py channels add --channel discord --use-env
openclaw gateway restart
openclaw status --deep
```

3. Do not treat Discord as healthy until `openclaw status --deep` shows `Discord OK`.
4. After Discord reaches `OK`, run a real outbound send test:

```powershell
python scripts/openclaw_notify.py --channel discord --to channel:<channel_id> --message "PMX Discord test"
```

## Current Verdict

As of March 24, 2026:

- WhatsApp is healthy and callable
- Discord is configured but not operational
- The missing/invalid Discord bot token is the blocking issue
