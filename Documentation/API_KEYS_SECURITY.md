# API Keys & Credentials Security Guide

**Status**: ACTIVE  
**Last Updated**: 2026-01-04  

This project treats credentials as **local-only secrets**. They must never be committed, printed, or embedded in git remotes.

## What Is Protected

Credentials include (non-exhaustive):
- Market data provider keys (Alpha Vantage, Finnhub, etc.)
- Brokerage credentials (cTrader, XTB, etc.)
- GitHub automation tokens (Projects PATs)

## Source Of Truth (Implementation)

Credentials are resolved through:
- `etl/secret_loader.py` (canonical secret loading logic)
- `etl/data_source_manager.py` and extractors (consume secrets via `load_secret`)
- `execution/ctrader_client.py` (cTrader env/.env parsing + supports `*_FILE` paths)
- Git helper scripts: `bash/git_sync.sh`, `bash/git_syn_to_local.sh` (support `.env`, but never persist tokens into remotes)

## Loading Order (Deterministic)

For any secret named `KEY`:
1. `KEY_FILE` (Docker secret / local secret file path) — first non-empty, non-comment line
2. `KEY` from environment
3. Missing → feature should degrade gracefully or fail with a clear missing-key message

## Local Development: `.env` (Git-Ignored)

- `.env` is ignored by git via `.gitignore`.
- `.env.template` is the committed reference file with variable names and placeholders.
- `etl/secret_loader.py` bootstraps `.env` (best-effort) without printing values and without overwriting already-set environment variables.

## Production / Docker: `*_FILE` Secrets

Prefer file-based secrets:
- Put secret values into files (e.g., `secrets/alpha_vantage_api_key.txt`)
- Provide the path via `KEY_FILE` (e.g., `ALPHA_VANTAGE_API_KEY_FILE=/run/secrets/alpha_vantage_api_key`)
- `etl/secret_loader.py` reads these files and never logs secret values

Use the setup helpers:
- Windows: `scripts/setup_secrets.ps1`
- Linux/Mac: `scripts/setup_secrets.sh`

## Using Secrets In Code (Approved Pattern)

Use `etl.secret_loader.load_secret()` everywhere. Do not call `load_dotenv()` across random modules and do not print secret values.

```python
from etl.secret_loader import load_secret

alpha_key = load_secret("ALPHA_VANTAGE_API_KEY")  # auto-detects ALPHA_VANTAGE_API_KEY_FILE too
```

## GitHub Authentication (Local Only)

Preferred (interactive):
- `gh auth login`
- `gh auth setup-git`

If you must use a PAT locally for non-interactive git operations:
- Put it in `.env` as `GitHub_TOKEN`, with `GitHub_Username` and `GitHub_Repo`
- Set `GIT_USE_ENV_REMOTE=1`
- Use `bash/git_sync.sh` or `bash/git_syn_to_local.sh`

Security note:
- These scripts use an **ephemeral `GIT_ASKPASS` helper**.
- They do **not** embed tokens into `git remote` URLs and do **not** persist secrets into `.git/config`.

## Never Do This

- Hardcode secrets in code or config files
- Print tokens/keys/passwords to logs or stdout
- Embed tokens in remote URLs like `https://user:token@github.com/...`
- Commit `.env` or `secrets/` files

## GitHub Actions (CI) Secret Handling

### Required behaviors
- Do not write repository secrets into `.env` files inside workflows unless absolutely necessary; prefer passing secrets via action inputs or environment variables scoped to the step.
- Workflows that require secrets (e.g., Projects automation tokens) must **skip** cleanly when secrets are unavailable (common for forked PRs).
- Avoid printing secret values; always rely on GitHub masking and never echo tokens/keys.

### Projects automation token
- If you use GitHub Projects (v2) automation, store a PAT as a repository secret (e.g., `PROJECTS_TOKEN`) with the minimum required scopes.
- Treat that automation as **nice-to-have**: it must not block merges if the token is absent or permissions are restricted.

## Verification Checklist (Before Any Push)

- `.env` is ignored:
  - `git check-ignore -v .env`
- `scripts/.env` is ignored:
  - `git check-ignore -v scripts/.env`
- `.env` is not tracked:
  - `git ls-files -- .env`
- `scripts/.env` is not tracked:
  - `git ls-files -- scripts/.env`
- No staged secrets:
  - `git diff --cached`
  - `rg -n "(api[_ -]?key|token|secret|password)\\s*=" -S .`

## Incident Response (If A Secret Leaks)

1. Revoke/rotate the credential at the provider immediately
2. Remove the secret from git history if it was committed
3. Update `.env` / your secret files with the rotated credential
4. Re-run security tests: `pytest -m security -v`

