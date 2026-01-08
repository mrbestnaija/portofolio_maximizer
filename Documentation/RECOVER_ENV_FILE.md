# Recovering Your `.env` File (Local Credentials)

**Status**: ACTIVE  
**Last Updated**: 2026-01-04  

This project uses `.env` for **local-only** credentials. The file must remain **git-ignored** and must never be pasted into tickets, chats, or committed.

## 1) Recreate The File Safely

From the repo root:

```bash
cp .env.template .env
```

Then edit `.env` locally and fill in real values.

## 2) Prefer `*_FILE` Secrets When Possible

For Docker/production or for improved local hygiene:
- Put a secret into a file (1 line, no quotes; comments allowed with `#`)
- Set `KEY_FILE=/path/to/secret_file`

Helpers:
- Windows: `scripts/setup_secrets.ps1`
- Linux/Mac: `scripts/setup_secrets.sh`

## 3) Where To Get Values

Choose the safest option:
- Password manager (recommended)
- Provider dashboard (regenerate if unsure)
- OS environment variables (only if you already keep them there)

## 4) Minimum Variables You May Need

Use `.env.template` as the authoritative list. Common keys:

**Market data**
- `ALPHA_VANTAGE_API_KEY` (or `ALPHA_VANTAGE_API_KEY_FILE`)
- `FINNHUB_API_KEY` (or `FINNHUB_API_KEY_FILE`)

**Brokerage / execution (optional; required for live)**
- cTrader: `USERNAME_CTRADER`/`EMAIL_CTRADER`, `PASSWORD_CTRADER`, `APPLICATION_NAME_CTRADER` (and optionally `CTRADER_APPLICATION_SECRET`)
- XTB: `XTB_USERNAME`, `XTB_PASSWORD`, `XTB_SERVER`

**GitHub automation (optional)**
- `PROJECTS_TOKEN` (for GitHub Projects workflow)

## 5) Verify You Didn’t Expose Anything

`.env` must be ignored:
```bash
git check-ignore -v .env
```

`.env` must not be tracked:
```bash
git ls-files -- .env
```

Search for accidental assignments in tracked files:
```bash
rg -n "(api[_ -]?key|token|secret|password)\\s*=" -S .
```

## 6) If A Secret Might Be Compromised

1. Rotate/revoke at the provider immediately
2. Replace the value in `.env` / secret file
3. Re-run security tests: `pytest -m security -v`

