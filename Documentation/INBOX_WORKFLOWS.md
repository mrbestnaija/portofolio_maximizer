# Inbox Workflows (Gmail + Proton Mail)

This repo supports local inbox workflows (read and optional send) so you can run higher-level automation without exposing credentials outside your machine.

The implementation is in:

- `scripts/inbox_workflow.py`
- `config/inbox_workflows.yml`
- `etl/secret_loader.py` (secrets loader, supports `*_FILE`)

## Security Model (Practical)

- Credentials stay local in `.env` or `*_FILE` paths. `.env` is git-ignored.
- Scans are read-only by default (no message state changes).
- Sending is disabled by default. Enable explicitly:
  - set `PMX_INBOX_ALLOW_SEND=1`, or
  - set `limits.allow_send: true` in `config/inbox_workflows.yml`
- You can also disable read/send per account via:
  - `config/inbox_workflows.yml` -> `accounts.<name>.capabilities.read/send`

## Gmail (Recommended: App Password + IMAP)

Gmail OAuth is more secure but adds setup complexity. The default path here is IMAP/SMTP using a Gmail App Password.

Checklist:

1. Enable IMAP in your Gmail settings.
2. Enable 2FA on your Google account (required for App Passwords).
3. Create an App Password for "Mail".
4. Put credentials in `.env` (or use `*_FILE`):
   - `PMX_EMAIL_USERNAME=your@gmail.com`
   - `PMX_EMAIL_PASSWORD=your_app_password_here`

Gmail defaults are already configured in `config/inbox_workflows.yml` under the `gmail` account.

## Proton Mail (Bridge)

Proton Mail does not provide direct IMAP/SMTP. The standard approach is Proton Mail Bridge, which runs locally and exposes IMAP/SMTP endpoints.

Steps:

1. Install Proton Mail Bridge and sign in.
2. In Bridge, copy the IMAP/SMTP host/port settings.
3. Add Bridge credentials to `.env`:
   - `PMX_PROTON_BRIDGE_USERNAME=...` (Bridge-generated username)
   - `PMX_PROTON_BRIDGE_PASSWORD=...` (Bridge-generated password)
   - `PMX_PROTON_FROM=you@proton.me` (optional; defaults to username if omitted)
4. Enable the account in `config/inbox_workflows.yml`:
   - set `accounts.proton_bridge.enabled: true`
   - update host/ports/tls to match Bridge settings

## CLI Usage

List accounts:

```powershell
python scripts/inbox_workflow.py list-accounts
```

Scan unread mail (all enabled accounts):

```powershell
python scripts/inbox_workflow.py scan
```

Scan a single account:

```powershell
python scripts/inbox_workflow.py scan --account gmail
```

Fetch a full message (writes an `.eml` file):

```powershell
python scripts/inbox_workflow.py fetch --account gmail --uid 12345
```

Send (disabled by default):

```powershell
$env:PMX_INBOX_ALLOW_SEND = "1"
python scripts/inbox_workflow.py send --account gmail --to "recipient@example.com" --subject "Hello" --body "Test"
```

## OpenClaw Notifications (Optional)

If `OPENCLAW_TO` is configured, `scan` will auto-send a short scan summary via OpenClaw unless disabled:

- Disable for all scripts: `PMX_NOTIFY_OPENCLAW=0`
- Disable for inbox scans only: `PMX_INBOX_NOTIFY_OPENCLAW=0`

See:

- `Documentation/OPENCLAW_INTEGRATION.md`
