---
name: pmx-inbox
description: Gmail + Proton inbox workflows for Portfolio Maximizer (local IMAP/SMTP), with safe defaults and optional OpenClaw notifications.
metadata: { "openclaw": { "requires": { "anyBins": ["python", "python3"] } } }
---

# Portfolio Maximizer Inbox Workflows

Use this skill to scan inboxes and (optionally) send emails using local credentials.

## Guardrails

- Never paste `.env` contents or secret values into chat/logs.
- Default to read-only inbox scans.
- Do not send emails unless explicitly enabled (PMX_INBOX_ALLOW_SEND=1 or config.limits.allow_send=true).
- Keep outputs short; do not dump entire emails unless the user explicitly requests it.

## Commands

List configured accounts:

- `python scripts/inbox_workflow.py list-accounts`

Scan unread messages (all enabled accounts):

- `python scripts/inbox_workflow.py scan`

Scan a specific account:

- `python scripts/inbox_workflow.py scan --account gmail`

Fetch a full message (writes an `.eml` file):

- `python scripts/inbox_workflow.py fetch --account gmail --uid <uid>`

Send email (disabled by default):

- `PMX_INBOX_ALLOW_SEND=1 python scripts/inbox_workflow.py send --account gmail --to "recipient@example.com" --subject "..." --body "..."`

## Configuration

- `config/inbox_workflows.yml`
- `Documentation/INBOX_WORKFLOWS.md`
