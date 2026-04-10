# Security Policy

This project is actively hardened for unattended autonomous operation. Security reports are handled privately and with priority.

## Supported Versions

Security fixes are issued for the currently active branch/release line only.

| Version / Branch | Supported | Notes |
| --- | --- | --- |
| `master` (latest) | :white_check_mark: | Full security updates and hardening changes |
| Latest prior release line (N-1) | :white_check_mark: | Critical fixes on a best-effort basis |
| Older release lines / stale forks | :x: | Upgrade required before support |

If you are running an older commit/tag, upgrade to latest `master` before reporting issues that may already be fixed.

## Reporting a Vulnerability

Do **not** post security issues publicly.

### Preferred channel (private)

Use GitHub Private Vulnerability Reporting:

`https://github.com/example-org/portofolio_maximizer/security/advisories/new`

### If private advisory submission is unavailable

Open a minimal public issue titled `Security contact request` with **no exploit details**. A maintainer will move the discussion to a private channel.

### What to include

- Impact summary (confidentiality, integrity, availability)
- Reproduction steps and affected paths/files
- Environment details (OS, Python version, commit hash)
- Proof-of-concept or logs (redacted; never include secrets)
- Suggested mitigations, if known

## Response Targets

- Acknowledgement: within 24 hours for critical reports, within 72 hours for others
- Initial triage/severity decision: within 5 business days
- Ongoing updates: at least weekly until resolution/disclosure decision

## Disclosure Process

When a report is accepted, we will:

1. Validate and classify severity.
2. Prepare and test a fix.
3. Publish remediation details and advisory notes.
4. Credit the reporter if requested.

We may request coordinated disclosure until users have a patch path.

## Secrets and Safe Testing

- Never post `.env` contents, API keys, tokens, passwords, or broker credentials.
- Follow `Documentation/API_KEYS_SECURITY.md` for local secret handling.
- Use good-faith testing only: no service disruption, no unauthorized access, no data destruction.
