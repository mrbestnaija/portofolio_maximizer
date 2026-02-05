# Security Policy

We take security seriously. If you find a vulnerability, please report it responsibly so we can fix it quickly and protect users.

## Supported Versions

Security fixes are provided for the most recent stable release line.

| Version | Supported |
| ------: | :-------: |
| Latest stable (main) | ✅ |
| Previous minor release | ✅ |
| Older releases | ❌ |

If you’re running an unsupported version, please upgrade before reporting issues that may already be resolved in newer releases.

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security reports.

## Secrets & Credentials

- This project treats credentials as **local-only secrets**; see `Documentation/API_KEYS_SECURITY.md`.
- Never commit or print secrets (e.g., `.env`, `scripts/.env`, tokens, broker creds).

### Primary reporting channel (preferred)
Use **GitHub Security Advisories** for all vulnerability reports:

**Repository → Security → Advisories → Report a vulnerability**

This is the fastest way for us to triage and track fixes privately.

### Email (for core / urgent issues)
If the issue is **core**, **actively exploitable**, or you cannot use GitHub Advisories, email us:

**security@yourdomain.com**
(Replace with the correct address for this project.)

### What to include
To help us act quickly, include:
- A clear description of the issue and impact
- Steps to reproduce (a safe proof-of-concept helps)
- Affected versions, commit hash, OS, and runtime details
- Any mitigations, workarounds, or suggested fixes (if available)

## Response Timeline

We keep timelines flexible, but we do move with urgency.

### Core / critical vulnerabilities
- **Acknowledgement:** within **24 hours**
- **Priority triage:** ASAP after acknowledgement
- **Updates:** provided as we confirm scope, severity, and fix path

### All other reports
- **Acknowledgement:** typically within **48 hours**
- **Initial assessment:** usually within **5 business days**
- **Fix plan:** shared once severity is confirmed and a patch approach is clear

## Coordinated Disclosure

If the report is valid, we will:
- Confirm severity and affected versions
- Prepare a patch and release notes
- Publish a GitHub Security Advisory (crediting you if you want)

We may request that details remain private until a fix is available. Disclosure windows vary by severity and exploitability.

## Dependabot Alerts

As of 2026-02-01, Dependabot reports CVE-2026-0994 in protobuf with no fixed release available. We are monitoring upstream and will upgrade once a fixed version is published; until then, keep the alert open (do not dismiss as fixed) and document any mitigations here.

## Safe Harbor

We support good-faith security research. Please:
- Avoid privacy violations, data destruction, and service disruption
- Do not access or modify data you do not own
- Only test against systems you are authorized to test

Thank you for helping keep this project secure.
