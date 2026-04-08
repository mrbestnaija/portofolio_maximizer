# Documentation History

This directory stores archived markdown that is useful for historical traceability
but not part of the active operator runbook set.

## Layout

- `status/`: dated status snapshots and migration notes.
- `sessions/`: dated session notes.
- `run_logs/`: historical pipeline run logs.

## Safety Rule

When moving markdown into history:

1. Update inbound links directly to the archived `history/` path whenever possible.
2. Update `Documentation/DOCUMENTATION_INDEX.md`.
3. Keep a temporary compatibility stub only if some references cannot be updated in the same change.
4. Verify links still resolve.
