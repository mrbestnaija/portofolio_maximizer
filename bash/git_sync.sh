#!/usr/bin/env bash
# Lightweight Git sync helper following Documentation/GIT_WORKFLOW.md
#
# Usage:
#   bash/git_sync.sh                 # pull --rebase and push current branch
#   bash/git_sync.sh feature/branch  # sync specific branch
#
# Safety:
# - Aborts if the worktree is dirty (uncommitted changes).
# - Uses pull --rebase to keep history linear.

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"

if [[ -z "$BRANCH" ]]; then
  echo "Unable to determine branch. Pass explicitly: bash/git_sync.sh my-branch" >&2
  exit 1
fi

# Ensure clean worktree
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Worktree is dirty. Commit or stash changes before syncing." >&2
  exit 1
fi

echo "Syncing branch: $BRANCH"

# Fetch and rebase
git fetch origin "$BRANCH" || {
  echo "Fetch failed for origin/$BRANCH" >&2
  exit 1
}

git checkout "$BRANCH"
git pull --rebase origin "$BRANCH"

# Push
git push origin "$BRANCH"

echo "Sync complete for $BRANCH"
