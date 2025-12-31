#!/usr/bin/env bash
# Lightweight Git sync helper following Documentation/GIT_WORKFLOW.md
#
# Usage:
#   bash/git_sync.sh                 # pull --rebase and push current branch
#   bash/git_sync.sh feature/branch  # sync specific branch
#
# Safety:
# - Auto-stashes any dirty worktree before syncing (including untracked files); restores on the original branch.
# - Uses pull --rebase to keep history linear.

set -euo pipefail

# Optional: load credentials and repo hints from .env (never commit .env).
# Strips BOM, comments, blanks, and CRLF.
if [[ -f ".env" ]]; then
  set -a
  source <(
    cat .env \
    | tr -d '\r' \
    | sed $'1s/^\xEF\xBB\xBF//' \
    | grep -Ev '^[[:space:]]*($|#)'
  )
  set +a
fi

# Optional: set git identity from env
if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config user.name "$GIT_USER_NAME"
fi
if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

# Optional: update origin URL with token if explicitly enabled (HTTPS only)
if [[ "${GIT_USE_ENV_REMOTE:-0}" == "1" ]]; then
  if [[ -n "${GitHub_Username:-}" && -n "${GitHub_TOKEN:-}" && -n "${GitHub_Repo:-}" && ! "${GitHub_TOKEN}" =~ xxx ]]; then
    git remote set-url origin "https://${GitHub_Username}:${GitHub_TOKEN}@github.com/${GitHub_Username}/${GitHub_Repo}.git"
  else
    echo "GIT_USE_ENV_REMOTE=1 set but GitHub creds missing/placeholder; skipping remote override." >&2
  fi
fi

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
STASHED=0
RESTORED=0

restore_stash() {
  if [[ $STASHED -eq 1 && $RESTORED -eq 0 ]]; then
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "$START_BRANCH" ]]; then
      echo "Stash created on $START_BRANCH. Skipping auto-restore to avoid applying changes onto $current_branch." >&2
      echo "Recover manually: git checkout $START_BRANCH && git stash pop" >&2
      return
    fi

    echo "Restoring stashed changes..."
    if git stash pop --quiet; then
      echo "Restored stashed changes."
    else
      echo "Failed to reapply stash. Your changes remain stashed. Run 'git stash pop' manually." >&2
    fi
    RESTORED=1
  fi
}

trap restore_stash EXIT

if [[ -z "$BRANCH" ]]; then
  echo "Unable to determine branch. Pass explicitly: bash/git_sync.sh my-branch" >&2
  exit 1
fi

# Auto-clean dirty worktree by stashing everything (including untracked)
if [[ -n "$(git status --porcelain)" ]]; then
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "Worktree is dirty. Auto-stashing changes before sync ($ts)..."
  git stash push -u -m "git_sync autostash $ts" >/dev/null
  STASHED=1
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

# If we stashed, bring the changes back before exiting
restore_stash
trap - EXIT

echo "Sync complete for $BRANCH"
