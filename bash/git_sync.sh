#!/usr/bin/env bash
# Commit + push helper (master-first) following Documentation/GIT_WORKFLOW.md
#
# Usage:
#   bash/git_sync.sh "commit message" [branch]
#   bash/git_sync.sh                  # auto message, current branch
#
# Behavior:
# - Adds all changes, commits with the provided message (or an auto-generated one),
#   rebases on origin/<branch>, then pushes to origin/<branch>.
# - Defaults to the current branch; pass a branch explicitly to target another.

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

BRANCH_DEFAULT="$(git rev-parse --abbrev-ref HEAD)"
COMMIT_MSG="${1:-}"
BRANCH="${2:-${BRANCH_DEFAULT}}"

if [[ -z "${BRANCH}" ]]; then
  echo "Unable to determine branch. Pass explicitly: bash/git_sync.sh \"msg\" my-branch" >&2
  exit 1
fi

if [[ -z "${COMMIT_MSG}" ]]; then
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  COMMIT_MSG="chore: sync ${BRANCH} @ ${ts}"
fi

git checkout "${BRANCH}"

# Ensure there is something to commit
if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit; ${BRANCH} is clean."
  exit 0
fi

echo "Staging changes..."
git add -A

echo "Committing..."
git commit -m "${COMMIT_MSG}"

echo "Fetching and rebasing onto origin/${BRANCH}..."
git fetch origin "${BRANCH}"
git rebase origin/"${BRANCH}"

echo "Pushing to origin/${BRANCH}..."
git push origin "${BRANCH}"

echo "Sync complete for ${BRANCH}"
