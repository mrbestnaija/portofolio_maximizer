#!/usr/bin/env bash
# Commit + push helper for remote-first workflows (see Documentation/GIT_WORKFLOW.md).
#
# Usage:
#   bash/git_sync.sh "commit message" [branch]
#   bash/git_sync.sh                  # auto message, current branch
#
# Behavior:
# - Refuses to push directly to master unless PMX_ALLOW_DIRECT_MASTER_PUSH=1.
# - If there are working-tree changes: stages all, commits, then rebases feature branch onto origin/master.
# - If there are no working-tree changes: still rebases feature branch onto origin/master and pushes if ahead.
# - Never persists tokens into remotes; supports ephemeral GIT_ASKPASS when GIT_USE_ENV_REMOTE=1.

set -euo pipefail

ASKPASS_FILE=""

cleanup() {
  if [[ -n "${ASKPASS_FILE}" && -f "${ASKPASS_FILE}" ]]; then
    rm -f "${ASKPASS_FILE}"
  fi
}

trap cleanup EXIT

pmx_load_env_file() {
  local env_path="${1:-.env}"
  [[ -f "${env_path}" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    line="${line#$'\xEF\xBB\xBF'}"
    [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue

    if [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      local key="${BASH_REMATCH[1]}"
      local value="${BASH_REMATCH[2]}"

      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"

      if [[ "${value}" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value}" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      fi

      export "${key}=${value}"
    fi
  done < "${env_path}"
}

pmx_setup_git_askpass() {
  if [[ "${GIT_USE_ENV_REMOTE:-0}" != "1" ]]; then
    return 0
  fi

  if [[ -z "${GitHub_Username:-}" || -z "${GitHub_TOKEN:-}" || -z "${GitHub_Repo:-}" || "${GitHub_TOKEN}" =~ xxx ]]; then
    echo "GIT_USE_ENV_REMOTE=1 set but GitHub creds missing/placeholder; skipping askpass auth." >&2
    return 0
  fi

  umask 077
  ASKPASS_FILE="$(mktemp -t pmx_git_askpass.XXXXXX)"
  cat > "${ASKPASS_FILE}" <<'EOF'
#!/usr/bin/env bash
prompt="${1:-}"
case "${prompt}" in
  *Username*|*username*) printf '%s\n' "${GitHub_Username:-}" ;;
  *Password*|*password*) printf '%s\n' "${GitHub_TOKEN:-}" ;;
  *) printf '%s\n' "${GitHub_TOKEN:-}" ;;
esac
EOF
  chmod 700 "${ASKPASS_FILE}"

  export GIT_ASKPASS="${ASKPASS_FILE}"
  export GIT_TERMINAL_PROMPT=0

  if [[ "${PMX_GIT_SET_REMOTE_FROM_ENV:-0}" == "1" ]]; then
    git remote set-url origin "https://github.com/${GitHub_Username}/${GitHub_Repo}.git" >/dev/null 2>&1 || true
  fi
}

pmx_timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

pmx_abort_if_staging_secrets() {
  local staged
  staged="$(git diff --cached --name-only || true)"
  if echo "${staged}" | grep -E -q '(^|/)\.env$|(^|/)scripts/\.env$|(^|/)secrets/|\.pem$|\.key$'; then
    git reset -q -- .env scripts/.env 2>/dev/null || true
    echo "Refusing to commit staged secrets (.env/scripts/.env/secrets/*/*.key/*.pem). Review 'git diff --cached'." >&2
    exit 1
  fi
}

# Optional: load credentials and repo hints from .env (never commit .env).
pmx_load_env_file ".env"

# Optional: set git identity from env
if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config user.name "$GIT_USER_NAME"
fi
if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

pmx_setup_git_askpass

REMOTE="${PMX_GIT_REMOTE:-origin}"
BASE_BRANCH="${PMX_BASE_BRANCH:-master}"
ALLOW_MASTER_PUSH="${PMX_ALLOW_DIRECT_MASTER_PUSH:-0}"
FORCE_WITH_LEASE="${PMX_FORCE_WITH_LEASE:-0}"

BRANCH_DEFAULT="$(git rev-parse --abbrev-ref HEAD)"
COMMIT_MSG="${1:-}"
BRANCH="${2:-${BRANCH_DEFAULT}}"

if [[ -z "${BRANCH}" ]]; then
  echo "Unable to determine branch. Pass explicitly: bash/git_sync.sh \"msg\" my-branch" >&2
  exit 1
fi

if [[ "${BRANCH}" == "master" && "${ALLOW_MASTER_PUSH}" != "1" ]]; then
  echo "Refusing to push directly to master (remote-first policy). Create a feature branch + PR. Set PMX_ALLOW_DIRECT_MASTER_PUSH=1 to override." >&2
  exit 2
fi

if [[ -z "${COMMIT_MSG}" ]]; then
  COMMIT_MSG="chore: sync ${BRANCH} @ $(pmx_timestamp)"
fi

git checkout "${BRANCH}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Staging changes..."
  git add -A
  pmx_abort_if_staging_secrets

  echo "Committing..."
  git commit -m "${COMMIT_MSG}"
fi

echo "Fetching from ${REMOTE}..."
git fetch --prune "${REMOTE}"

if [[ "${BRANCH}" == "master" ]]; then
  echo "Updating local master (fast-forward only)..."
  git pull --ff-only "${REMOTE}" master
  echo "Done."
  exit 0
fi

echo "Rebasing ${BRANCH} onto ${REMOTE}/${BASE_BRANCH}..."
git rebase "${REMOTE}/${BASE_BRANCH}"

echo "Pushing to ${REMOTE}/${BRANCH}..."
if git push -u "${REMOTE}" "${BRANCH}"; then
  echo "Sync complete for ${BRANCH}"
  exit 0
fi

if [[ "${FORCE_WITH_LEASE}" == "1" ]]; then
  echo "Push rejected; retrying with --force-with-lease (feature branches only)..."
  git push --force-with-lease -u "${REMOTE}" "${BRANCH}"
  echo "Sync complete for ${BRANCH}"
  exit 0
fi

echo "Push rejected. If you rewrote history on this feature branch, rerun with PMX_FORCE_WITH_LEASE=1 (never for master)." >&2
exit 3
