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

  # Never persist tokens into git remotes. Use an ephemeral askpass helper instead.
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

  git remote set-url origin "https://github.com/${GitHub_Username}/${GitHub_Repo}.git" >/dev/null 2>&1 || true
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
