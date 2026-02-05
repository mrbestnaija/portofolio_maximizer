#!/usr/bin/env bash
# Safe sync helper (remote -> local) for remote-first workflows.
#
# Usage:
#   bash/git_syn_to_local.sh               # sync master from origin
#   bash/git_syn_to_local.sh branch-name   # sync specific branch from origin
#
# Safety:
# - Auto-stashes a dirty worktree (unless PMX_NO_STASH=1)
# - Fast-forwards only (never rebases or merges)
# - Never pushes
# - Refuses to proceed if the local branch is ahead/diverged unless PMX_HARD_RESET=1

set -euo pipefail

ASKPASS_FILE=""

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

cleanup_askpass() {
  if [[ -n "${ASKPASS_FILE}" && -f "${ASKPASS_FILE}" ]]; then
    rm -f "${ASKPASS_FILE}"
  fi
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
  date -u +"%Y%m%dT%H%M%SZ"
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
BRANCH="${1:-master}"
HARD_RESET="${PMX_HARD_RESET:-0}"
NO_STASH="${PMX_NO_STASH:-0}"
START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STASHED=0
RESTORED=0
CONFLICT_DETECTED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

restore_stash() {
  if [[ ${STASHED} -eq 1 && ${RESTORED} -eq 0 ]]; then
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"

    if [[ "${current_branch}" != "${START_BRANCH}" ]]; then
      echo -e "${YELLOW}Stash created on ${START_BRANCH}. Skipping auto-restore to avoid applying changes onto ${current_branch}.${NC}"
      echo -e "${YELLOW}Recover manually: git checkout ${START_BRANCH} && git stash pop${NC}"
      return
    fi

    if [[ ${CONFLICT_DETECTED} -eq 1 ]]; then
      echo -e "${YELLOW}Sync required manual action. Stash not restored to avoid conflict propagation.${NC}"
      echo -e "${YELLOW}Restore manually when ready: git stash pop${NC}"
      return
    fi

    echo "Restoring stashed changes..."
    if git stash pop --quiet; then
      echo -e "${GREEN}✓ Restored stashed changes.${NC}"
    else
      echo -e "${RED}Failed to reapply stash. Your changes remain stashed. Run 'git stash pop' manually.${NC}"
    fi
    RESTORED=1
  fi
}

cleanup_on_exit() {
  cleanup_askpass
  restore_stash
}

trap cleanup_on_exit EXIT

if [[ -z "${BRANCH}" ]]; then
  echo -e "${RED}Unable to determine branch. Pass explicitly: bash/git_syn_to_local.sh my-branch${NC}"
  exit 1
fi

echo "========================================"
echo "Sync: ${REMOTE} → local (remote-first)"
echo "========================================"
echo "Remote: ${REMOTE}"
echo "Branch: ${BRANCH}"
echo "Hard reset allowed: ${HARD_RESET}"
echo "========================================"

if [[ "${NO_STASH}" != "1" && -n "$(git status --porcelain)" ]]; then
  ts="$(pmx_timestamp)"
  echo -e "${YELLOW}Worktree has uncommitted changes. Stashing before sync (${ts})...${NC}"
  git stash push -u -m "pmx_sync_autostash_${ts}" >/dev/null
  STASHED=1
  echo -e "${GREEN}✓ Changes stashed.${NC}"
fi

echo "Fetching from ${REMOTE}..."
git fetch --prune "${REMOTE}"
echo -e "${GREEN}✓ Fetch complete.${NC}"

if ! git show-ref --verify --quiet "refs/remotes/${REMOTE}/${BRANCH}"; then
  echo -e "${RED}Remote branch ${REMOTE}/${BRANCH} not found.${NC}"
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "Local branch ${BRANCH} doesn't exist. Creating tracking branch..."
  git checkout -b "${BRANCH}" --track "${REMOTE}/${BRANCH}"
  echo -e "${GREEN}✓ Created and switched to ${BRANCH}.${NC}"
  restore_stash
  trap - EXIT
  echo -e "${GREEN}Sync complete.${NC}"
  exit 0
fi

if [[ "${START_BRANCH}" != "${BRANCH}" ]]; then
  git checkout "${BRANCH}"
fi

read -r ahead behind < <(git rev-list --left-right --count "${BRANCH}...${REMOTE}/${BRANCH}")

if [[ "${ahead}" == "0" && "${behind}" == "0" ]]; then
  echo -e "${GREEN}✓ Local ${BRANCH} is up to date with ${REMOTE}/${BRANCH}.${NC}"
  restore_stash
  trap - EXIT
  echo -e "${GREEN}Sync complete.${NC}"
  exit 0
fi

if [[ "${ahead}" == "0" && "${behind}" != "0" ]]; then
  echo "Fast-forwarding local ${BRANCH} to ${REMOTE}/${BRANCH}..."
  git merge --ff-only "${REMOTE}/${BRANCH}"
  echo -e "${GREEN}✓ Updated local ${BRANCH}.${NC}"
  restore_stash
  trap - EXIT
  echo -e "${GREEN}Sync complete.${NC}"
  exit 0
fi

ts="$(pmx_timestamp)"

if [[ "${ahead}" != "0" && "${behind}" == "0" ]]; then
  echo -e "${YELLOW}Local ${BRANCH} is ahead of ${REMOTE}/${BRANCH} by ${ahead} commit(s).${NC}"
  echo -e "${YELLOW}Remote is canonical; do not leave unpushed history on local machines.${NC}"
  echo -e "${YELLOW}Suggested: push these commits on a feature branch + PR (see Documentation/GIT_WORKFLOW.md).${NC}"
  git branch "backup/local-ahead-${BRANCH}-${ts}" >/dev/null 2>&1 || true
  CONFLICT_DETECTED=1
  exit 2
fi

echo -e "${YELLOW}Local ${BRANCH} diverged from ${REMOTE}/${BRANCH} (ahead=${ahead}, behind=${behind}).${NC}"
git branch "backup/diverged-${BRANCH}-${ts}" >/dev/null 2>&1 || true

if [[ "${HARD_RESET}" == "1" ]]; then
  echo -e "${YELLOW}PMX_HARD_RESET=1 set; resetting local ${BRANCH} to ${REMOTE}/${BRANCH}.${NC}"
  git reset --hard "${REMOTE}/${BRANCH}"
  echo -e "${GREEN}✓ Reset complete.${NC}"
  restore_stash
  trap - EXIT
  echo -e "${GREEN}Sync complete.${NC}"
  exit 0
fi

echo -e "${RED}Refusing to overwrite local history without explicit consent.${NC}"
echo -e "${YELLOW}Options:${NC}"
echo -e "${YELLOW}  - If local commits matter: push them on a feature branch + PR.${NC}"
echo -e "${YELLOW}  - If local commits are disposable: rerun with PMX_HARD_RESET=1 to hard reset.${NC}"
CONFLICT_DETECTED=1
exit 3
