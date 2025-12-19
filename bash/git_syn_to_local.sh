#!/usr/bin/env bash
# Client PC Git Sync Script - GitHub as Source of Truth
#
# Usage:
#   bash/git_syn_to_local.sh                 # sync current branch from GitHub
#   bash/git_syn_to_local.sh branch-name     # sync specific branch from GitHub
#
# Safety:
# - Auto-stashes any dirty worktree before syncing
# - Checks if local changes conflict with remote
# - Only pulls from GitHub, never pushes
# - Restores stashed changes after sync if safe

set -euo pipefail

# Optional: load credentials and repo hints from .env (never commit .env)
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

# Optional: set git identity from env
if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config user.name "$GIT_USER_NAME"
fi
if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

# Configuration
REMOTE="origin"
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
STASHED=0
RESTORED=0
CONFLICT_DETECTED=0

# Optional: update origin URL with token if provided (HTTPS only)
if [[ -n "${GitHub_Username:-}" && -n "${GitHub_TOKEN:-}" && -n "${GitHub_Repo:-}" ]]; then
  git remote set-url origin "https://${GitHub_Username}:${GitHub_TOKEN}@github.com/${GitHub_Username}/${GitHub_Repo}.git"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to restore stash safely
restore_stash() {
  if [[ $STASHED -eq 1 && $RESTORED -eq 0 ]]; then
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    # Only restore if we're on the same branch and no conflicts were detected
    if [[ "$current_branch" != "$START_BRANCH" ]]; then
      echo -e "${YELLOW}Stash created on $START_BRANCH. Skipping auto-restore to avoid applying changes onto $current_branch.${NC}"
      echo -e "${YELLOW}Recover manually: git checkout $START_BRANCH && git stash pop${NC}"
      return
    fi
    
    if [[ $CONFLICT_DETECTED -eq 1 ]]; then
      echo -e "${YELLOW}Conflicts detected during sync. Stash not restored to avoid conflict propagation.${NC}"
      echo -e "${YELLOW}Review remote changes first, then restore stash manually: git stash pop${NC}"
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

# Cleanup on exit
trap restore_stash EXIT

# Validate branch
if [[ -z "$BRANCH" ]]; then
  echo -e "${RED}Unable to determine branch. Pass explicitly: bash/git_syn_to_local.sh my-branch${NC}"
  exit 1
fi

echo "========================================"
echo "Client Sync: GitHub → Local"
echo "========================================"
echo "Remote: $REMOTE"
echo "Branch: $BRANCH"
echo "========================================"

# Check if we have uncommitted changes
if [[ -n "$(git status --porcelain)" ]]; then
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo -e "${YELLOW}Worktree has uncommitted changes. Stashing before sync ($ts)...${NC}"
  git stash push -u -m "client_sync_autostash_$ts" >/dev/null
  STASHED=1
  echo -e "${GREEN}✓ Changes stashed.${NC}"
fi

# Fetch latest from GitHub
echo "Fetching latest from GitHub..."
if ! git fetch "$REMOTE" "$BRANCH"; then
  echo -e "${RED}Fetch failed for $REMOTE/$BRANCH${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Fetch complete.${NC}"

# Check if we're behind remote
LOCAL_HASH=$(git rev-parse "$BRANCH" 2>/dev/null || echo "")
REMOTE_HASH=$(git rev-parse "$REMOTE/$BRANCH" 2>/dev/null || echo "")

if [[ -z "$REMOTE_HASH" ]]; then
  echo -e "${YELLOW}Remote branch $REMOTE/$BRANCH not found.${NC}"
  exit 1
fi

if [[ -z "$LOCAL_HASH" ]]; then
  echo "Local branch $BRANCH doesn't exist. Creating from remote..."
  git checkout -b "$BRANCH" "$REMOTE/$BRANCH"
  echo -e "${GREEN}✓ Created and switched to branch $BRANCH from remote.${NC}"
elif [[ "$LOCAL_HASH" == "$REMOTE_HASH" ]]; then
  echo -e "${GREEN}✓ Local branch is up to date with GitHub.${NC}"
  
  # Switch to branch if not already there
  if [[ "$START_BRANCH" != "$BRANCH" ]]; then
    git checkout "$BRANCH"
    echo -e "${GREEN}✓ Switched to branch $BRANCH.${NC}"
  fi
  
  # Early exit - nothing to do
  restore_stash
  trap - EXIT
  echo "========================================"
  echo -e "${GREEN}Sync complete - no changes needed.${NC}"
  exit 0
else
  # Check if local has commits not in remote
  if git merge-base --is-ancestor "$REMOTE_HASH" "$LOCAL_HASH" 2>/dev/null; then
    echo -e "${YELLOW}Local branch has commits not in GitHub.${NC}"
    echo -e "${YELLOW}This PC is client-only. Consider:${NC}"
    echo -e "${YELLOW}  1. git log $REMOTE/$BRANCH..$BRANCH (view local-only commits)${NC}"
    echo -e "${YELLOW}  2. If these should be on GitHub, use master PC to push them${NC}"
    echo -e "${YELLOW}  3. Otherwise, consider resetting: git reset --hard $REMOTE/$BRANCH${NC}"
    CONFLICT_DETECTED=1
  fi
  
  # Switch to branch
  if [[ "$START_BRANCH" != "$BRANCH" ]]; then
    git checkout "$BRANCH"
  fi
  
  # Pull with rebase to keep history linear
  echo "Pulling changes from GitHub (with rebase)..."
  if git pull --rebase "$REMOTE" "$BRANCH"; then
    echo -e "${GREEN}✓ Successfully pulled and rebased.${NC}"
  else
    echo -e "${RED}Rebase failed due to conflicts.${NC}"
    echo -e "${YELLOW}Resolve conflicts manually, then:${NC}"
    echo -e "${YELLOW}  1. Fix the conflicting files${NC}"
    echo -e "${YELLOW}  2. git add <resolved-files>${NC}"
    echo -e "${YELLOW}  3. git rebase --continue${NC}"
    echo -e "${YELLOW}Or abort with: git rebase --abort${NC}"
    CONFLICT_DETECTED=1
    exit 1
  fi
fi

# Verify sync
LOCAL_HASH_AFTER=$(git rev-parse "$BRANCH")
if [[ "$LOCAL_HASH_AFTER" == "$REMOTE_HASH" ]]; then
  echo -e "${GREEN}✓ Local branch now matches GitHub.${NC}"
else
  echo -e "${YELLOW}Note: Local branch differs from GitHub after sync.${NC}"
  echo -e "${YELLOW}This is normal if you had local commits that were rebased.${NC}"
fi

# Restore stash if we had one
restore_stash
trap - EXIT

echo "========================================"
echo -e "${GREEN}Client sync complete for $BRANCH.${NC}"
echo "========================================"
echo -e "${YELLOW}REMEMBER: This PC is CLIENT-ONLY${NC}"
echo -e "${YELLOW}To update GitHub, use your master PC${NC}"
echo "========================================"
