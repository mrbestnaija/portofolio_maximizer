#!/usr/bin/env bash
set -euo pipefail

# repo_cleanup.sh
# Archive (or optionally delete) repositories never updated or reviewed by a user.
# - "Updated by" => any commit authored by the user in the repository history
# - "Reviewed by" => any PR in the repo with a review by the user (via GitHub search)
#
# Requirements:
# - GitHub CLI (gh) installed and authenticated: `gh auth login` (repo scope; delete_repo for --delete)
# - No external deps (uses gh --jq for JSON parsing)
#
# Usage examples:
#   Dry-run for your user (auto-detect login):
#     scripts/repo_cleanup.sh
#
#   Dry-run for explicit user:
#     scripts/repo_cleanup.sh --user yourlogin
#
#   Apply archiving for your user (exclude forks, skip already-archived):
#     scripts/repo_cleanup.sh --apply
#
#   Target an organization (requires org admin permissions for actions):
#     scripts/repo_cleanup.sh --org YourOrg --apply
#
#   Actually delete instead of archive (DANGEROUS; requires --yes):
#     scripts/repo_cleanup.sh --delete --yes --apply
#
# Flags:
#   --user <login>           Target GitHub username (default: authenticated viewer)
#   --org <org>              Target organization instead of user
#   --include-forks          Include forks (default: exclude)
#   --include-archived       Include archived repos in consideration (default: exclude)
#   --visibility <v>         Filter: public|private|all (default: all)
#   --limit <n>              Max repos to act on (default: unlimited)
#   --apply                  Perform actions (default: dry-run)
#   --delete                 Delete instead of archive (requires --yes; dangerous)
#   --yes                    Skip confirmation prompts for deletions
#   --help                   Show help
#

SCRIPT_NAME=$(basename "$0")
ACTION="archive"      # archive|delete
DRY_RUN=1              # 1=dry-run, 0=apply
INCLUDE_FORKS=0
INCLUDE_ARCHIVED=0
VISIBILITY="all"      # all|public|private
OWNER_TYPE="user"     # user|org
TARGET_LOGIN=""
LIMIT=""
CONFIRM_YES=0

print_help() {
  cat <<EOF
$SCRIPT_NAME - Archive or delete repos never updated or reviewed by a user

Usage:
  $SCRIPT_NAME [--user <login> | --org <org>] [options]

Options:
  --user <login>         Target GitHub username (default: authenticated viewer)
  --org <org>            Target organization instead of user
  --include-forks        Include forks (default: exclude)
  --include-archived     Include archived repos (default: exclude)
  --visibility <v>       Filter repos: public|private|all (default: all)
  --limit <n>            Max repos to act on (default: unlimited)
  --apply                Perform actions (default: dry-run)
  --delete               Delete instead of archive (requires --yes)
  --yes                  Confirm destructive delete without prompt
  --help                 Show this help

Behavior:
- A repo is a candidate if BOTH conditions hold:
  1) No commits authored by the target user
  2) No PRs in the repo reviewed by the target user
- Default action is dry-run report. Use --apply to perform actions.
- Default operation archives repos. Use --delete (with --yes) to delete instead.
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      TARGET_LOGIN="$2"; shift 2;;
    --org)
      OWNER_TYPE="org"; TARGET_LOGIN="$2"; shift 2;;
    --include-forks)
      INCLUDE_FORKS=1; shift;;
    --include-archived)
      INCLUDE_ARCHIVED=1; shift;;
    --visibility)
      VISIBILITY="$2"; shift 2;;
    --limit)
      LIMIT="$2"; shift 2;;
    --apply)
      DRY_RUN=0; shift;;
    --delete)
      ACTION="delete"; shift;;
    --yes)
      CONFIRM_YES=1; shift;;
    --help|-h)
      print_help; exit 0;;
    *)
      echo "Unknown option: $1" >&2; print_help; exit 1;;
  esac
done

# Safety guardrails
if [[ "$ACTION" == "delete" && $CONFIRM_YES -ne 1 ]]; then
  echo "Refusing to delete without --yes. Use --apply --delete --yes if you are absolutely sure." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi

# Verify authentication (needed to act on private repos or perform any modifications)
if ! gh auth status >/dev/null 2>&1; then
  echo "You are not authenticated. Run: gh auth login" >&2
  exit 1
fi

# Determine default target login from viewer if not provided
if [[ -z "$TARGET_LOGIN" ]]; then
  TARGET_LOGIN=$(gh api graphql -f query='query { viewer { login } }' --jq '.data.viewer.login')
fi

if [[ -z "$TARGET_LOGIN" ]]; then
  echo "Unable to determine target login." >&2
  exit 1
fi

# Visibility filter helper
matches_visibility() {
  local is_private="$1"  # true/false
  case "$VISIBILITY" in
    all) return 0;;
    public)
      [[ "$is_private" == "false" ]];;
    private)
      [[ "$is_private" == "true" ]];;
    *)
      echo "Invalid visibility: $VISIBILITY" >&2; return 1;;
  esac
}

# Enumerate repositories (owner-typed)
# Emit TSV: owner\tname\tfull_name\tarchived\tfork\tprivate
list_repos() {
  if [[ "$OWNER_TYPE" == "org" ]]; then
    gh api --paginate \
      -X GET "/orgs/${TARGET_LOGIN}/repos" -F per_page=100 -F type=all \
      --jq '.[] | [.owner.login, .name, .full_name, .archived, .fork, .private] | @tsv'
  else
    # If targeting the authenticated user, /user/repos with affiliation=owner is most accurate
    # Otherwise, fall back to /users/:username/repos (public only for other users)
    local viewer
    viewer=$(gh api graphql -f query='query { viewer { login } }' --jq '.data.viewer.login')
    if [[ "$TARGET_LOGIN" == "$viewer" ]]; then
      gh api --paginate \
        -X GET "/user/repos" -F per_page=100 -F affiliation=owner \
        --jq '.[] | [.owner.login, .name, .full_name, .archived, .fork, .private] | @tsv'
    else
      gh api --paginate \
        -X GET "/users/${TARGET_LOGIN}/repos" -F per_page=100 -F type=owner \
        --jq '.[] | [.owner.login, .name, .full_name, .archived, .fork, .private] | @tsv'
    fi
  fi
}

# Check if target user authored any commit in repo (0/1)
commit_count_by_user() {
  local owner="$1" repo="$2" user="$3"
  # Per-page 1 to minimize
  gh api "/repos/${owner}/${repo}/commits" -F per_page=1 -F author="$user" --jq 'length' 2>/dev/null || echo 0
}

# Check if target user reviewed any PR in the repo via search API total_count
review_count_by_user() {
  local full_name="$1" user="$2"
  # Search across all PRs; includes open/closed/merged
  local q="is:pr repo:${full_name} reviewed-by:${user}"
  gh api "/search/issues" -F q="$q" --jq '.total_count' 2>/dev/null || echo 0
}

confirm_delete() {
  local full_name="$1"
  if [[ $CONFIRM_YES -eq 1 ]]; then return 0; fi
  read -r -p "Delete ${full_name}? Type 'delete ${full_name}' to confirm: " ans
  [[ "$ans" == "delete ${full_name}" ]]
}

acts=0
candidates=0

report_header() {
  echo "Target: $OWNER_TYPE $TARGET_LOGIN"
  echo "Mode: ${DRY_RUN:+DRY-RUN}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}${DRY_RUN:0:0}$( [[ $DRY_RUN -eq 0 ]] && echo APPLY )"
  echo "Action: $ACTION"
  echo "Filters: include_forks=$INCLUDE_FORKS include_archived=$INCLUDE_ARCHIVED visibility=$VISIBILITY limit=${LIMIT:-none}"
  echo "--"
}

report_header

while IFS=$'\t' read -r owner name full_name archived fork private; do
  # Visibility filter
  if ! matches_visibility "$private"; then
    continue
  fi
  # Skip forks unless included
  if [[ $INCLUDE_FORKS -ne 1 && "$fork" == "true" ]]; then
    continue
  fi
  # Skip archived unless included
  if [[ $INCLUDE_ARCHIVED -ne 1 && "$archived" == "true" ]]; then
    continue
  fi

  # Activity checks
  commits=$(commit_count_by_user "$owner" "$name" "$TARGET_LOGIN" || echo 0)
  reviews=$(review_count_by_user "$full_name" "$TARGET_LOGIN" || echo 0)

  # Normalize numeric
  commits=${commits:-0}
  reviews=${reviews:-0}
  if [[ "$commits" =~ ^[0-9]+$ ]] && [[ "$reviews" =~ ^[0-9]+$ ]]; then
    :
  else
    # If API errors produce non-numeric, treat as 0 but warn
    echo "WARN: Non-numeric activity result for $full_name (commits='$commits', reviews='$reviews'), treating as 0" >&2
    commits=0
    reviews=0
  fi

  if [[ $commits -eq 0 && $reviews -eq 0 ]]; then
    candidates=$((candidates+1))
    echo "CANDIDATE: $full_name (commits=$commits, reviews=$reviews, fork=$fork, archived=$archived, private=$private)"

    if [[ -n "$LIMIT" && $acts -ge ${LIMIT} ]]; then
      continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
      continue
    fi

    if [[ "$ACTION" == "archive" ]]; then
      echo "Archiving $full_name ..."
      gh api -X PATCH "/repos/${owner}/${name}" -F archived=true >/dev/null
      acts=$((acts+1))
    else
      # delete
      if confirm_delete "$full_name"; then
        echo "Deleting $full_name ..."
        gh api -X DELETE "/repos/${owner}/${name}" >/dev/null
        acts=$((acts+1))
      else
        echo "Skip delete for $full_name (confirmation failed)"
      fi
    fi
  fi

done < <(list_repos)

echo "--"
echo "Candidates: $candidates"
echo "${DRY_RUN:+Actions planned}${DRY_RUN:0:0}$( [[ $DRY_RUN -eq 0 ]] && echo Actions performed ): $acts"

