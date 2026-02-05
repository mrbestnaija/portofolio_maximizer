# Git Workflow - Remote as Source of Truth

**Policy**: Remote repository (versioned, reviewed) is the source of truth
**Last Updated**: 2026-01-10
**Status**: ENFORCED
**Version**: 3.0

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Remote Repository](#remote-repository)
- [Git Configuration](#git-configuration)
- [Authentication & Security](#authentication--security)
- [Workflow Principles](#workflow-principles)
- [Common Operations](#common-operations)
- [Contributor Workflow](#contributor-workflow)
- [Conflict Resolution](#conflict-resolution)
- [Testing & Verification](#testing--verification)
- [Branch Strategy](#branch-strategy)
- [Backup Strategy](#backup-strategy)
- [Emergency Recovery](#emergency-recovery)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### For New Contributors

1. Clone the repository and check out `master`:
   ```bash
   git clone https://github.com/mrbestnaija/portofolio_maximizer.git
   cd portofolio_maximizer
   git checkout master
   ```

2. Sync from remote (remote is canonical):
   ```bash
   git fetch origin
   git pull --ff-only origin master
   ```

3. Create a feature branch for changes:
   ```bash
   git checkout -b feature/short-description
   ```


### For Existing Contributors

1. Sync from remote first (remote is truth):
   ```bash
   git fetch origin
   git checkout master
   git pull --ff-only origin master
   ```

2. Ensure your local branch is clean and tests pass:
   ```bash
   git status
   pytest tests/
   ```

3. Create a feature branch for changes (recommended):
   ```bash
   git checkout -b feature/short-description
   ```

4. Commit your work with a descriptive message:
   ```bash
   git add .
   git commit -m "feat(hyperopt): add higher-order post-eval driver"
   ```

5. Sync with remote (feature branch; remote is truth):
   ```bash
   # Bash/Zsh
   git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)" || true
   git push origin "$(git rev-parse --abbrev-ref HEAD)"
   ```

   ```powershell
   # PowerShell
   git pull --rebase origin (git rev-parse --abbrev-ref HEAD)
   if ($LASTEXITCODE -ne 0) { Write-Output "pull failed (continuing per remote-first policy)" }
   git push origin (git rev-parse --abbrev-ref HEAD)
   ```

6. Optional: use the automated helpers:
   ```bash
   # Sync local from remote (safe, non-destructive by default)
   bash/git_syn_to_local.sh master

   # Commit + push current branch (feature branch recommended)
   bash/git_sync.sh "feat: my change" feature/branch-name
   ```


---

## Remote Repository

**GitHub Repository**: https://github.com/mrbestnaija/portofolio_maximizer.git
**Primary Branch**: `master`

**Authentication policy (GitHub)**:
- Password authentication over HTTPS is **not supported**. Use SSH or a Personal Access Token (PAT).

### Remote Configuration

```bash
# Show current remote
git remote -v

# Recommended: SSH (avoids PAT/credential-helper issues)
git remote set-url origin git@github.com:mrbestnaija/portofolio_maximizer.git
```

### Adding Upstream (For Contributors)

```bash
# For forks: add upstream and keep it read-only
git remote add upstream git@github.com:mrbestnaija/portofolio_maximizer.git
git fetch upstream
```

---

## Git Configuration

The following git settings reduce divergence and keep remote history authoritative:


### What These Settings Do

| Setting | Value | Effect | Why |
|---------|-------|--------|-----|
| `pull.rebase` | `true` (recommended) | Pull rebases local commits on top of remote | Keeps feature branches linear |
| `pull.ff` | `only` (recommended on `master`) | Pull fails if a merge is required | Prevents accidental merge commits on `master` |
| `push.default` | `current` | Only push current branch to remote | Prevents accidental pushes of other branches |
| `merge.ff` | `false` (optional) | Merge commits when merging locally | Use PR merges for canonical history |

### Verification

```bash
git config --show-origin --get-all pull.rebase || true
git config --show-origin --get-all pull.ff || true
git config --show-origin --get-all push.default || true
```

### Recommended Defaults (Remote-First)

```bash
# Feature branches: rebase when pulling
git config --global pull.rebase true

# master: only allow fast-forward updates
git config --global pull.ff only

# Safer pushing
git config --global push.default current
```

---

## Authentication & Security

### GitHub Authentication (Required)
- **Preferred**: SSH (`git@github.com:mrbestnaija/portofolio_maximizer.git`).
- **Alternative**: HTTPS + **PAT** (classic or fine-grained) stored via a credential helper.
- **Forbidden**: GitHub account password for git operations (will fail).

### SSH (Recommended)
```bash
# Generate a key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add public key (~/.ssh/id_ed25519.pub) to GitHub > Settings > SSH and GPG keys
git remote set-url origin git@github.com:mrbestnaija/portofolio_maximizer.git

# Test auth
ssh -T git@github.com
```

### HTTPS + PAT (If You Must)
- Create a PAT with least privilege (repo scope for private repos; fine-grained token where possible).
- Do **not** embed tokens in the remote URL (it will leak via shell history/config).
- Prefer a credential manager prompt + secure storage.

### Credential Helpers (Windows vs WSL)
- **Windows (Git for Windows)**: use Git Credential Manager.
  - Verify: `git config --show-origin --get-all credential.helper`
- **WSL**: do not point `credential.helper` at a Windows `.exe` path (causes `Exec format error`).
  - Fix if misconfigured: `git config --global --unset-all credential.helper`
  - Then use SSH (recommended) or install a Linux-native credential manager.

### Secret Hygiene (Non-Negotiable)
- Never commit secrets: `.env`, API keys, tokens, SSH private keys, broker creds.
- Never commit runtime artifacts: SQLite DBs under `data/` (e.g., `data/portfolio_maximizer.db`, `data/dashboard_audit.db`) and generated payloads like `visualizations/dashboard_data.json` must stay untracked (verify `.gitignore` and review `git status` before staging).
- Always review staged changes: `git diff --cached` before commit.
- If a secret is committed: rotate it immediately and remove from history **before pushing**.

### Dependency Security Updates
- For dependency bumps and vulnerability fixes:
  - Prefer a dedicated branch + PR (e.g., `chore/deps-*`).
  - Run `python -m pip_audit -r requirements.txt` (and any extras files) before pushing.
  - Keep changes minimal and validate against the existing test baseline.

## Workflow Principles

### 1. **Remote Is Canonical**
- ✅ Remote branch history is the source of truth
- ✅ Local clones are working copies; sync forward from remote and avoid rewriting remote history
- ✅ Prefer PRs for reviewable, progressive development and to avoid retrogression

### 2. **Preserve History**
- ✅ Never rewrite published history on `master`
- ✅ Use PR merges (merge commit or squash) for canonical history
- ✅ Create backup branches before major refactors
- ✅ Document significant changes in commit messages

### 3. **Test Before Push**
- ✅ Run test suite before pushing: `pytest tests/`
- ✅ Verify code quality: Check linter output
- ✅ Test critical paths manually if needed
- ✅ Update documentation for architectural changes

### 4. **Clear Communication**
- ✅ Use conventional commit messages
- ✅ Document breaking changes
- ✅ Update relevant documentation files
- ✅ Communicate with team about major changes

---

## Common Operations

### Daily Development Workflow

1. Sync local `master` from remote (fast-forward only):
   ```bash
   git checkout master
   git fetch origin
   git pull --ff-only origin master
   ```

2. Create or update your feature branch:
   ```bash
   git checkout -b feature/short-description  # first time
   # or:
   git checkout feature/short-description
   ```

3. Commit locally (small, reviewable commits):
   ```bash
   git add -A
   git commit -m "feat(scope): short description"
   ```

4. Rebase your feature branch on the latest remote `master`, then push:
   ```bash
   git fetch origin
   git rebase origin/master
   git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
   ```

5. Open a PR and merge via GitHub (preferred). After merge, sync local `master` again:
   ```bash
   git checkout master
   git pull --ff-only origin master
   ```


---

## Contributor Workflow

### For External Contributors (Pull Requests)

1. Fork the repo on GitHub.
2. Add `upstream` and keep your fork synced:
   ```bash
   git remote add upstream git@github.com:mrbestnaija/portofolio_maximizer.git
   git fetch upstream
   git checkout master
   git pull --ff-only upstream master
   git push origin master
   ```
3. Create a branch in your fork, push, and open a PR to `upstream/master`.


### Keeping Fork Up to Date

```bash
git fetch upstream
git checkout master
git pull --ff-only upstream master
git push origin master
```

### For Team Members (Direct Access)

- Prefer feature branches + PRs for anything non-trivial.
- Avoid direct pushes to `master` unless it’s a simple, low-risk change and CI will validate it immediately.

---

## Conflict Resolution

### Understanding Conflicts

Conflicts occur when:
- Local and remote have different changes to the same lines
- Both branches modified the same file in incompatible ways
- Merge cannot be automatically resolved

### Resolution Strategy

**Policy**: Remote is canonical; resolve conflicts by preserving remote history and applying local changes on top (usually on a feature branch).

#### Method 1: Keep Remote Version (Recommended on `master`)

If your local `master` diverged (should be rare), keep remote as canonical:
```bash
git checkout master
git fetch origin
git branch backup/local-master-before-sync-$(date -u +%Y%m%dT%H%M%SZ)
git reset --hard origin/master
```

#### Method 2: Keep Local Version (Feature branches / local experiments)

If your local commits matter, move them to a feature branch and push as a PR:
```bash
git checkout -b feature/recover-local-work
git push -u origin feature/recover-local-work
```

#### Method 3: Manual Resolution

For feature branches, rebase and resolve conflicts:
```bash
git fetch origin
git rebase origin/master
# resolve conflicts, then:
git add <files>
git rebase --continue
```

### Conflict Prevention

- Keep `master` fast-forward only.
- Rebase feature branches frequently onto `origin/master`.
- Keep commits small and scoped.

---

## Testing & Verification

### Pre-Commit Checklist

Before committing, verify:


### Pre-Push Checklist

Before pushing to remote:

- Run unit/integration tests relevant to your changes.
- For profit-critical or hyperopt changes:
  - Run the time series-first brutal suite when feasible:
    ```bash
    bash/comprehensive_brutal_test.sh
    # Optional legacy LLM path:
    # BRUTAL_ENABLE_LLM=1 bash/comprehensive_brutal_test.sh
    ```
- If you touched hyperopt/strategy tuning:
  - Sanity-check `bash/run_post_eval.sh` on a small hyperopt run:
    ```bash
    HYPEROPT_ROUNDS=1 bash/run_post_eval.sh
    ```

### WSL Note
- Repository scripts under `bash/` assume a Unix-like shell; run them under WSL for consistency.

### GitHub Actions Checks (What Blocks Merges)

In PRs, treat checks in two classes:

1) **Required (must be green)**:
   - `CI / test` (runs `pip install -r requirements.txt`, `pip check`, then `pytest`).

2) **Nice-to-have (should not block merges)**:
   - Project automation workflows (e.g. `Sync issues and PRs to Project 7`) that depend on repo secrets and permissions.

If a non-critical automation check fails, fix its permissions/guards or make it skip cleanly when secrets are absent.

### Fixing a Failing CI/Test Run (Fast Playbook)

1) Open the failing `CI / test` job log and find the **first** real error (ignore setup noise).
2) Reproduce locally with the **same Python version** used in CI and a clean environment:
   - Install deps from `requirements.txt`
   - Run `pip check`
   - Run the same `pytest` command as CI
3) For dependency/security PRs:
   - Run `python -m pip_audit -r requirements.txt` (and any optional extras files) and keep the diff minimal.
4) If the breakage is a dependency constraint conflict:
   - Prefer bumping the conflicting library instead of pinning around it, unless the pin is strictly temporary and documented.

### Automated Testing (Recommended)

Create a pre-push hook (optional):


---

## Branch Strategy

### Current Setup

- **Primary Branch**: `master`
- **Development Model**: Single branch with feature branches as needed
- **Remote Tracking**: `origin/master` is canonical; locals must fast-forward or rebase feature branches

### Branch Types

| Branch Type | Purpose | Naming Convention | Lifecycle |
|-------------|---------|-------------------|-----------|
| `master` | Production-ready code | `master` | Permanent |
| Feature | New features | `feature/description` | Temporary, merge then delete |
| Bugfix | Bug fixes | `fix/description` | Temporary, merge then delete |
| Backup | Safety snapshots | `backup/description-date` | Temporary, manual cleanup |

### Creating Feature Branches


### Backup Branches


---

## Backup Strategy

### Remote as Source of Truth (Canonical)

- ✅ Remote `origin/master` is the canonical, versioned history.
- ✅ Local clones are disposable working copies; don’t treat a local machine as authoritative.
- ✅ Progressive development happens via feature branches + PRs + green CI.
- ✅ Never “repair” remote history from a local machine via force push on `master`.

### Local Backups

Use local-only backups to protect work-in-progress without rewriting remote history:
```bash
git branch backup/wip-$(date -u +%Y%m%dT%H%M%SZ)
```

### Backup Best Practices

1. **Before Major Refactors**: Create backup branch
2. **Before Risky Git Operations**: Create backup branch/tag
3. **Regular Pushes**: Push feature branches frequently; merge via PR
4. **Documentation**: Document backup strategy in commit messages

---

## Emergency Recovery

### Scenario 1: Accidental Remote Override

**Situation**: A bad change was merged to `master`.

**Response**: Revert via a PR; do not force push:
```bash
git checkout -b fix/revert-bad-merge
# git revert <sha> ...
git push -u origin fix/revert-bad-merge
```


### Scenario 2: Local Repository Corruption

**Situation**: Local git repository is corrupted or lost

**Response**: Re-clone and sync from remote:
```bash
git clone https://github.com/mrbestnaija/portofolio_maximizer.git
```

### Scenario 3: Need to Inspect Remote Changes

**Situation**: Want to see what changed on remote without merging

```bash
git fetch origin
git log --oneline --decorate --graph --max-count=30 origin/master
```

### Scenario 4: Undo Last Commit (Before Push)

```bash
git reset --soft HEAD~1
```

### Scenario 5: Undo Last Commit (After Push)

```bash
git revert HEAD
```

---

## Force Push Policy

### When Force Push is Allowed

✅ **DISALLOWED on `master`** - remote is canonical:
- Never force push `master`.

✅ **Allowed on feature branches (with care)**:
- Before opening a PR (to clean up work-in-progress history).
- Only when you understand the impact on collaborators.

### Safe Force Push

**Always use `--force-with-lease` instead of `--force`**:

```bash
git push --force-with-lease origin "$(git rev-parse --abbrev-ref HEAD)"
```

### Force Push Best Practices

1. ✅ Always use `--force-with-lease`
2. ✅ Create backup branch/tag before force push
3. ✅ Communicate with team before force pushing to shared branches
4. ✅ Verify local changes are correct before force push
5. ✅ Document reason in commit message if force push was needed

---

## .gitignore Strategy

### Files Ignored

Configured to ignore:
- **Generated data files**: `data/raw/*.parquet`, `data/processed/*.parquet`
- **Virtual environments**: `simpleTrader_env/`, `portfolio_env/`, `venv/`
- **Cache files**: `__pycache__/`, `.pytest_cache/`, `*.pyc`
- **Sensitive files**: `.env`, `*.key`, `secrets/`, `*.pem`
- **Large model files**: `*.pkl`, `*.h5`, `*.model`
- **IDE files**: `.vscode/`, `.idea/`, `*.swp`
- **OS files**: `.DS_Store`, `Thumbs.db`
- **Database files**: `*.db-shm`, `*.db-wal` (SQLite temporary files)

### Files Tracked

Keep in version control:
- ✅ Source code (`etl/`, `scripts/`, `tests/`, `ai_llm/`)
- ✅ Configuration templates (`config/*.yml`)
- ✅ Documentation (`Documentation/`)
- ✅ Empty directory markers (`.gitkeep` files)
- ✅ Test fixtures and sample data
- ✅ Build scripts and automation

### Verifying .gitignore


---

## Verification Commands

### Check Git Configuration


### Check Repository Status


### Check Uncommitted Changes


### Check Commit History


---

## Best Practices

### Commit Practices

1. **Commit Often**: Small, atomic commits locally
   - Each commit should represent a logical change
   - Easier to review and revert if needed

2. **Descriptive Messages**: Follow conventional commits
   - Format: `<type>(<scope>): <subject>`
   - Include body for complex changes
   - Reference issues/PRs when applicable

3. **Test Before Commit**: Run relevant tests
   - Unit tests for changed modules
   - Integration tests for pipeline changes
   - Manual testing for critical paths

4. **Review Changes**: Check what you're committing
   - `git diff --cached` before commit
   - Verify no sensitive data
   - Check for accidental large files

### Push Practices

1. **Test Before Push**: Run full test suite
   ```bash
   pytest tests/ --cov=etl
   ```

2. **Verify Status**: Check repository state
   ```bash
   git status
   git log origin/master..HEAD
   ```

3. **Push Safely**: Prefer rebase/push on the current branch; only force-with-lease when history must be rewritten
   ```bash
   # Sync current branch
   git pull --rebase origin $(git rev-parse --abbrev-ref HEAD)
   git push origin $(git rev-parse --abbrev-ref HEAD)

   # Rare: rewrite published history safely
   git push --force-with-lease origin $(git rev-parse --abbrev-ref HEAD)
   ```

4. **Push Regularly**: Don't accumulate too many local commits
   - Push after each significant milestone
   - Push at least daily for active development

### Documentation Practices

1. **Update Docs**: When architecture changes
   - Update `Documentation/implementation_checkpoint.md`
   - Update `Documentation/arch_tree.md`
   - Update `README.md` if user-facing changes

2. **Document Breaking Changes**: In commit messages
   - Use `BREAKING CHANGE:` footer
   - Explain migration path
   - Update version numbers if applicable

3. **Keep Changelog**: Document significant changes
   - Major features
   - Breaking changes
   - Performance improvements

---

## Conventional Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(etl): Add intelligent caching` |
| `fix` | Bug fix | `fix(preprocessor): Handle missing data` |
| `refactor` | Code restructuring | `refactor: Streamline architecture` |
| `test` | Add/update tests | `test(etl): Add cache mechanism tests` |
| `docs` | Documentation changes | `docs: Update README with Ollama setup` |
| `chore` | Maintenance tasks | `chore: Update dependencies` |
| `perf` | Performance improvements | `perf(cache): Optimize cache lookup` |
| `style` | Code style changes | `style: Format with black` |
| `ci` | CI/CD changes | `ci: Add GitHub Actions workflow` |

### Scope Examples

- `etl`: ETL pipeline modules
- `ai_llm`: LLM integration modules
- `scripts`: Executable scripts
- `tests`: Test files
- `config`: Configuration files
- `docs`: Documentation

### Examples


### Footer Options

```
🤖 Generated with co-companion

Authored-By: Bestman Ezekwu Enock
Email : csgtmalice@protonmail.ch

BREAKING CHANGE: Description of breaking change
Fixes #123
Closes #456
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Your branch is ahead of 'origin/master'"

**Solution**: This should not happen on `master`. Create a branch for the commits and push that branch:
```bash
git checkout -b feature/recover-local-commits
git push -u origin feature/recover-local-commits
```

#### Issue 2: "Your branch and 'origin/master' have diverged"

**Solution**: Remote is canonical. Reset local `master` to `origin/master` (after creating a backup branch):
```bash
git checkout master
git fetch origin
git branch backup/diverged-master-$(date -u +%Y%m%dT%H%M%SZ)
git reset --hard origin/master
```

#### Issue 3: Merge conflicts during pull

**Solution**: On `master`, avoid merges: keep `master` fast-forward only. On feature branches, rebase and resolve conflicts, then push.

#### Issue 4: Accidentally committed sensitive data

**Solution**: Remove from history (before push)

#### Issue 5: Want to undo last commit (not pushed)

**Solution**: Reset to previous commit

#### Issue 6: Git configuration not applied

**Solution**: Reapply configuration

#### Issue 7: Remote URL incorrect

**Solution**: Update remote URL

#### Issue 8: "CI / test" failing on a PR

**Solution**: Open the job log and reproduce locally; the first traceback is the root cause. For dependency PRs, run `pip check` and `python -m pip_audit -r requirements.txt` before pushing.

#### Issue 9: "Sync issues and PRs to Project 7" failing quickly

**Solution**: This is usually repo permissions or missing `PROJECTS_TOKEN` secret. Either configure the secret with correct scopes or ensure the workflow skips when the token is absent (do not let it block merges).

#### Issue 10: "Password authentication is not supported"

**Solution**: Use SSH or HTTPS with a PAT (never a password).

#### Issue 11: "git-credential-manager.exe: Exec format error" (WSL)

**Solution**: Your WSL git is trying to run a Windows credential helper; unset it and use SSH.

#### Issue 12: "Invalid username or token"

**Solution**: Regenerate a PAT (correct scopes) or switch to SSH; clear cached credentials if needed.

### Getting Help

1. **Check Git Status**: `git status`
2. **View Recent Commits**: `git log --oneline -10`
3. **Check Configuration**: `git config --local --list`
4. **Review Documentation**: This file and `README.md`
5. **Git Help**: `git help <command>`

---

## Remote Collaboration Guidelines

### For Team Members

1. **Communication**: Never force push `master`; coordinate before rewriting any shared branch
2. **Feature Branches**: Use feature branches for experimental work
3. **Regular Sync**: Push changes at least daily
4. **Test Before Push**: Always run tests before pushing
5. **Document Changes**: Update relevant documentation

### For External Contributors

1. **Fork First**: Always fork before contributing
2. **Feature Branches**: Create feature branch in your fork
3. **Pull Requests**: Open PR for review before merging
4. **Follow Standards**: Use conventional commits, run tests
5. **Stay Updated**: Keep your fork synced with upstream

### Collaboration Workflow


---

## Summary

✅ **Configured**: Remote-first git workflow
✅ **Policy**: Remote history is canonical
✅ **Remote**: https://github.com/mrbestnaija/portofolio_maximizer.git
✅ **Settings**: recommend `pull.rebase=true` and `pull.ff=only` on `master`
✅ **Conflicts**: preserve remote history; apply local changes via feature branches
✅ **Force Push**: never on `master`; feature branches only with care
✅ **Testing**: Run tests before push
✅ **Documentation**: Update docs with architectural changes

---

**Document Version**: 3.0
**Last Updated**: 2026-01-10
**Status**: ACTIVE
**Review Schedule**: Monthly or before major changes
**Maintainer**: Bestman Ezekwu Enock (csgtmalice@protonmail.ch)

---

## Appendix: Platform-Specific Notes

### Windows (PowerShell)

```powershell
# Git commands work the same
git status
git add .
git commit -m "feat: Description"

# Date formatting for backup branches
git branch backup/before-refactor-$(Get-Date -Format "yyyyMMdd")
```

### Linux/WSL


### macOS


---

**End of Document**
