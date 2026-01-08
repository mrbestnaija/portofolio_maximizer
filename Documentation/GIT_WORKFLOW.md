# Git Workflow - Local as Source of Truth

**Policy**: Local repository is ALWAYS the source of truth  
**Last Updated**: 2025-11-24  
**Status**: ENFORCED  
**Version**: 2.1

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Remote Repository](#remote-repository)
- [Git Configuration](#git-configuration)
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


### For Existing Contributors

1. Ensure your local branch is clean and tests pass:
   ```bash
   git status
   pytest tests/
   ```

2. Commit your work with a descriptive message:
   ```bash
   git add .
   git commit -m "feat(hyperopt): add higher-order post-eval driver"
   ```

3. Sync with remote (local is source of truth, use current branch):
   ```bash
   git pull --rebase origin $(git rev-parse --abbrev-ref HEAD) || true
   git push origin $(git rev-parse --abbrev-ref HEAD)
   ```

4. Optional: use the automated sync helper to run the same pull/push sequence for the current (or a named) branch:
   ```bash
   # Sync current branch
   bash/bash/git_sync.sh

   # Or sync a specific branch
   bash/bash/git_sync.sh feature/branch-name
   ```


---

## Remote Repository

**GitHub Repository**: https://github.com/mrbestnaija/portofolio_maximizer.git  
**Primary Branch**: `master`  
**Last Sync**: 2025-11-24 (local hyperopt + doc updates, not yet pushed in this repo snapshot)

### Remote Configuration


### Adding Upstream (For Contributors)


---

## Git Configuration

The following git settings ensure local changes always take precedence and maintain a clear history:


### What These Settings Do

| Setting | Value | Effect | Why |
|---------|-------|--------|-----|
| `pull.rebase` | `false` | Pull creates merge commits, preserving local history | Maintains complete development timeline |
| `push.default` | `current` | Only push current branch to remote | Prevents accidental pushes of other branches |
| `merge.ff` | `false` | Always create merge commit (no fast-forward) | Clear merge history, easier to track changes |

### Verification


---

## Workflow Principles

### 1. **Local Changes Take Priority**
- ✅ All development happens locally first
- ✅ Local commits are never rebased or amended after creation
- ✅ Remote is a backup/sync point, not the primary workspace
- ✅ Local repository is the definitive source of truth

### 2. **Preserve History**
- ✅ Never rewrite published history
- ✅ Use merge commits to combine changes
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

#### 1. Make Local Changes


#### 2. Sync to Remote (Backup)


**Note**: `--force-with-lease` is safer than `--force` as it checks if remote has changed since last fetch.

#### 3. Pull from Remote (When Needed)


---

## Contributor Workflow

### For External Contributors (Pull Requests)


### Keeping Fork Up to Date


### For Team Members (Direct Access)


---

## Conflict Resolution

### Understanding Conflicts

Conflicts occur when:
- Local and remote have different changes to the same lines
- Both branches modified the same file in incompatible ways
- Merge cannot be automatically resolved

### Resolution Strategy

**Policy**: Local changes take priority (local is source of truth)

#### Method 1: Keep Local Version (Recommended)


#### Method 2: Keep Remote Version


#### Method 3: Manual Resolution


### Conflict Prevention


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
    bash/bash/comprehensive_brutal_test.sh
    # Optional legacy LLM path:
    # BRUTAL_ENABLE_LLM=1 bash/bash/comprehensive_brutal_test.sh
    ```
- If you touched hyperopt/strategy tuning:
  - Sanity-check `bash/run_post_eval.sh` on a small hyperopt run:
    ```bash
    HYPEROPT_ROUNDS=1 bash/bash/run_post_eval.sh
    ```

### Automated Testing (Recommended)

Create a pre-push hook (optional):


---

## Branch Strategy

### Current Setup

- **Primary Branch**: `master`
- **Development Model**: Single branch with feature branches as needed
- **Remote Tracking**: Optional, local is definitive

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

### Remote as Backup

- ✅ Remote repository (GitHub) serves as backup
- ✅ Local always has latest, most accurate code
- ✅ Push frequency: After each significant milestone or daily
- ✅ Force push allowed with `--force-with-lease` for safety

### Local Backups


### Backup Best Practices

1. **Before Major Refactors**: Create backup branch
2. **Before Force Push**: Create backup tag
3. **Regular Pushes**: Push to remote after each milestone
4. **Documentation**: Document backup strategy in commit messages

---

## Emergency Recovery

### Scenario 1: Accidental Remote Override

**Situation**: Remote was accidentally updated, need to restore local version


### Scenario 2: Local Repository Corruption

**Situation**: Local git repository is corrupted or lost


### Scenario 3: Need to Inspect Remote Changes

**Situation**: Want to see what changed on remote without merging


### Scenario 4: Undo Last Commit (Before Push)


### Scenario 5: Undo Last Commit (After Push)


---

## Force Push Policy

### When Force Push is Allowed

✅ **ALLOWED** - Since local is source of truth:
- After local rebase (if needed)
- To restore local version after accidental remote update
- To clean up commit history (before sharing)
- When working on feature branches

### Safe Force Push

**Always use `--force-with-lease` instead of `--force`**:


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

**Solution**: Push your local commits

#### Issue 2: "Your branch and 'origin/master' have diverged"

**Solution**: Use force-with-lease (local takes priority)

#### Issue 3: Merge conflicts during pull

**Solution**: Keep local version

#### Issue 4: Accidentally committed sensitive data

**Solution**: Remove from history (before push)

#### Issue 5: Want to undo last commit (not pushed)

**Solution**: Reset to previous commit

#### Issue 6: Git configuration not applied

**Solution**: Reapply configuration

#### Issue 7: Remote URL incorrect

**Solution**: Update remote URL

### Getting Help

1. **Check Git Status**: `git status`
2. **View Recent Commits**: `git log --oneline -10`
3. **Check Configuration**: `git config --local --list`
4. **Review Documentation**: This file and `README.md`
5. **Git Help**: `git help <command>`

---

## Remote Collaboration Guidelines

### For Team Members

1. **Communication**: Inform team before force pushing to master
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

✅ **Configured**: Local-first git workflow  
✅ **Policy**: Local is always source of truth  
✅ **Remote**: https://github.com/mrbestnaija/portofolio_maximizer.git  
✅ **Settings**: `pull.rebase=false`, `push.default=current`, `merge.ff=false`  
✅ **Conflicts**: Always resolve with `git checkout --ours`  
✅ **Force Push**: Allowed and encouraged with `--force-with-lease`  
✅ **Testing**: Run tests before push  
✅ **Documentation**: Update docs with architectural changes

### Recent Sync Status

- **Last Push**: 2025-11-06
- **Last Commit**: 6839f0b (refactor(pipeline): Implement remote synchronization enhancements)
- **Status**: All local changes synced to GitHub ✅



---

**Document Version**: 2.0  
**Last Updated**: 2025-11-06  
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

### Recent Sync Status

- **Last Push**: 2025-11-06
- **Last Commit**: 6839f0b (refactor(pipeline): Implement remote synchronization enhancements)
- **Status**: All local changes synced to GitHub o. *(New work since then includes SQLite self-healing, MSSA/SARIMAX/visualization fixes, test-DB isolation, and documentation updatescommit and push as described below.)*

### Quick Reference

```bash
# Daily workflow (docs + code)
git status
git add Documentation/implementation_checkpoint.md Documentation/arch_tree.md Documentation/BRUTAL_TEST_README.md Documentation/FORECASTING_IMPLEMENTATION_SUMMARY.md bash/comprehensive_brutal_test.sh
git commit -m "docs(brutal+forecasting): Sync docs with DB/test isolation fixes"
git push origin master

# If conflicts
git checkout --ours .
git add .
git commit -m "merge: Resolve conflicts"

# If remote diverged
git push --force-with-lease origin master

# Verify
git status
git log --oneline -5
```
