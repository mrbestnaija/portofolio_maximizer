# Git Workflow - Local as Source of Truth

**Policy**: Local repository is ALWAYS the source of truth  
**Last Updated**: 2025-11-06  
**Status**: ENFORCED  
**Version**: 2.0

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

```bash
# 1. Clone the repository
git clone https://github.com/mrbestnaija/portofolio_maximizer.git
cd portofolio_maximizer

# 2. Configure git settings (one-time setup)
git config --local pull.rebase false
git config --local push.default current
git config --local merge.ff false

# 3. Verify configuration
git config --local --list | grep -E "(pull|push|merge)"

# 4. Check repository status
git status
git remote -v
```

### For Existing Contributors

```bash
# Verify your setup is current
git config --local --list | grep -E "(pull|push|merge)"

# Expected output:
# pull.rebase=false
# push.default=current
# merge.ff=false
```

---

## Remote Repository

**GitHub Repository**: https://github.com/mrbestnaija/portofolio_maximizer.git  
**Primary Branch**: `master`  
**Last Sync**: 2025-11-06 (Commit 6839f0b)

### Remote Configuration

```bash
# View current remote setup
git remote -v

# Expected output:
# origin  https://github.com/mrbestnaija/portofolio_maximizer.git (fetch)
# origin  https://github.com/mrbestnaija/portofolio_maximizer.git (push)
```

### Adding Upstream (For Contributors)

```bash
# Add upstream remote (if contributing from fork)
git remote add upstream https://github.com/mrbestnaija/portofolio_maximizer.git

# Verify remotes
git remote -v
```

---

## Git Configuration

The following git settings ensure local changes always take precedence and maintain a clear history:

```bash
# Apply configuration (run once per repository)
git config --local pull.rebase false      # Merge, don't rebase on pull
git config --local push.default current   # Push current branch only
git config --local merge.ff false         # Always create merge commits
```

### What These Settings Do

| Setting | Value | Effect | Why |
|---------|-------|--------|-----|
| `pull.rebase` | `false` | Pull creates merge commits, preserving local history | Maintains complete development timeline |
| `push.default` | `current` | Only push current branch to remote | Prevents accidental pushes of other branches |
| `merge.ff` | `false` | Always create merge commit (no fast-forward) | Clear merge history, easier to track changes |

### Verification

```bash
# Check configuration
git config --local --list | grep -E "(pull|push|merge)"

# Expected output:
# pull.rebase=false
# push.default=current
# merge.ff=false
```

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

```bash
# Edit files normally
# (Use your preferred editor: vim, VS Code, etc.)

# Stage specific files
git add path/to/file.py

# Or stage all changes
git add .

# Commit with descriptive message
git commit -m "feat(etl): Add intelligent caching to yfinance extractor"
```

#### 2. Sync to Remote (Backup)

```bash
# Standard push (if no conflicts)
git push origin master

# If remote has diverged, use force-with-lease (safer than --force)
git push --force-with-lease origin master

# Verify push succeeded
git log --oneline -1
```

**Note**: `--force-with-lease` is safer than `--force` as it checks if remote has changed since last fetch.

#### 3. Pull from Remote (When Needed)

```bash
# Fetch latest changes without merging
git fetch origin

# View what changed
git log HEAD..origin/master

# Pull and merge
git pull origin master

# If conflicts occur, see [Conflict Resolution](#conflict-resolution)
```

---

## Contributor Workflow

### For External Contributors (Pull Requests)

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/portofolio_maximizer.git
cd portofolio_maximizer

# 3. Add upstream remote
git remote add upstream https://github.com/mrbestnaija/portofolio_maximizer.git

# 4. Create feature branch
git checkout -b feature/your-feature-name

# 5. Make changes and commit
git add .
git commit -m "feat(scope): Your feature description"

# 6. Push to your fork
git push origin feature/your-feature-name

# 7. Open Pull Request on GitHub
# GitHub will provide the PR link after push
```

### Keeping Fork Up to Date

```bash
# Fetch latest from upstream
git fetch upstream

# Merge upstream/master into your branch
git checkout master
git merge upstream/master

# Push updated master to your fork
git push origin master
```

### For Team Members (Direct Access)

```bash
# 1. Clone repository
git clone https://github.com/mrbestnaija/portofolio_maximizer.git
cd portofolio_maximizer

# 2. Configure git (one-time)
git config --local pull.rebase false
git config --local push.default current
git config --local merge.ff false

# 3. Create feature branch (recommended)
git checkout -b feature/your-feature

# 4. Make changes, test, commit
git add .
git commit -m "feat(scope): Feature description"

# 5. Push feature branch
git push origin feature/your-feature

# 6. Merge to master locally (if approved)
git checkout master
git merge --no-ff feature/your-feature

# 7. Push master
git push origin master
```

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

```bash
# During merge conflict
git checkout --ours <file>
git add <file>
git commit -m "merge: Resolve conflicts, keep local changes"
```

#### Method 2: Keep Remote Version

```bash
# If you want remote version instead
git checkout --theirs <file>
git add <file>
git commit -m "merge: Resolve conflicts, accept remote changes"
```

#### Method 3: Manual Resolution

```bash
# 1. Open conflicted file in editor
# 2. Look for conflict markers:
#    <<<<<<< HEAD
#    (local changes)
#    =======
#    (remote changes)
#    >>>>>>> origin/master
# 3. Edit to resolve conflict
# 4. Remove conflict markers
# 5. Stage and commit
git add <file>
git commit -m "merge: Manually resolve conflicts"
```

### Conflict Prevention

```bash
# Before starting work, fetch latest
git fetch origin

# Check if local is behind
git status

# Update local if needed
git pull origin master

# Then start your work
```

---

## Testing & Verification

### Pre-Commit Checklist

Before committing, verify:

```bash
# 1. Run tests
pytest tests/ -v

# 2. Check for linting errors
# (If using flake8, black, etc.)
# flake8 .
# black --check .

# 3. Verify imports work
python -c "from scripts.run_etl_pipeline import execute_pipeline; print('✓ Imports OK')"

# 4. Check git status
git status

# 5. Review changes
git diff --cached
```

### Pre-Push Checklist

Before pushing to remote:

```bash
# 1. Run full test suite
pytest tests/ --cov=etl

# 2. Verify no sensitive data
git diff origin/master | grep -i "password\|api_key\|secret"

# 3. Check commit messages are clear
git log origin/master..HEAD

# 4. Verify branch is up to date (optional)
git fetch origin
git log HEAD..origin/master
```

### Automated Testing (Recommended)

Create a pre-push hook (optional):

```bash
# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "Running pre-push checks..."
pytest tests/ -v --tb=short || exit 1
echo "✓ All checks passed"
EOF

chmod +x .git/hooks/pre-push
```

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

```bash
# Create and switch to feature branch
git checkout -b feature/portfolio-optimization

# Work on feature
git add .
git commit -m "feat(portfolio): Add Markowitz optimization"

# Push feature branch
git push origin feature/portfolio-optimization

# Merge to master (locally)
git checkout master
git merge --no-ff feature/portfolio-optimization

# Push master
git push origin master

# Clean up feature branch (optional)
git branch -d feature/portfolio-optimization
git push origin --delete feature/portfolio-optimization
```

### Backup Branches

```bash
# Create backup before major changes
git branch backup/before-refactor-$(date +%Y%m%d)

# List all backup branches
git branch --list 'backup/*'

# Restore from backup if needed
git checkout backup/before-refactor-20251106
git checkout -b recovery-from-backup
```

---

## Backup Strategy

### Remote as Backup

- ✅ Remote repository (GitHub) serves as backup
- ✅ Local always has latest, most accurate code
- ✅ Push frequency: After each significant milestone or daily
- ✅ Force push allowed with `--force-with-lease` for safety

### Local Backups

```bash
# Create timestamped backup branch
git branch backup/before-refactor-$(date +%Y%m%d_%H%M%S)

# Create backup tag
git tag backup-v1.0-$(date +%Y%m%d)

# List backups
git branch --list 'backup/*'
git tag --list 'backup-*'

# Push backup branches/tags (optional)
git push origin backup/before-refactor-20251106
git push origin backup-v1.0-20251106
```

### Backup Best Practices

1. **Before Major Refactors**: Create backup branch
2. **Before Force Push**: Create backup tag
3. **Regular Pushes**: Push to remote after each milestone
4. **Documentation**: Document backup strategy in commit messages

---

## Emergency Recovery

### Scenario 1: Accidental Remote Override

**Situation**: Remote was accidentally updated, need to restore local version

```bash
# Verify local has correct version
git log --oneline -5

# Force push local to remote (overwrites remote)
git push --force-with-lease origin master

# Verify push succeeded
git log --oneline -5
```

### Scenario 2: Local Repository Corruption

**Situation**: Local git repository is corrupted or lost

```bash
# 1. Clone fresh copy from remote
cd /path/to/parent/directory
git clone https://github.com/mrbestnaija/portofolio_maximizer.git portfolio_maximizer_recovery

# 2. Copy important local files (if any) before replacing
cp -r portfolio_maximizer_v45/local_changes/* portfolio_maximizer_recovery/

# 3. Replace corrupted repository
rm -rf portfolio_maximizer_v45
mv portfolio_maximizer_recovery portfolio_maximizer_v45

# 4. Reconfigure git settings
cd portfolio_maximizer_v45
git config --local pull.rebase false
git config --local push.default current
git config --local merge.ff false
```

### Scenario 3: Need to Inspect Remote Changes

**Situation**: Want to see what changed on remote without merging

```bash
# Fetch remote changes
git fetch origin

# View remote changes
git log master..origin/master --oneline

# See detailed diff
git diff master origin/master

# View specific file changes
git diff master origin/master -- path/to/file.py

# Merge only if you want remote changes
git merge origin/master
```

### Scenario 4: Undo Last Commit (Before Push)

```bash
# Undo last commit, keep changes
git reset --soft HEAD~1

# Undo last commit, discard changes (careful!)
git reset --hard HEAD~1

# Verify
git status
```

### Scenario 5: Undo Last Commit (After Push)

```bash
# Create revert commit (recommended)
git revert HEAD
git push origin master

# Or force push (only if no one else has pulled)
git reset --hard HEAD~1
git push --force-with-lease origin master
```

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

```bash
# Safe: Checks remote hasn't changed
git push --force-with-lease origin master

# Dangerous: Overwrites remote without checking
# git push --force origin master  # ❌ Avoid unless necessary
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

```bash
# Check what would be ignored
git status --ignored

# Test if file is ignored
git check-ignore -v path/to/file

# View .gitignore rules
cat .gitignore
```

---

## Verification Commands

### Check Git Configuration

```bash
# View all local git config
git config --local --list

# Check specific settings
git config --local --get pull.rebase
git config --local --get push.default
git config --local --get merge.ff

# Expected values:
# pull.rebase=false
# push.default=current
# merge.ff=false
```

### Check Repository Status

```bash
# Full status
git status

# Short status
git status --short

# View recent commits
git log --oneline -10

# View remote tracking
git remote -v

# Check branch information
git branch -vv
```

### Check Uncommitted Changes

```bash
# Unstaged changes
git diff

# Staged changes
git diff --cached

# All changes (staged + unstaged)
git diff HEAD

# Changes in specific file
git diff path/to/file.py

# Summary of changes
git diff --stat
```

### Check Commit History

```bash
# Recent commits
git log --oneline -10

# Detailed recent commit
git log -1

# Commits not yet pushed
git log origin/master..HEAD

# Commits in remote not in local
git log HEAD..origin/master

# Visual graph
git log --oneline --graph --all -10
```

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

3. **Use Safe Force Push**: Always `--force-with-lease`
   ```bash
   git push --force-with-lease origin master
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

```bash
# Feature with scope
git commit -m "feat(etl): Add intelligent caching to yfinance extractor"

# Bug fix
git commit -m "fix(preprocessor): Handle missing data edge cases"

# Refactoring
git commit -m "refactor(pipeline): Extract testable execute_pipeline() function"

# Documentation
git commit -m "docs: Update README with Ollama prerequisite documentation"

# Test
git commit -m "test(etl): Add comprehensive cache mechanism tests"

# Multi-line commit message
git commit -m "feat(etl): Add timestamp and run_id to parquet filenames

- Include timestamp in filename to prevent overwrites
- Add run_id parameter for unique run identification
- Persist comprehensive metadata alongside artifacts
- Maintain backward compatibility with existing code"
```

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
```bash
git push origin master
```

#### Issue 2: "Your branch and 'origin/master' have diverged"

**Solution**: Use force-with-lease (local takes priority)
```bash
git push --force-with-lease origin master
```

#### Issue 3: Merge conflicts during pull

**Solution**: Keep local version
```bash
git checkout --ours .
git add .
git commit -m "merge: Resolve conflicts, keep local changes"
```

#### Issue 4: Accidentally committed sensitive data

**Solution**: Remove from history (before push)
```bash
# Remove file from last commit
git reset --soft HEAD~1
git reset HEAD sensitive_file.txt
git commit -m "feat: Your original message"

# Add to .gitignore
echo "sensitive_file.txt" >> .gitignore
git add .gitignore
git commit -m "chore: Add sensitive_file.txt to .gitignore"
```

#### Issue 5: Want to undo last commit (not pushed)

**Solution**: Reset to previous commit
```bash
# Keep changes
git reset --soft HEAD~1

# Discard changes (careful!)
git reset --hard HEAD~1
```

#### Issue 6: Git configuration not applied

**Solution**: Reapply configuration
```bash
git config --local pull.rebase false
git config --local push.default current
git config --local merge.ff false

# Verify
git config --local --list | grep -E "(pull|push|merge)"
```

#### Issue 7: Remote URL incorrect

**Solution**: Update remote URL
```bash
# View current remote
git remote -v

# Update remote URL
git remote set-url origin https://github.com/mrbestnaija/portofolio_maximizer.git

# Verify
git remote -v
```

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

```bash
# 1. Before starting work
git fetch origin
git status

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Make changes, test, commit
# ... make changes ...
pytest tests/
git add .
git commit -m "feat(scope): Description"

# 4. Push feature branch
git push origin feature/your-feature

# 5. (For PRs) Open Pull Request on GitHub
# (For direct access) Merge to master after review
git checkout master
git merge --no-ff feature/your-feature
git push origin master
```

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

### Quick Reference

```bash
# Daily workflow
git add .
git commit -m "type(scope): Description"
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

```bash
# Standard git commands
git status
git add .
git commit -m "feat: Description"

# Date formatting for backup branches
git branch backup/before-refactor-$(date +%Y%m%d)
```

### macOS

```bash
# Standard git commands (same as Linux)
git status
git add .
git commit -m "feat: Description"

# Date formatting
git branch backup/before-refactor-$(date +%Y%m%d)
```

---

**End of Document**
