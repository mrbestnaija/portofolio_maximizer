# Git Workflow - Local as Source of Truth

**Policy**: Local repository is ALWAYS the source of truth
**Last Updated**: 2025-10-04
**Status**: ENFORCED

---

## Remote Repository

**GitHub Fork**: https://github.com/mrbestnaija/portofolio_maximizer.git
**Branch**: master
**Last Sync**: 2025-10-04 (Commit fc07d06)

### Remote Configuration
```bash
# Current remote setup
origin  https://github.com/mrbestnaija/portofolio_maximizer.git (fetch)
origin  https://github.com/mrbestnaija/portofolio_maximizer.git (push)
```

---

## Git Configuration

The following git settings ensure local changes always take precedence:

```bash
# Applied to this repository
git config --local pull.rebase false      # Merge, don't rebase on pull
git config --local push.default current   # Push current branch only
git config --local merge.ff false         # Always create merge commits
```

### What These Settings Do

| Setting | Value | Effect |
|---------|-------|--------|
| `pull.rebase` | `false` | Pull creates merge commits, preserving local history |
| `push.default` | `current` | Only push current branch to remote |
| `merge.ff` | `false` | Always create merge commit (no fast-forward) |

---

## Workflow Principles

### 1. **Local Changes Take Priority**
- All development happens locally first
- Local commits are never rebased or amended after creation
- Remote is a backup/sync point, not the primary workspace

### 2. **Conflict Resolution**
When conflicts occur during pull:
```bash
# Always choose local version
git checkout --ours <file>
git add <file>
git commit
```

### 3. **Force Push Policy**
**ALLOWED** - Since local is source of truth:
```bash
# Update remote to match local exactly
git push --force-with-lease origin master
```

**Note**: `--force-with-lease` is safer than `--force` as it checks remote hasn't changed.

---

## Common Operations

### Initial Setup (Already Done)
```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
git config --local pull.rebase false
git config --local push.default current
git config --local merge.ff false
```

### Daily Workflow

#### 1. Make Local Changes
```bash
# Edit files normally
vim etl/yfinance_extractor.py

# Stage and commit locally
git add etl/yfinance_extractor.py
git commit -m "feat: Add cache invalidation logic"
```

#### 2. Sync to Remote (Backup)
```bash
# Push local to remote (overwrites remote if needed)
git push origin master

# If remote has diverged, force local version
git push --force-with-lease origin master
```

#### 3. Pull from Remote (Rare)
```bash
# Only if you need to pull from another machine
git pull origin master

# If conflicts, always choose local
git checkout --ours .
git add .
git commit -m "merge: Resolve conflicts, keep local changes"
```

---

## Branch Strategy

### Current Setup
- **Single Branch**: `master`
- **No Main Branch**: Not configured (local development only)
- **Remote Tracking**: Optional, local is definitive

### Future Multi-Branch Setup
If needed:
```bash
# Create feature branch
git checkout -b feature/portfolio-optimization

# Work locally
git commit -m "feat: Add Markowitz optimization"

# Push feature branch
git push origin feature/portfolio-optimization

# Merge to master (locally)
git checkout master
git merge --no-ff feature/portfolio-optimization

# Push master
git push origin master
```

---

## Backup Strategy

### Remote as Backup
- Remote repository (GitHub/GitLab) serves as backup
- Local always has latest, most accurate code
- Push frequency: After each significant milestone

### Local Backups
Additional safety:
```bash
# Create local backup branch
git branch backup/before-refactor-$(date +%Y%m%d)

# List backups
git branch --list 'backup/*'
```

---

## Emergency Recovery

### Scenario 1: Accidental Remote Override
```bash
# Remote was accidentally updated, restore from local
git push --force origin master
```

### Scenario 2: Local Corruption
```bash
# If local git repo corrupted, restore from GitHub backup
cd /mnt/c/Users/Bestman/personal_projects/
git clone https://github.com/mrbestnaija/portofolio_maximizer.git portfolio_maximizer_recovery
# Copy files from backup to main directory
cp -r portfolio_maximizer_recovery/* portfolio_maximizer_v45/
```

### Scenario 3: Need Remote Changes
```bash
# Fetch remote without merging
git fetch origin

# Inspect remote changes
git diff master origin/master

# Decide: merge or ignore
git merge origin/master  # Only if you want remote changes
# OR
# Ignore and keep local
```

---

## .gitignore Strategy

Configured to ignore:
- Generated data files (`data/raw/*.parquet`, `data/processed/*.parquet`)
- Virtual environments (`simpleTrader_env/`, `portfolio_env/`)
- Cache files (`__pycache__/`, `.pytest_cache/`)
- Sensitive files (`.env`, `*.key`)
- Large model files (`*.pkl`, `*.h5`)

Keep in version control:
- Source code (`etl/`, `scripts/`, `tests/`)
- Configuration templates (`config/*.yml`)
- Documentation (`Documentation/`)
- Empty directory markers (`.gitkeep` files)

---

## Verification Commands

### Check Git Configuration
```bash
git config --local --list | grep -E "(pull|push|merge)"
```

Expected output:
```
pull.rebase=false
push.default=current
merge.ff=false
```

### Check Repository Status
```bash
git status
git log --oneline -5
git remote -v
```

### Check Uncommitted Changes
```bash
git diff          # Unstaged changes
git diff --cached # Staged changes
git status --short
```

---

## Best Practices

1. **Commit Often**: Small, atomic commits locally
2. **Descriptive Messages**: Follow conventional commits (feat, fix, refactor, etc.)
3. **Test Before Push**: Run test suite before pushing to remote
4. **Document Changes**: Update Documentation/ when architecture changes
5. **Backup Branches**: Create backup branches before major refactors

---

## Conventional Commit Messages

Follow this format:
```
<type>(<scope>): <subject>

<body>

🤖 Generated with co-companion 

Authored-By: Bestman Ezekwu Enock
Email : csgtmalice@protonmail.ch

```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Add/update tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `perf`: Performance improvements

Examples:
```bash
git commit -m "feat(etl): Add intelligent caching to yfinance extractor"
git commit -m "fix(preprocessor): Handle missing data edge cases"
git commit -m "refactor: Streamline architecture to match v3.0 specification"
git commit -m "test(etl): Add comprehensive cache mechanism tests"
```

---

## Summary

✅ **Configured**: Local-first git workflow
✅ **Policy**: Local is always source of truth
✅ **Remote**: https://github.com/mrbestnaija/portofolio_maximizer.git
✅ **Settings**: pull.rebase=false, push.default=current, merge.ff=false
✅ **Conflicts**: Always resolve with `git checkout --ours`
✅ **Force Push**: Allowed and encouraged with `--force-with-lease`

### Recent Sync Status
- **Last Push**: 2025-10-04
- **Commits Pushed**: 6 commits (fc07d06 to d4db60c)
- **Force Update**: Yes (377b63e → fc07d06)
- **Status**: All local changes synced to GitHub ✅

**Document Version**: 1.1
**Status**: ACTIVE
**Review**: Monthly or before major changes
