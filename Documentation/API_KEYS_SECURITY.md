# API Keys Security Guide

**Version**: 1.0
**Date**: 2025-10-07
**Status**: ACTIVE

---

## Security Status

✅ **API Keys Protected**
- All API keys stored in `.env` file
- `.env` is in `.gitignore` (never committed to git)
- Template file (`.env.template`) provided for reference
- Configurations read from environment variables only

---

## Current API Keys Configuration

### 1. Alpha Vantage API Key

**Environment Variable**: `ALPHA_VANTAGE_API_KEY`

**Location**: `.env` (line 6)

**Configuration Reference**:
- `config/alpha_vantage_config.yml` - Line 14: `api_key_env: "ALPHA_VANTAGE_API_KEY"`

**Usage**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
```

**API Limits**:
- Free Tier: 5 calls/minute, 500 calls/day
- Premium Tier: 75 calls/minute (if upgraded)

**Get Your Key**: https://www.alphavantage.co/support/#api-key

---

### 2. Finnhub API Key

**Environment Variable**: `FINNHUB_API_KEY`

**Location**: `.env` (line 9)

**Configuration Reference**:
- `config/finnhub_config.yml` - Line 14: `api_key_env: "FINNHUB_API_KEY"`

**Usage**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FINNHUB_API_KEY')
```

**API Limits**:
- Free Tier: 60 calls/minute
- Premium Tier: 300 calls/minute (if upgraded)

**Get Your Key**: https://finnhub.io/register

**Additional Variables**:
- `FINNHUB_EMAIL` - Your registered email
- `FINNHUB_WEBHOOK_SECRET` - For webhook authentication

---

### 3. Anthropic API Key (Claude)

**Environment Variable**: `ANTHROPIC_API_KEY`

**Location**: `.env` (line 2)

**Usage**: Claude Code CLI integration

**Get Your Key**: https://console.anthropic.com/

---

## Security Best Practices

### ✅ DO:

1. **Keep `.env` Local**
   - Never commit `.env` to git
   - Always verify: `git check-ignore -v .env` (should show ignored)

2. **Use Environment Variables**
   - Load with `python-dotenv`: `load_dotenv()`
   - Access with `os.getenv('KEY_NAME')`

3. **Use `.env.template`**
   - Commit template without actual keys
   - Document required variables
   - Provide setup instructions

4. **Rotate Keys Regularly**
   - Change keys every 90 days
   - Update `.env` file locally
   - No code changes needed

5. **Verify Before Push**
   ```bash
   # Always check before git push
   git status --short | grep .env
   # Should return nothing (file ignored)

   git diff --cached | grep -i "api.*key"
   # Should not show any API keys
   ```

### ❌ DON'T:

1. **Never Hardcode Keys**
   ```python
   # ❌ BAD - Never do this
   API_KEY = 'sk-ant-api03-...'

   # ✅ GOOD - Use environment variables
   API_KEY = os.getenv('ANTHROPIC_API_KEY')
   ```

2. **Never Commit .env**
   ```bash
   # If accidentally staged
   git reset .env

   # If accidentally committed
   git rm --cached .env
   git commit -m "chore: Remove .env from git tracking"
   ```

3. **Never Share Keys Publicly**
   - Don't paste in chat/forums
   - Don't include in screenshots
   - Don't email unencrypted

4. **Never Use Keys in Logs**
   ```python
   # ❌ BAD - Key might appear in logs
   logger.info(f"Using API key: {api_key}")

   # ✅ GOOD - Log without exposing key
   logger.info("API key loaded successfully")
   ```

---

## Setup Instructions

### For New Developers

1. **Copy Template**
   ```bash
   cp .env.template .env
   ```

2. **Add Your Keys**
   ```bash
   # Edit .env and replace placeholders
   nano .env
   # or
   vim .env
   ```

3. **Verify Security**
   ```bash
   # Ensure .env is ignored
   git check-ignore -v .env
   # Expected output: .gitignore:3:*.env	.env

   # Verify not tracked
   git ls-files | grep "\.env$"
   # Should return nothing
   ```

4. **Test Configuration**
   ```bash
   # Run a simple test
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ Loaded' if os.getenv('ALPHA_VANTAGE_API_KEY') else '✗ Failed')"
   ```

---

## Git Configuration Verification

### Check .gitignore

```bash
cat .gitignore | grep -A5 "Sensitive files"
```

**Expected Output**:
```
# Sensitive files and credentials
.env
*.env
*.key
*.pem
*.p12
.env.*
```

### Verify Files Not Tracked

```bash
# Should return nothing
git ls-files | grep -E "\.(env|key|pem)$"

# Check status
git status --ignored | grep .env
```

---

## Emergency Procedures

### Scenario 1: API Key Accidentally Committed

```bash
# Step 1: Remove from git history (if just committed)
git reset HEAD~1
git reset .env

# Step 2: Verify removal
git log --all --full-history -- .env

# Step 3: Force push to remote (if already pushed)
git push --force-with-lease origin master

# Step 4: Rotate compromised keys immediately
# - Go to API provider dashboard
# - Generate new keys
# - Update .env with new keys
```

### Scenario 2: Key Exposed Publicly

1. **Immediate Action**: Revoke/rotate key at provider
2. **Alpha Vantage**: Login → https://www.alphavantage.co/support/#api-key → Regenerate
3. **Finnhub**: Login → https://finnhub.io/dashboard → Regenerate API Key
4. **Anthropic**: Login → https://console.anthropic.com/ → Manage Keys → Revoke & Create New

### Scenario 3: Lost .env File

```bash
# Recreate from template
cp .env.template .env

# Add keys from your secure password manager
# or regenerate at each provider's dashboard
```

---

## Configuration File Architecture

### How Keys Flow Through System

```
.env (LOCAL ONLY - NOT IN GIT)
    ↓
python-dotenv loads environment variables
    ↓
config/*.yml references via `api_key_env`
    ↓
etl/*_extractor.py reads from config
    ↓
API calls use loaded keys
```

### Example: Alpha Vantage Flow

1. **`.env`**: `ALPHA_VANTAGE_API_KEY='API_KEY'`
2. **`config/alpha_vantage_config.yml`**:
   ```yaml
   authentication:
     api_key_env: "ALPHA_VANTAGE_API_KEY"
   ```
3. **`etl/alpha_vantage_extractor.py`**:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   config = yaml.safe_load('config/alpha_vantage_config.yml')
   api_key = os.getenv(config['authentication']['api_key_env'])
   ```

---

## Verification Checklist

Before any git operation:

- [ ] `.env` file exists and contains keys
- [ ] `.env` is in `.gitignore`
- [ ] `.env` is NOT tracked by git: `git ls-files | grep "\.env$"` returns nothing
- [ ] `.env.template` is tracked (template without real keys)
- [ ] No hardcoded keys in source code
- [ ] Config files reference environment variables only
- [ ] All tests pass with loaded environment variables

---

## Monitoring & Auditing

### Check for Exposed Keys

```bash
# Search codebase for potential key exposures
grep -r "sk-ant-api03" . --exclude-dir={.git,simpleTrader_env,node_modules}
grep -r "api.*key.*=" . --include="*.py" --exclude-dir={.git,simpleTrader_env}

# Should only find references in .env (which is ignored)
```

### Audit Git History

```bash
# Check if .env was ever committed
git log --all --full-history --source --all -- .env
# Should return nothing

# Check for accidentally committed keys
git log --all -p | grep -i "api.*key"
```

---

## Summary

✅ **Current Status**:
- All API keys in `.env` (locally protected)
- `.env` in `.gitignore` (never committed)
- `.env.template` created (for reference)
- Config files use environment variables
- No hardcoded keys in source code

✅ **Active Keys**:
- Alpha Vantage API Key (configured)
- Finnhub API Key (configured)
- Anthropic API Key (configured)

✅ **Security Verified**:
- `git check-ignore .env` → Confirmed ignored
- `git ls-files | grep .env` → Not tracked
- No keys in git history

**Status**: 🔒 **SECURE** - All API keys properly protected

---

**Document Version**: 1.0
**Last Updated**: 2025-10-07
**Next Review**: Monthly or before sharing repository
