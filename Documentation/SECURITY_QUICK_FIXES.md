# Security Quick Fixes - Immediate Actions

**Date**: 2025-01-27  
**Priority**: 🔴 **CRITICAL - Execute Immediately**

---

## 🚨 IMMEDIATE ACTIONS (Do Today)

### 1. **Disable Jupyter Notebook in Production** (5 minutes)

**File**: `docker-compose.yml`

**Action**: Comment out or remove the Jupyter service:

```yaml
# portfolio-notebook:
#   extends: portfolio-maximizer
#   image: portfolio-maximizer:v4.5-notebook
#   container_name: portfolio_notebook
#   
#   ports:
#     - "8888:8888"
#   ...
```

**Why**: Jupyter is exposed without authentication - **CRITICAL SECURITY RISK**

---

### 2. **Add Database File Permissions** (10 minutes)

**File**: `etl/database_manager.py`

**Add after line 62**:
```python
# Set secure file permissions
import os
if self.db_path.exists():
    os.chmod(self.db_path, 0o600)  # Read/write for owner only
else:
    # Create with secure permissions
    self.db_path.touch()
    os.chmod(self.db_path, 0o600)
```

**Why**: Prevents unauthorized database access on shared systems

---

### 3. **Create Error Sanitization Utility** (15 minutes)

**New File**: `etl/security_utils.py`

```python
"""Security utilities for error handling and sanitization."""
import os
import logging

logger = logging.getLogger(__name__)

def sanitize_error(exc: Exception, is_production: bool = None) -> str:
    """
    Sanitize error messages for production.
    
    Args:
        exc: Exception object
        is_production: Whether running in production mode
        
    Returns:
        Safe error message
    """
    if is_production is None:
        is_production = os.getenv('PORTFOLIO_ENV', 'development') == 'production'
    
    if is_production:
        # Generic error message for users
        logger.error(f"Internal error: {exc}", exc_info=True)
        return "An error occurred. Please contact support."
    else:
        # Detailed error for development
        return str(exc)
```

**Update error handlers**:
```python
from etl.security_utils import sanitize_error

try:
    # ... code ...
except Exception as e:
    safe_error = sanitize_error(e, is_production=True)
    logger.error(f"Failed operation: {safe_error}")
```

**Why**: Prevents information leakage in error messages

---

### 4. **Move Secrets to Docker Secrets** (30 minutes)

**File**: `docker-compose.yml`

**Instead of**:
```yaml
environment:
  - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
  - FINNHUB_API_KEY=${FINNHUB_API_KEY}
```

**Use**:
```yaml
secrets:
  - alpha_vantage_api_key
  - finnhub_api_key

secrets:
  alpha_vantage_api_key:
    file: ./secrets/alpha_vantage_api_key.txt
  finnhub_api_key:
    file: ./secrets/finnhub_api_key.txt
```

**Create secrets directory**:
```bash
mkdir -p secrets
chmod 700 secrets
echo "YOUR_API_KEY" > secrets/alpha_vantage_api_key.txt
chmod 600 secrets/*.txt
```

**Add to `.gitignore`**:
```
secrets/
*.txt
```

**Why**: Secrets in environment variables are visible in process lists

---

### 5. **Add Security Headers** (if using web interface) (20 minutes)

**Create**: `scripts/security_middleware.py`

```python
"""Security middleware for web applications."""
from functools import wraps

def add_security_headers(response):
    """Add security headers to HTTP response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
```

**Why**: Protects against common web vulnerabilities

---

## 📋 THIS WEEK CHECKLIST

- [ ] Disable Jupyter notebook
- [ ] Add database file permissions
- [ ] Create error sanitization
- [ ] Move secrets to Docker secrets
- [ ] Review and update `.gitignore`
- [ ] Audit all log statements for sensitive data
- [ ] Review all environment variable usage
- [ ] Check for hardcoded credentials
- [ ] Review Dockerfile security
- [ ] Add security headers (if web interface exists)

---

## 🔍 SECURITY AUDIT COMMANDS

Run these to check for common issues:

```bash
# Check for hardcoded secrets
grep -r "password\s*=" --include="*.py" .
grep -r "api_key\s*=" --include="*.py" .
grep -r "secret\s*=" --include="*.py" .

# Check for SQL injection risks (should find parameterized queries only)
grep -r "execute.*%" --include="*.py" .

# Check for dangerous functions
grep -r "eval\|exec\|__import__" --include="*.py" .

# Check for exposed ports
grep -r "ports:" docker-compose.yml

# Check for secrets in docker-compose
grep -r "API_KEY\|PASSWORD\|SECRET" docker-compose.yml
```

**Windows PowerShell equivalents**:
```powershell
# Check for hardcoded secrets
Select-String -Path "*.py" -Pattern "password\s*=" -Recursive
Select-String -Path "*.py" -Pattern "api_key\s*=" -Recursive
Select-String -Path "*.py" -Pattern "secret\s*=" -Recursive

# Check for SQL injection risks
Select-String -Path "*.py" -Pattern "execute.*%" -Recursive

# Check for dangerous functions
Select-String -Path "*.py" -Pattern "eval|exec|__import__" -Recursive

# Check for exposed ports
Select-String -Path "docker-compose.yml" -Pattern "ports:"

# Check for secrets in docker-compose
Select-String -Path "docker-compose.yml" -Pattern "API_KEY|PASSWORD|SECRET"
```

---

## 📚 NEXT STEPS

After completing quick fixes, proceed to:

1. **Authentication System** (Phase 1)
2. **Encryption** (Phase 1)
3. **Rate Limiting** (Phase 2)
4. **Monitoring** (Phase 2)

See `SECURITY_AUDIT_AND_HARDENING.md` for complete roadmap.

---

## ⚠️ IMPORTANT NOTES

1. **Do NOT deploy to production** until at least items 1-4 are completed
2. **Test all changes** in development environment first
3. **Backup database** before making permission changes
4. **Document all changes** for future reference
5. **Review git history** to ensure no secrets were committed

## Secrets/PAT Leak Guard (Recommended)

This repo includes a local guard to prevent accidentally staging credentials (PATs, API keys, passwords, private keys) and to detect credential-bearing git remote URLs.

```bash
# One-time setup
pre-commit install
pre-commit install --hook-type pre-push

# Manual scan (recommended before any push)
python tools/secrets_guard.py scan --staged
```

---

**Status**: 🔴 **ACTION REQUIRED - Complete before production deployment**

