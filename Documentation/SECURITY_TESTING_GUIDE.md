# Security Testing Guide

**Date**: 2025-01-27  
**Status**: ✅ **Tests Created & Integrated**

---

## 🧪 QUICK START

### Run All Security Tests

```bash
# Run all security-marked tests
pytest -m security -v

# Run specific test file
pytest tests/etl/test_security_utils.py -v
pytest tests/etl/test_secret_loader.py -v
pytest tests/etl/test_database_security.py -v
pytest tests/integration/test_security_integration.py -v

# Use test runner script
python tests/run_security_tests.py
```

### Secret Leak Guard (Recommended Before Any Push)

This repo includes a local secrets/PAT guard that blocks accidentally staging credentials and also detects credential-bearing git remote URLs.

```bash
# Scan staged changes only (fast, best signal)
python tools/secrets_guard.py scan --staged

# Strict mode (also fails on WARN)
python tools/secrets_guard.py scan --staged --strict

# CI / deep check: scan tracked files for high-confidence token patterns
python tools/secrets_guard.py scan --tracked --strict
```

### Autonomous Agent Guard Tests (OpenClaw)

This repo now includes an OpenClaw autonomous-action guard in `utils/openclaw_cli.py`.
It blocks high-risk autonomous requests unless trusted approval is passed for the specific run,
and it always prepends an anti-prompt-injection policy to agent turns.

```bash
# Focused autonomy guard tests
pytest tests/utils/test_openclaw_cli.py -k "autonomy_guard or AutonomyGuard" -v

# Validate OpenClaw config/env hardening alignment
python scripts/verify_openclaw_config.py
```

High-value assertions:
- High-risk requests (credential exfiltration, irreversible trade/account actions) return `403` without trusted approval.
- Same requests pass only when `--approve-high-risk` or `OPENCLAW_APPROVE_HIGH_RISK=1` is set for that run.
- Agent-turn prompts always include policy prefix `[PMX_AUTONOMY_POLICY]`; user-supplied policy markers are inert.
- Optional strict mode (`OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS=1`) blocks prompt-injection phrases.

---

## 📋 TEST SUITE OVERVIEW

### **1. Security Utilities Tests** (`tests/etl/test_security_utils.py`)

**Purpose**: Test error sanitization and log message sanitization

**Test Classes**:
- `TestErrorSanitization` - Error message sanitization
- `TestLogMessageSanitization` - Log message sanitization
- `TestSecurityIntegration` - Integration scenarios

**Key Tests**:
- ✅ Production mode returns generic errors
- ✅ Development mode returns detailed errors
- ✅ Auto-detection of production environment
- ✅ API key redaction in logs
- ✅ Password redaction in logs
- ✅ Token redaction in logs
- ✅ Multiple sensitive data redaction

**Run**: `pytest tests/etl/test_security_utils.py -v`

---

### **2. Secret Loader Tests** (`tests/etl/test_secret_loader.py`)

**Purpose**: Test secure secret loading from Docker secrets or environment variables

**Test Classes**:
- `TestSecretLoader` - Core secret loading functionality
- `TestConvenienceFunctions` - Convenience functions
- `TestSecretLoaderIntegration` - Integration scenarios

**Key Tests**:
- ✅ Load from environment variable
- ✅ Load from Docker secret file
- ✅ Fallback chain (Docker → env → None)
- ✅ Handle missing files
- ✅ Ignore comments in secret files
- ✅ Handle empty files
- ✅ Auto-detect _FILE suffix

**Run**: `pytest tests/etl/test_secret_loader.py -v`

---

### **3. Database Security Tests** (`tests/etl/test_database_security.py`)

**Purpose**: Test database file permissions and secure operations

**Test Classes**:
- `TestDatabaseFilePermissions` - File permission security
- `TestDatabaseSecurityIntegration` - Integration scenarios

**Key Tests**:
- ✅ New database has secure permissions (0o600)
- ✅ Existing database permissions updated
- ✅ Database operations work with secure permissions
- ✅ Cross-platform compatibility

**Run**: `pytest tests/etl/test_database_security.py -v`

---

### **4. Integration Tests** (`tests/integration/test_security_integration.py`)

**Purpose**: Test security features in actual ETL pipeline context

**Test Classes**:
- `TestSecurityIntegrationInPipeline` - Pipeline integration
- `TestDatabaseSecurityIntegration` - Database security
- `TestSecretLoadingIntegration` - Secret loading

**Key Tests**:
- ✅ DatabaseManager uses error sanitization
- ✅ DataSourceManager uses secret_loader
- ✅ Error handling in production vs development
- ✅ Log sanitization in pipeline
- ✅ Database security in real scenarios
- ✅ Secret loader fallback chain

**Run**: `pytest tests/integration/test_security_integration.py -v -m integration`

---

## 🔍 TEST MARKERS

All security tests are marked with `@pytest.mark.security`:

```python
@pytest.mark.security
def test_security_feature():
    # Test implementation
```

**Run security tests only**:
```bash
pytest -m security -v
```

**Run all tests except security**:
```bash
pytest -m "not security" -v
```

---

## 📊 EXPECTED TEST RESULTS

### **Successful Run**

```
============================= test session starts ==============================
tests/etl/test_security_utils.py::TestErrorSanitization::test_sanitize_error_production_mode_generic PASSED
tests/etl/test_security_utils.py::TestErrorSanitization::test_sanitize_error_development_mode_detailed PASSED
...
tests/etl/test_secret_loader.py::TestSecretLoader::test_load_secret_from_environment_variable PASSED
...
tests/etl/test_database_security.py::TestDatabaseFilePermissions::test_new_database_has_secure_permissions PASSED
...
tests/integration/test_security_integration.py::TestSecurityIntegrationInPipeline::test_database_manager_uses_error_sanitization PASSED
...
======================== 42 passed in 2.34s =========================
```

---

## 🔄 INTEGRATION STATUS

### **Components Integrated with Security Features**

| Component | Security Feature | Status |
|-----------|-----------------|--------|
| `DatabaseManager` | Error sanitization | ✅ Integrated |
| `DataSourceManager` | Secret loading | ✅ Integrated |
| `AlphaVantageExtractor` | Secret loading | ✅ Integrated |
| `FinnhubExtractor` | Secret loading | ✅ Integrated |
| `run_etl_pipeline.py` | Error sanitization | ✅ Integrated |

---

## 📝 TESTING CHECKLIST

Before deploying:

- [ ] Run all security tests: `pytest -m security -v`
- [ ] Verify all tests pass
- [ ] Check error sanitization works in production mode
- [ ] Verify secret loading works with Docker secrets
- [ ] Verify secret loading falls back to env vars
- [ ] Check database permissions are set correctly
- [ ] Verify integration tests pass
- [ ] Run full test suite: `pytest -v`
- [ ] Run OpenClaw autonomy guard tests: `pytest tests/utils/test_openclaw_cli.py -k "autonomy_guard or AutonomyGuard" -v`
- [ ] Verify OpenClaw hardening flags/config: `python scripts/verify_openclaw_config.py`

---

## 🚀 CI/CD INTEGRATION

### **Add to GitHub Actions**

```yaml
name: Security Tests

on: [push, pull_request]

jobs:
  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest -m security -v
```

---

## 🔧 TROUBLESHOOTING

### **Test Failures**

1. **Import Errors**: Ensure pytest is installed
   ```bash
   pip install pytest
   ```

2. **Permission Errors**: On Windows, file permissions are handled by OS
   - Tests may skip permission checks on Windows
   - This is expected behavior

3. **Secret File Errors**: Ensure test secret files are created
   - Tests use temporary files
   - No manual setup required

---

**Status**: ✅ **All Security Tests Created and Integrated**  
**Next**: Run tests and add to CI/CD pipeline

