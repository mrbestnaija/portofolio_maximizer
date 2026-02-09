# Infrastructure Recovery Plan
**Generated**: 2026-02-07
**Status**: Partial recovery completed, manual steps required

## Current State

### Recovered
- ✅ Source code cloned from GitHub (origin/master)
- ✅ .env file with API credentials preserved
- ✅ Git repository functional
- ✅ Dependencies documented (requirements.txt, requirements-ml.txt)

### Requires Manual Installation
- ❌ Python 3.10-3.12 not installed
- ❌ WSL Ubuntu not installed (network timeout)
- ❌ Model artifacts (.pt files) not located

## Manual Recovery Steps

### 1. Install Python (OFFLINE METHOD - Bandwidth Resilient)

**Option A: Official Python Installer (Recommended)**
1. On a machine with stable internet, download Python 3.12:
   - URL: `https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe`
   - Size: ~25 MB
2. Transfer installer to this machine via USB/local network
3. Run installer with these options:
   - ✅ Add Python to PATH
   - ✅ Install for all users
   - Install location: `C:\Python312`

**Option B: Portable Python**
1. Download embeddable Python from python.org
2. Extract to `C:\Python312Portable`
3. Manually add to system PATH

**Verify Installation:**
```bash
python --version  # Should show Python 3.12.x
```

### 2. Install WSL Ubuntu (OFFLINE/RESUMABLE METHOD)

**Option A: Manual Appx Download (Resumable with Download Manager)**
1. On stable internet or using download manager with resume capability:
   - Download: `https://aka.ms/wslubuntu2204`
   - File: Ubuntu 22.04 LTS (~500 MB)
   - Use wget/curl/browser download manager that supports resume

2. Transfer .appx file to this machine

3. Install manually:
   ```powershell
   # In PowerShell as Administrator
   Add-AppxPackage .\Ubuntu-22.04.appx
   ```

4. Initialize Ubuntu:
   ```bash
   wsl --set-default Ubuntu-22.04
   ubuntu2204.exe
   # Follow setup prompts - create user 'mrbestnaija'
   ```

**Option B: Import Existing Ubuntu Tar (If Available)**
If you have a backup Ubuntu rootfs:
```bash
wsl --import Ubuntu-22.04 C:\WSL\Ubuntu ubuntu-rootfs.tar.gz
```

### 3. Set Up Python Virtual Environment

Once Python is installed:

```bash
cd c:/Users/Bestman/personal_projects/portfolio_maximizer_v45/portfolio_maximizer_v45

# Create fresh virtual environment
python -m venv venv

# Activate
source venv/Scripts/activate  # Git Bash
# or
venv\Scripts\activate.bat     # CMD
# or
venv\Scripts\Activate.ps1     # PowerShell

# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies (NO GPU - for initial setup)
pip install -r requirements.txt --no-cache-dir

# Optional: GPU dependencies (only if CUDA available)
# pip install -r requirements-ml.txt
```

### 4. Verify Project Structure

```bash
# Test basic imports
python -c "import pandas; import numpy; import torch; print('Core imports successful')"

# Test project modules
python -c "from etl import *; from ai_llm import *; print('Project imports successful')"
```

### 5. Model Artifacts Recovery

**Search Locations:**
- Check WSL filesystem once accessible: `/home/mrbestnaija/`
- Check Windows: `C:\Users\Bestman\`
- Check external drives/backups

**Expected Files:**
- Trained model weights: `*.pt`, `*.pth`
- Model configs: `*.json`, `*.yaml`
- Checkpoints: `data/checkpoints/*`

**If models are lost:**
- Retrain from scratch using recovered pipeline
- Check if models were backed up to cloud storage (DVC, S3, etc.)

### 6. Database Recovery

The project uses a SQLite database with WSL symlink:
```
data/portfolio_maximizer.db.wsl -> /home/mrbestnaija/portfolio_data/data/portfolio_maximizer.db
```

**Once WSL is restored:**
```bash
# In WSL Ubuntu
mkdir -p /home/mrbestnaija/portfolio_data/data

# Option A: Recreate empty database (project will rebuild)
# Database will auto-initialize on first run

# Option B: Restore from backup if available
# cp /path/to/backup/portfolio_maximizer.db /home/mrbestnaija/portfolio_data/data/
```

### 7. Test End-to-End

```bash
# Activate venv
source venv/Scripts/activate

# Run basic tests
pytest tests/ -v

# Test data pipeline
python -m etl.pipeline --test

# Test ML pipeline (if models available)
python -m ai_llm.market_analyzer --test
```

## Deterministic Rebuild Checklist

- [ ] Python 3.10-3.12 installed
- [ ] WSL Ubuntu 22.04 installed and initialized
- [ ] Virtual environment created: `venv/`
- [ ] Core dependencies installed from requirements.txt
- [ ] GPU dependencies installed (if CUDA GPU available)
- [ ] .env file in place with valid API keys
- [ ] WSL user 'mrbestnaija' created
- [ ] Database directory structure recreated
- [ ] Model artifacts located/retrained
- [ ] Basic imports working
- [ ] Tests passing
- [ ] Entry points functional

## Hardening Against Future Loss

### Critical Changes:
1. **Never rely on temp directories**: All state in version-controlled or permanent locations
2. **Virtual environments are disposable**: Document with requirements.txt, recreate as needed
3. **Model artifacts need backup strategy**:
   - Use DVC (Data Version Control) for large files
   - Separate storage from code repository
   - Regular backups to external/cloud storage
4. **WSL data**: Keep critical data on Windows filesystem, not WSL-only paths
5. **Database**: Consider cloud-hosted DB or regular backups to persistent Windows storage

### Best Practices:
```bash
# Always freeze dependencies after changes
pip freeze > requirements.txt

# Track model artifacts with DVC
dvc add data/checkpoints/model_v1.pt
dvc push

# Use .env.template for structure, never commit actual .env
cp .env .env.template
# Edit .env.template to remove sensitive values
git add .env.template
```

## Next Steps After Manual Installation

Once Python and WSL are installed, run:

```bash
cd c:/Users/Bestman/personal_projects/portfolio_maximizer_v45/portfolio_maximizer_v45
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest tests/ --collect-only  # Verify test discovery
```

## Support Resources

- Project README: [README.md](./README.md)
- Quick Reference: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- Claude Context: [CLAUDE.md](./CLAUDE.md)
- Documentation: [Documentation/](./Documentation/)

## Rollback Plan

If issues arise, the old directory is still available:
```
c:\Users\Bestman\personal_projects\portfolio_maximizer_v45\  (OLD - corrupted git)
c:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\  (NEW - fresh clone)
```

Old data directories (if needed):
- `.env.backup` - Original environment file
- `data/` - May contain remnants of old data

---

**Recovery Principle**: Rebuild deterministically from version control and dependency manifests. Avoid speculative recovery of corrupted environments.
