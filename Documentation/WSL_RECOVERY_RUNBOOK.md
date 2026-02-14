# WSL Runtime Recovery Runbook (Checklist-Valid Runs)

## Current Symptom (Observed 2026-02-13)

WSL2 cannot create its utility VM. Any attempt to register/install a distro fails with:

- `Wsl/Service/RegisterDistro/CreateVm/HCS/ERROR_FILE_NOT_FOUND`

Examples from this host:

- `wsl.exe --install --from-file "%TEMP%\\Ubuntu-24.04 (5).wsl" --name Ubuntu-24.04 --no-launch --version 2`
- `ubuntu2204.exe install --root`

This blocks **checklist-valid** runs under WSL `simpleTrader_env/bin/python`.

## Required Admin Repair (Run In Elevated PowerShell)

1) Ensure WSL + virtualization features are enabled:

```powershell
dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism /online /enable-feature /featurename:HypervisorPlatform /all /norestart
```

2) Reboot Windows.

3) After reboot, update WSL and confirm it can start the utility VM:

```powershell
wsl --update
wsl --shutdown
wsl --status
```

4) Install a distro (no interactive launch during install):

Option A: From the cached `.wsl` package downloaded by prior attempts:

```powershell
wsl --install --from-file "$env:TEMP\\Ubuntu-24.04 (5).wsl" --name Ubuntu-24.04 --no-launch --version 2
```

Option B: Web download:

```powershell
wsl --install -d Ubuntu-24.04 --no-launch --web-download
```

5) First launch (will prompt for user creation):

```powershell
wsl -d Ubuntu-24.04
```

## Repo Location Note (Linux venv vs Windows venv name collision)

This repo currently has a Windows virtualenv folder named `simpleTrader_env\\` in the Windows checkout.

Checklist-valid WSL runs expect a **Linux venv** at `simpleTrader_env/bin/python`, which conflicts if you run from `/mnt/c/...` pointing at the same checkout.

Recommended options:

1) Clone the repo inside the WSL filesystem (ext4) and create the Linux venv there.
2) Or rename the Windows venv folder (e.g., `simpleTrader_env_win`) and create the Linux venv at `simpleTrader_env/` under WSL.

## Checklist Fingerprint (After WSL Is Working)

From WSL:

```bash
cd <repo>
source simpleTrader_env/bin/activate

which python
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Acceptance:

- `which python` points to `.../simpleTrader_env/bin/python`
- `torch` import succeeds (CUDA availability depends on host setup)

