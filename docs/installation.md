# Installation & Setup Guide

This guide covers setting up Ordinale across Windows, macOS, and Linux, configuring offline model caching, and resolving operating-system specific caveats.

[← Back to README](../README.md) • [Next: CLI Reference & Workflows →](cli_and_workflows.md)

---

## Prerequisites

- **Python**: 3.9, 3.10, 3.11, or 3.12 (Python 3.10 or 3.11 recommended for maximum PyTorch and CUDA compatibility).
- **Operating System**: Windows 10/11, Linux (Ubuntu, Debian, Fedora, Arch), or macOS.
- **Hardware**: Any modern CPU; NVIDIA GPU optional but recommended for high-volume batch processing (see the [CUDA & GPU Acceleration Guide](cuda.md)).

---

## 1. Clone the Repository

```bash
git clone https://github.com/Ivan-Kouznetsov/Ordinale.git
cd Ordinale
```

---

## 2. Create and Activate a Virtual Environment

Always run Ordinale within an isolated Python virtual environment.

### On Windows (Command Prompt `cmd.exe`):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.)*

### On Linux / macOS (Bash / Zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install runtime dependencies
pip install -r requirements.txt

# Download the lightweight spaCy English NLP model for entity and header heuristics:
python -m spacy download en_core_web_sm

# (Optional) Install Ordinale in editable mode for global CLI commands:
pip install -e .
```

---

## Model Caching & Offline Detection

On its initial run, Ordinale fetches the lightweight Laya model weights (`convaiinnovations/laya`) from Hugging Face Hub to your local cache (`~/.cache/huggingface/hub` or `%USERPROFILE%\.cache\huggingface\hub`).

- **Automatic Offline Mode**: Once cached locally, Ordinale automatically operates in offline mode on subsequent runs. This eliminates network latency, bypasses Hugging Face API checks, and works in air-gapped environments.
- **Force Online Mode**: To check Hugging Face Hub for updated model revisions, pass the `--online` flag:
  ```bash
  ordinale --scan "C:/path/to/docs" --online
  ```
- **Force Offline Mode**: To strictly forbid any network checks even if a cache check might be triggered, pass `--offline`.

---

## Windows System Caveats & Tips

### 1. Symlink Privilege Error (`WinError 1314`)
Windows standard user accounts require Administrator privileges or Developer Mode to create filesystem symlinks. Ordinale automatically sets `HF_HUB_DISABLE_SYMLINKS=1` in Python before importing Hugging Face libraries, avoiding this error completely.

### 2. Path Length Limit (`MAX_PATH = 260` characters)
When scanning deep nested directory structures, Windows may throw file path truncation or `FileNotFoundError` exceptions. To remove the 260-character limit:
1. Open PowerShell as Administrator.
2. Run:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
     -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
3. Restart your terminal session.

---

## Next Steps

- Explore available flags and workflows in [CLI Reference & Workflows](cli_and_workflows.md).
- Accelerate inference with NVIDIA GPUs in the [CUDA Guide](cuda.md).
- Customize scan rules and extensions in [Configuration](configuration.md).
