# CUDA & GPU Acceleration Guide

Ordinale supports GPU acceleration via NVIDIA CUDA. Running on CUDA accelerates document categorization down to **a few milliseconds per document**.

[← Back to README](../README.md) • [CLI Reference & Workflows](cli_and_workflows.md)

---

## 1. Enable CUDA

Pass `--device cuda` (or specify an explicit GPU index like `--device cuda:0`):

```bash
ordinale --scan "./documents" --target "./organized" --device cuda
```

---

## 2. Install PyTorch with CUDA Support

Standard `pip install torch` from PyPI often defaults to CPU-only wheels on Windows and Linux. To run on CUDA, install PyTorch with the CUDA wheel matching your installed NVIDIA driver and GPU architecture:

```bash
# Recommended for CUDA 13.2 (RTX 50-series Blackwell / Compute Capability 12.0+ / Drivers 590+):
pip install torch --index-url https://download.pytorch.org/whl/cu132

# For CUDA 12.4 / 12.6 (RTX 40-series / 30-series / Drivers 550+):
pip install torch --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (Legacy GPUs / Older Drivers):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. Verify CUDA in Python

Verify that PyTorch detects your NVIDIA GPU correctly:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```text
CUDA Available: True
Device Name: NVIDIA GeForce RTX ...
```

---

## Troubleshooting CUDA & GPU Issues

Ordinale incorporates strict CUDA verification in `doc_classifier.py`: **if you specify `--device cuda`, Ordinale will never silently downgrade to CPU.** Instead, it will fail fast with a descriptive `CudaDeviceError` so you immediately know why your GPU is not running.

Below are common CUDA issues and how to resolve them:

### 1. Newer GPU Architectures & Precompiled Binary Incompatibilities

#### Symptom:
When running on newer NVIDIA GPU architectures (such as **Ada Lovelace / RTX 40-series** with compute capability `sm_89`, or **Blackwell / RTX 50-series** with `sm_100`/`sm_120`), you may encounter:
```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors / device-side assert triggered
Process finishes with CudaDeviceError: CUDA execution failed during model warmup
```

#### Cause:
Precompiled Python wheels (such as older PyTorch binaries or third-party compiled C++/CUDA extensions) are compiled against a fixed set of target CUDA architectures (e.g., `sm_75` for Turing, `sm_80`/`sm_86` for Ampere). If your GPU's compute capability is newer than what was baked into the installed PyTorch wheel, CUDA cannot find compatible machine code or PTX JIT kernels for your chip.

#### Solution:
1. **Upgrade to PyTorch with CUDA 12.4+ or Nightly**:
   Newer GPU architectures require PyTorch wheels compiled with modern CUDA toolkits:
   ```bash
   pip uninstall torch -y
   pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
   ```
   For newer or unreleased hardware architectures, use the PyTorch Nightly build:
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
   ```
2. **Set `TORCH_CUDA_ARCH_LIST` (If Building from Source)**:
   If building custom CUDA wheels or PyTorch extensions, explicitly specify your GPU's architecture:
   - **PowerShell (Windows)**:
     ```powershell
     $env:TORCH_CUDA_ARCH_LIST="8.9"   # For RTX 4080 / 4090 (Ada Lovelace)
     ```
   - **Command Prompt (Windows)**:
     ```cmd
     set TORCH_CUDA_ARCH_LIST=8.9
     ```
   - **Linux (Bash)**:
     ```bash
     export TORCH_CUDA_ARCH_LIST="8.9"
     ```
3. **Verify Driver Version**:
   Run `nvidia-smi` in your terminal. Ensure the top-right `CUDA Version: XX.X` reported by your driver is **greater than or equal** to the CUDA toolkit version of your PyTorch build (e.g., Driver CUDA version $\ge$ 12.4). Update your driver from [NVIDIA's website](https://www.nvidia.com/download/index.aspx) if necessary.

---

### 2. `torch.cuda.is_available()` returns `False`

#### Cause:
You have a CPU-only build of PyTorch installed in your virtual environment.

#### Solution:
Check your current installation:
```bash
python -c "import torch; print(torch.__version__)"
```
If the version ends in `+cpu` (or does not contain `+cu12X`), reinstall the CUDA version:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

### 3. Strict CUDA Enforcement & `CudaDeviceError`

#### Symptom:
```text
Device Error: CUDA device 'cuda' was explicitly specified, but torch.cuda.is_available() is False.
Exiting immediately without falling back to CPU because CUDA was specified.
```

#### Cause:
Ordinale strictly enforces requested hardware devices without falling back to CPU. In many other libraries, requesting `cuda` silently downgrades to `cpu` when an initialization error occurs, causing unexplained slowdowns. Ordinale avoids silent fallback.

#### Solution:
- If you intended to run on CPU, omit the `--device` flag or pass `--device cpu`.
- If you intended to run on CUDA, resolve the PyTorch CUDA installation using the steps in [Section 2](#2-torchcudais_available-returns-false).

---

### 4. CUDA Out-of-Memory (OOM)

#### Symptom:
```text
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

#### Cause:
Laya is a lightweight decision model (< 1 GB VRAM footprint). However, OOM can happen if:
- Another GPU-intensive process (e.g., video generation, local 70B LLM server, game) is consuming all VRAM.
- Multiple worker threads allocate concurrent GPU contexts.

#### Solution:
1. **Reduce Worker Threads or Run Sequentially**:
   ```bash
   ordinale --scan "./my_docs" --target "./organized" --device cuda --sequential
   # or limit to 2 workers:
   ordinale --scan "./my_docs" --target "./organized" --device cuda --workers 2
   ```
2. **Clear GPU Memory**: Close background applications holding GPU allocations (check VRAM usage using `nvidia-smi`).
