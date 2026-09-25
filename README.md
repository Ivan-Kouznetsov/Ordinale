# Ordinale 📁⚡

> **Local, privacy-preserving document categorization and filing engine powered by Laya.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)
[![Powered By: Laya](https://img.shields.io/badge/Powered%20By-Laya-teal.svg)](https://huggingface.co/convaiinnovations/laya)
[![Accelerated: CUDA](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-green.svg)](#running-on-cuda-gpu-acceleration)

**Ordinale** is an intelligent, high-performance desktop assistant designed to clean up cluttered directories (such as `Downloads`, `Desktop`, or chaotic document archives). Using the zero-generation [Laya decision model](https://huggingface.co/convaiinnovations/laya), Ordinale inspects document snippets in-memory, categorizes them across multiple dimensions (category, financial sub-type, school vs. academic research, retention priority, and privacy sensitivity), and routes them into structured folders—all with safe dry-run visual previews, deduplication, and 1-command rollback.

---

## Table of Contents

- [What Ordinale Does](#what-ordinale-does)
- [Why Use Ordinale?](#why-use-ordinale)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [How to Use Ordinale](#how-to-use-ordinale)
  - [Command-Line Options](#command-line-options)
  - [Common Workflows](#common-workflows)
- [Running on CUDA (GPU Acceleration)](#running-on-cuda-gpu-acceleration)
- [Troubleshooting CUDA & GPU Issues](#troubleshooting-cuda--gpu-issues)
  - [Newer GPU Architectures & Precompiled Wheel Incompatibilities](#1-newer-gpu-architectures--precompiled-binary-incompatibilities)
  - [CUDA is Available is False](#2-torchcudais_available-returns-false)
  - [Strict CUDA Enforcement and CudaDeviceError](#3-strict-cuda-enforcement--cudadeviceerror)
  - [CUDA Out-of-Memory (OOM)](#4-cuda-out-of-memory-oom)
  - [Windows Symlinks & Long Path Caveats](#5-windows-symlinks--long-path-caveats)
- [Testing & Benchmarking](#testing--benchmarking)
- [Project Architecture](#project-architecture)
- [License](#license)

---

## What Ordinale Does

1. **Automated Multi-Format In-Memory Extraction**:
   - Scans directories recursively (or top-level) for `.docx`, `.pdf`, `.html`, `.htm`, `.txt`, `.md`, and `.rtf`.
   - Reads only the first 1–2 pages or representative text snippets (up to 1,200 characters) directly in memory.
   - **Never writes unencrypted intermediate text dumps or plain text snippets to disk.**

2. **Multi-Dimensional AI Classification**:
   - **Primary Categories**:
     - `Financial` (tax assessments, banking, payroll, investments, general loans)
     - `Receipts & Invoices` (purchase receipts, vendor invoices, utility bills)
     - `Contracts & Legal` (NDAs, residential leases, employment agreements, terms of service)
     - `Education & Academic` (coursework, homework, syllabi, arXiv preprints, journal articles)
     - `Web Articles & Clippings` (HTML snapshots, saved blogs, online articles)
     - `Career & Resumes` (curriculum vitae, resumes, cover letters)
     - `Personal & Identity` (medical records, insurance policies, vehicle registrations)
     - `Manuals & Guides` (technical documentation, user manuals, appliance setup)
     - `Notes & Drafts` (scratchpad notes, meeting minutes, rough drafts)
   - **Specialized Financial Sub-Routing**:
     - Automatically routes documents into subfolders such as `Financial/Taxes & Government` (e.g. CRA Notices of Assessment, T4/T5 slips), `Financial/Banking`, `Financial/Investments`, `Financial/Payroll`, or `Financial/General`.
   - **Intelligent Academic vs. School Disambiguation**:
     - Distinguishes student coursework from peer-reviewed research papers using regex heuristics for course codes (e.g., `CS 240`, `BIO:101`), preprint indicators (`arXiv`, `bioRxiv`, `SSRN`), publisher markers (`IEEE`, `ACM`, `Springer`, `Nature`), and file format priors.
   - **Retention Scoring**:
     - Scores whether a document is ephemeral/prunable (`0`), active reference (`1`), or a permanent archive (`2`).
   - **Privacy & Sensitivity Detection**:
     - Flags sensitive personal identity, tax identifiers (SSN / SIN), financial numbers, or confidential clauses.

3. **Safe File Operations & Rollback Engine**:
   - **Safe Dry-Run by Default**: Renders an interactive, color-coded visual triage table with confidence percentages and planned file movements before touching anything on disk.
   - **SHA-256 Collision Protection**: Identifies identical files and skips duplicate copies. If different documents share the same name, Ordinale appends safe incremental suffixes (e.g., `invoice (1).pdf`) instead of overwriting.
   - **Folder Hierarchy Preservation**: Maintains relative subfolder structures within destination category folders.
   - **1-Command Undo / Rollback**: Records every move action with SHA-256 hashes and original paths into `.organizer_manifest.json`, allowing complete restoration at any time.

---

## Why Use Ordinale?

| Feature | Ordinale | Traditional Cloud LLMs / Scripts |
| :--- | :--- | :--- |
| **Data Privacy** | **100% Local**. Zero cloud API calls. Confidential taxes, IDs, and financial records never leave your machine. | Files or snippets sent over the internet to third-party APIs. |
| **Inference Model** | **Zero-Generation Classification**. Uses Laya's direct scoring / choice engine in a single forward pass (<10–50 ms). | Generative LLMs generate long tokens, suffer latency, and can hallucinate filenames. |
| **Execution Safety** | **Dry-Run by Default** with Rich tables, low-confidence review prompts, and SHA-256 deduplication. | Might move/rename files or overwrite by default. |
| **Instant Rollback** | **Manifest Ledger (`--undo`)** reverts entire batches back to original locations instantaneously. | Manual recovery or lost original folder paths. |
| **Operating Cost** | **$0**. Unlimited local scans without API rate limits or subscription fees. | Per-token / per-document API charges. |

---

## How It Works

```
Messy Folder (PDF, DOCX, HTML, TXT, MD)
                 │
                 ▼
     DocumentTextExtractor
     (In-Memory Snippet Extraction: 400-1200 chars)
                 │
                 ▼
       DocumentClassifier
     (Laya Decision Engine: Choice + Scoring + Sensitivity)
                 │
                 ├── Category & Confidence
                 ├── Financial / Academic Subcategory
                 ├── Retention Score (0 - 2)
                 └── Privacy Sensitivity Flag
                 │
                 ▼
       OrganizerEngine Plan
     (Collision Resolution + SHA-256 Deduplication)
                 │
         ┌───────┴───────┐
         ▼               ▼
    [Dry-Run Preview]  [--execute]
   (Rich Terminal UI)    │
                         ▼
                   Safe Atomic Moves
                         │
                         ▼
             .organizer_manifest.json
              (Enables instant --undo)
```

---

## Installation & Setup

### Prerequisites
- **Python**: 3.9, 3.10, 3.11, or 3.12 (Python 3.10 or 3.11 recommended for maximum PyTorch and CUDA compatibility).
- **Operating System**: Windows 10/11, Linux (Ubuntu, Debian, Fedora, Arch), or macOS.
- **Hardware**: Any modern CPU; NVIDIA GPU optional but recommended for ultra-fast batch processing.

### 1. Clone the Repository
```bash
git clone https://github.com/Ivan-Kouznetsov/Ordinale.git
cd Ordinale
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# (Optional) Install Ordinale in editable mode for the CLI commands:
pip install -e .
```

> **Note on Initial Run**: On its first run, Ordinale automatically downloads the lightweight Laya model weights (`convaiinnovations/laya`) from Hugging Face Hub to your local cache. If running in an air-gapped or offline environment, pre-download the model and pass `--offline`.

---

## How to Use Ordinale

Once installed, you can invoke the CLI using `ordinale`, `doc-organizer`, or `python -m ordinale.doc_organizer`.

### Command-Line Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `--scan` | `<PATH>` | Source directory containing documents to organize. |
| `--target` | `<PATH>` | Destination root directory (defaults to `<source>/Organized_Documents`). |
| `--execute` | *None* | Perform actual file moves. *(Without this flag, Ordinale runs in safe dry-run preview mode.)* |
| `--interactive` | *None* | Prompts for confirmation when routing low-confidence documents. |
| `--undo` | *None* | Reverts the last executed move batch using the manifest ledger. |
| `--samples` | *None* | Runs categorization benchmark against test sample documents (default action if no flags provided). |
| `--samples-file` | `<FILE>` | Path to custom benchmark JSON dataset. |
| `--device` | `cpu` \| `cuda` \| `mps` | Hardware device to use. Defaults to automatic detection. |
| `--workers`, `-w` | `<INT>` | Number of parallel worker threads for analysis (defaults to `min(8, CPU cores)`). |
| `--sequential` | *None* | Force single-threaded processing. |
| `--no-recursive` | *None* | Only scan the top-level directory, ignoring subdirectories. |
| `--no-preserve-folders` | *None* | Do not preserve relative source subdirectories under destination categories. |
| `--offline` | *None* | Strictly use locally cached Hugging Face model weights without network checks. |

---

### Common Workflows

#### 1. Dry-Run Visual Preview (Default Safe Mode)
Inspect what Ordinale plans to do without altering or moving any files:
```bash
ordinale --scan "C:/Users/username/Downloads" --target "C:/Users/username/Documents/Organized"
```
Ordinale renders a Rich table displaying:
- Source document name & relative path
- Predicted destination subfolder
- Confidence score (color-coded green, yellow, red)
- Document retention lifespan (Permanent Archive, Active Reference, Prunable)
- Sensitive document indicator
- Planned triage action (`MOVE`, `MOVE (SECURE)`, `REVIEW NEEDED`, or `DUPLICATE (Skip)`)

#### 2. Execute File Movement
Once you are satisfied with the dry-run plan, apply the changes with `--execute`:
```bash
ordinale --scan "C:/Users/username/Downloads" --target "C:/Users/username/Documents/Organized" --execute
```

#### 3. Interactive Review for Ambiguous Documents
Prompt for manual confirmation before moving files that fall below the confidence threshold:
```bash
ordinale --scan "./my_archive" --target "./organized" --execute --interactive
```

#### 4. Undo / Rollback
If you ever want to revert an entire batch back to its original folder structure:
```bash
ordinale --target "C:/Users/username/Documents/Organized" --undo
```
Ordinale reads `.organizer_manifest.json`, verifies file hashes, moves every file back to its original location, and updates the manifest.

#### 5. Fast Parallel Processing on Large Collections
Tune the concurrency worker threads for multi-core CPUs:
```bash
ordinale --scan "/path/to/archive" --target "/path/to/target" --workers 8 --execute
```

---

## Running on CUDA (GPU Acceleration)

Ordinale supports GPU acceleration via NVIDIA CUDA. Running on CUDA accelerates document categorization down to **a few milliseconds per document**.

### 1. Enable CUDA
Pass `--device cuda` (or specify a specific GPU index like `--device cuda:0`):
```bash
ordinale --scan "./documents" --target "./organized" --device cuda
```

### 2. Install PyTorch with CUDA Support
Standard `pip install torch` from PyPI often defaults to CPU-only wheels on Windows and Linux. To run on CUDA, you must install PyTorch with the CUDA wheel matching your installed NVIDIA driver:

```bash
# Recommended for CUDA 12.4 (Modern GPUs / Drivers 550+):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (Legacy GPUs / Older Drivers):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify CUDA in Python
To verify that PyTorch detects your NVIDIA GPU correctly:
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
   For cutting-edge or unreleased hardware architectures, use the PyTorch Nightly build:
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
Ordinale enforces device honesty. In many other libraries, requesting `cuda` silently downgrades to `cpu` when an initialization error occurs, causing unexplained slowdowns. Ordinale prohibits this fallback behavior.

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
- Another GPU-intensive process (e.g. video generation, local 70B LLM server, game) is consuming all VRAM.
- Multiple worker threads allocate concurrent GPU contexts.

#### Solution:
1. **Reduce Worker Threads or Run Sequentially**:
   ```bash
   ordinale --scan "./my_docs" --target "./organized" --device cuda --sequential
   # or limit to 2 workers:
   ordinale --scan "./my_docs" --target "./organized" --device cuda --workers 2
   ```
2. **Clear GPU Memory**: Close background applications holding GPU allocations (check VRAM usage using `nvidia-smi`).

---

### 5. Windows Symlinks & Long Path Caveats

- **Symlink Privilege Error (`WinError 1314`)**:
  Windows requires Administrator privileges or Developer Mode to create filesystem symlinks. Ordinale automatically sets `HF_HUB_DISABLE_SYMLINKS=1` in Python before importing Hugging Face libraries, avoiding this error completely.
- **Path Length Limit (MAX_PATH = 260 characters)**:
  When scanning deep nested folders, Windows may throw path truncation errors. Enable Long Paths in Windows:
  1. Open PowerShell as Administrator.
  2. Run:
     ```powershell
     New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
       -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
     ```

---

## Testing & Benchmarking

Ordinale includes an extensive test suite and synthetic sample documents representing diverse file types (tax notices, bank statements, homework assignments, arXiv papers, receipts, resumes, and medical records).

### Run Test Suite
```bash
pytest -v
```

### Run Benchmark Suite
To test classification accuracy and latency against the bundled fixtures:
```bash
ordinale --samples
```
This generates a detailed benchmark table and summary statistics:
```text
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ ID / Filename       ┃ Predicted Category  ┃ Expected ┃ Subcategory / Path          ┃ Retention        ┃ Sensitive┃ Action      ┃ Latency  ┃
┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━┛
...
Benchmark Performance:
Total Documents Tested: 12 | Category Accuracy: 100.0% | Avg Latency: ~18.5 ms/doc | Cost: $0 (100% Local)
```

---

## Project Architecture

```
Ordinale/
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Production dependencies (laya, rich, pypdf, python-docx, etc.)
├── requirements-dev.txt        # Development dependencies (pytest)
├── README.md                   # Project documentation
├── src/
│   └── ordinale/
│       ├── __init__.py         # Package exports
│       ├── doc_organizer.py    # CLI entry point, visual Rich tables, and argument parsing
│       ├── doc_classifier.py   # Laya integration, multi-task scoring, & CUDA validation
│       ├── extractor.py        # In-memory document text extractor (.docx, .pdf, .html, .txt, .md)
│       └── organizer_engine.py # Planning, collision resolution, safe moves, & undo manifest
└── tests/
    ├── fixtures/
    │   └── sample_documents.json # Synthetic sample benchmark dataset
    ├── test_doc_classifier.py  # Classifier logic, heuristics, & CUDA error handling tests
    ├── test_doc_cli.py         # CLI parameter & flag tests
    ├── test_doc_dataset.py     # End-to-end dataset tests
    ├── test_extractor.py       # Format extractor tests
    └── test_organizer_engine.py# Safe movement, deduplication, & undo rollback tests
```

---

## License

This project is licensed under the [MIT License](LICENSE).