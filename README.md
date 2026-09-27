# Ordinale

> **Local, privacy-preserving document categorization and filing engine powered by Laya.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)
[![Powered By: Laya](https://img.shields.io/badge/Powered%20By-Laya-teal.svg)](https://huggingface.co/convaiinnovations/laya)
[![Accelerated: CUDA](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-green.svg)](docs/cuda.md)

---

**Ordinale** is a desktop document organizer for cluttered directories (such as `Downloads`, `Desktop`, or document archives). Using the zero-generation [Laya decision model](https://huggingface.co/convaiinnovations/laya) and NLP-based heuristics, Ordinale inspects document snippets in memory, classifies them across multiple dimensions (financial sub-types, research vs. coursework, retention priority, and privacy sensitivity), and routes them into organized folders with dry-run visual previews and rollback support.

```
Messy Folder ──▶ In-Memory Extractor ──▶ Laya Decision Engine ──▶ Safe Triage Plan
(PDF/DOCX/HTML)  (400-1200 chars in RAM) (Zero-Gen Classification)   (Dry-Run Table & Rollback)
```

---

## Why Ordinale?

- 🔒 **100% Local & Private**: No cloud APIs or telemetry. Sensitive taxes, IDs, and financial records never leave your machine.
- ⚡ **Zero-Generation Speed**: Classifies documents in a single forward pass (<10–50 ms) without LLM hallucinations.
- 🛡️ **Safe Dry-Run by Default**: Renders an interactive, color-coded terminal table before touching any files on disk.
- ↩️ **1-Command Undo**: Reverts entire batches back to original locations instantaneously using `.organizer_manifest.json`.
- 🧩 **Multi-Dimensional Routing**: Disambiguates tax slips, banking records, student homework, arXiv preprints, receipts, and notes.

---

## Quickstart (30 Seconds)

### On Windows (Command Prompt `cmd.exe`)

```cmd
git clone https://github.com/Ivan-Kouznetsov/Ordinale.git
cd Ordinale

python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt
python -m spacy download en_core_web_sm

:: Safe dry-run preview (does not move files):
python -m ordinale.doc_organizer --scan "%USERPROFILE%\Downloads"
```

### On Windows (PowerShell)

```powershell
git clone https://github.com/Ivan-Kouznetsov/Ordinale.git
cd Ordinale

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Safe dry-run preview (does not move files):
python -m ordinale.doc_organizer --scan "$HOME\Downloads"
```

### On Linux / macOS (Bash / Zsh)

```bash
git clone https://github.com/Ivan-Kouznetsov/Ordinale.git
cd Ordinale

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Safe dry-run preview (does not move files):
python -m ordinale.doc_organizer --scan ~/Downloads
```

---

## Essential Commands

| Action | Command |
| :--- | :--- |
| **Dry-Run Preview** | `ordinale --scan "C:/path/to/docs" --target "C:/path/to/organized"` |
| **Execute Move** | `ordinale --scan "C:/path/to/docs" --target "C:/path/to/organized" --execute` |
| **Interactive Triage** | `ordinale --scan "C:/path/to/docs" --execute --interactive` |
| **Instant Rollback** | `ordinale --target "C:/path/to/organized" --undo` |
| **Benchmark Suite** | `ordinale --samples` |

---

## Documentation Hub

Explore the guides in the `docs/` folder for complete details:

| Guide | Description |
| :--- | :--- |
| 🚀 **[Installation & Setup](docs/installation.md)** | Step-by-step virtual environment setup, offline model caching, and Windows path caveats. |
| 📖 **[CLI Reference & Workflows](docs/cli_and_workflows.md)** | Full command-line options reference, parallel processing, and interactive review. |
| ⚙️ **[Configuration Reference](docs/configuration.md)** | Customizing `ordinale.toml` / `ordinale.json`, custom file extensions, and NLP settings. |
| ⚡ **[CUDA & GPU Acceleration](docs/cuda.md)** | PyTorch CUDA wheel matrix, GPU detection, and troubleshooting GPU architectures / OOM. |
| 🧠 **[Classification & Heuristics](docs/classification.md)** | Primary categories, financial sub-routing, coursework vs. research detection, and retention scoring. |
| 🧪 **[Development & Architecture](docs/development.md)** | Running unit tests (`pytest`), synthetic benchmark fixtures, and repository codebase architecture. |

---

## License

This project is licensed under the [MIT License](LICENSE).
