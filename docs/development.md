# Development, Testing & Architecture

This guide covers running the test suite, evaluating classification latency with the benchmark suite, and navigating the internal codebase architecture.

[← Back to README](../README.md) • [CLI Reference & Workflows](cli_and_workflows.md)

---

## 1. Running the Test Suite

Ordinale uses `pytest` for unit and integration testing across extraction, classification heuristics, and organizer engine operations.

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Run Tests
```bash
pytest -v
```

### Running Specific Test Modules
```bash
# Test in-memory file extractors (.pdf, .docx, .html, .md, .txt)
pytest tests/test_extractor.py -v

# Test classification logic, heuristics, and CUDA error handling
pytest tests/test_doc_classifier.py -v

# Test safe moves, collision resolution, and undo manifest rollback
pytest tests/test_organizer_engine.py -v

# Test CLI arguments and flags
pytest tests/test_doc_cli.py -v
```

---

## 2. Running the Benchmark Suite

Ordinale includes synthetic sample documents representing diverse file types (tax notices, bank statements, homework assignments, arXiv papers, receipts, resumes, and medical records).

To measure classification accuracy and latency on your current hardware:

```bash
ordinale --samples
```

This processes the fixtures in `tests/fixtures/sample_documents.json` and renders a benchmark summary table:

```text
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ ID / Filename       ┃ Predicted Category  ┃ Expected ┃ Subcategory / Path          ┃ Retention        ┃ Sensitive┃ Action      ┃ Latency  ┃
┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━┛
...
Benchmark Performance:
Total Documents Tested: 12 | Category Accuracy: 100.0% | Avg Latency: ~18.5 ms/doc
```

You can also pass a custom benchmark dataset:
```bash
ordinale --samples-file "/path/to/my_samples.json"
```

---

## 3. Project Architecture

```
Ordinale/
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Production dependencies (laya, rich, pypdf, python-docx, etc.)
├── requirements-dev.txt        # Development dependencies (pytest)
├── README.md                   # Project landing page & quickstart
├── docs/                       # Modular documentation guides
│   ├── installation.md         # In-depth setup, venvs, and OS troubleshooting
│   ├── cli_and_workflows.md    # CLI options and triage workflows
│   ├── configuration.md        # Settings file specification (ordinale.toml / .json)
│   ├── cuda.md                 # CUDA setup, wheel matrix, and GPU troubleshooting
│   ├── classification.md       # Category taxonomy, heuristics, and metadata scoring
│   └── development.md          # Testing, benchmarks, and architecture (this file)
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

### Module Responsibilities

- **`extractor.py`**:
  Handles zero-disk extraction. Reads byte streams of `.pdf`, `.docx`, `.html`, `.rtf`, and text files directly in RAM and produces normalized text snippets for the classifier.
- **`doc_classifier.py`**:
  Interfaces with the Laya decision model. Implements hardware device negotiation (strict CUDA validation), zero-generation scoring, sub-routing heuristics, retention scoring, and privacy detection.
- **`organizer_engine.py`**:
  Computes planned moves, handles collision avoidance (SHA-256 deduplication and numeric suffixes), executes atomic moves, and manages `.organizer_manifest.json` for rollback.
- **`doc_organizer.py`**:
  Argparse CLI parsing, Rich terminal styling, dry-run display tables, and interactive prompts.
