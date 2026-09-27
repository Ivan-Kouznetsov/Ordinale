# CLI Reference & Workflows

Ordinale provides a command-line interface with dry-run previews, interactive review, and rollback support.

[← Back to README](../README.md) • [Configuration Guide →](configuration.md)

---

## Invoking the CLI

Depending on your installation method, you can invoke the CLI using any of the following:

```bash
# If installed with pip install -e .
ordinale [OPTIONS]
doc-organizer [OPTIONS]

# Or via Python module execution:
python -m ordinale.doc_organizer [OPTIONS]
```

---

## Command-Line Options Reference

| Flag | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `--scan` | `<PATH>` | *None* | Source directory containing documents to organize. |
| `--target` | `<PATH>` | `<source>/Organized_Documents` | Destination root directory for sorted folders. |
| `--execute` | *None* | *Disabled (Dry-run)* | Perform actual file moves. *(By default, runs in safe preview mode without moving files.)* |
| `--interactive` | *None* | *Disabled* | Prompts for manual confirmation when routing low-confidence documents. |
| `--undo` | *None* | *Disabled* | Reverts the last executed move batch using the manifest ledger. |
| `--samples` | *None* | *Active (Default action)* | Runs categorization benchmark against test sample documents (runs when neither `--scan` nor `--undo` is passed). |
| `--samples-file` | `<FILE>` | `sample_documents.json` | Path to benchmark JSON dataset. |
| `--device` | `cpu` \| `cuda` \| `mps` | `auto` | Hardware compute device (auto-detects CUDA $\rightarrow$ MPS $\rightarrow$ CPU). |
| `--workers`, `-w` | `<INT>` | `min(8, CPU cores)` | Number of parallel worker threads for file analysis. |
| `--sequential` | *None* | *Disabled (Parallel)* | Force single-threaded processing. |
| `--no-recursive` | *None* | *Disabled (Recursive)* | Only scan the top-level directory, ignoring subdirectories. |
| `--no-preserve-folders` | *None* | *Disabled (Preserve)* | Do not preserve relative source subdirectories under destination categories. |
| `--offline` | *None* | *Auto (On if cached)* | Force offline mode (reads directly from local cache without checking Hugging Face Hub). Enabled automatically if model is already downloaded. |
| `--online` | *None* | *Disabled* | Force online mode to check Hugging Face Hub for model updates even if cached locally. |
| `--config` | `<PATH>` | Auto-discovered / bundled | Path to a custom settings file (`.toml` or `.json`). |

---

## Common Workflows

### 1. Dry-Run Visual Preview (Default Safe Mode)

Always start with a dry run to inspect the planned actions before moving any files:

```bash
ordinale --scan "C:/Users/username/Downloads" --target "C:/Users/username/Documents/Organized"
```

Ordinale scans files, extracts snippets in memory, runs classification, and renders a Rich terminal table showing:
- **Source File**: Original relative filename and directory.
- **Predicted Destination**: Target subfolder path (including specialized sub-categories like `Financial/Taxes & Government`).
- **Confidence**: Prediction score with color coding:
  - 🟢 **Green ($\ge 75\%$)**: High confidence.
  - 🟡 **Yellow ($50\% - 74\%$)**: Moderate confidence.
  - 🔴 **Red ($< 50\%$)**: Low confidence (flagged for review).
- **Retention**: Document lifecycle priority (`Permanent Archive`, `Active Reference`, `Prunable`).
- **Sensitive**: Security indicator flagging documents with confidential markers or identifiers.
- **Action**: Planned operation (`MOVE`, `MOVE (SECURE)`, `REVIEW NEEDED`, or `DUPLICATE (Skip)`).

---

### 2. Execute File Movement

Once satisfied with the triage plan, add `--execute` to apply changes:

```bash
ordinale --scan "C:/Users/username/Downloads" --target "C:/Users/username/Documents/Organized" --execute
```

- Target folders are created automatically.
- Collisions are safely resolved using SHA-256 deduplication: identical files are skipped, while different files sharing the same name receive incremental suffixes (e.g., `statement (1).pdf`).
- An atomic move ledger is recorded to `.organizer_manifest.json` in the destination directory.

---

### 3. Interactive Review for Low-Confidence Documents

When sorting messy archives where documents may be ambiguous, use `--interactive` alongside `--execute`:

```bash
ordinale --scan "./messy_archive" --target "./organized" --execute --interactive
```

When a document's classification confidence falls below the configured threshold, Ordinale pauses and displays:
- Document snippet preview.
- Top candidate categories with scores.
- Interactive prompt allowing you to accept the prediction, assign a different category, or skip the document.

---

### 4. Undo / Rollback

If you need to restore files to their exact pre-move locations:

```bash
ordinale --target "C:/Users/username/Documents/Organized" --undo
```

Ordinale reads `.organizer_manifest.json`, validates file integrity, restores all files to their original paths, and updates the manifest.

---

### 5. Multi-Core Concurrency & Worker Tuning

For directories containing thousands of documents, scale the worker threads to saturate available CPU cores:

```bash
# Analyze with 12 parallel threads:
ordinale --scan "/path/to/large_archive" --target "/path/to/target" --workers 12 --execute

# Force single-threaded processing (useful for low-memory or debugging):
ordinale --scan "/path/to/large_archive" --target "/path/to/target" --sequential --execute
```

---

### 6. Preserving or Flattening Directory Hierarchies

- **Preserve Hierarchy (Default)**: If scanning `Downloads/2023/receipt.pdf`, it is routed to `Target/Receipts & Invoices/2023/receipt.pdf`.
- **Flatten Hierarchy (`--no-preserve-folders`)**: Routes files directly into destination categories without reproducing source subdirectories:
  ```bash
  ordinale --scan "./messy_tree" --target "./organized" --no-preserve-folders --execute
  ```

---

## Related Documentation

- [Configuration Guide](configuration.md) – Set defaults, configure custom file extensions and type mappings.
- [CUDA & GPU Acceleration](cuda.md) – Run classification models on NVIDIA GPUs.
- [Classification Taxonomy & Heuristics](classification.md) – Deep dive into routing logic and retention scores.
