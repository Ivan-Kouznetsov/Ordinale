# Configuration Reference

Ordinale can be configured via a configuration file (`ordinale.toml` or `ordinale.json`) to customize scanned extensions, map custom file formats to text extractors, tune heuristic weights, and set default model behaviors.

[← Back to README](../README.md) • [CLI Reference & Workflows](cli_and_workflows.md)

---

## Configuration Discovery Order

Ordinale looks for settings in the following order of precedence:

1. **Explicit CLI Argument**: `--config /path/to/my_settings.toml`
2. **Current Working Directory**: `./ordinale.toml` or `./ordinale.json`
3. **User Home Directory**: `~/.config/ordinale/ordinale.toml` or `%USERPROFILE%\.config\ordinale\ordinale.toml`
4. **Bundled Defaults**: Built-in default rules shipped with the package.

---

## Configuration Sections

### 1. `[scanner]`
Configures which files are scanned.

```toml
[scanner]
# List of file extensions that Ordinale will discover and process.
# Extensions are case-insensitive and leading dots are optional.
extensions = [
    ".docx",
    ".pdf",
    ".html",
    ".htm",
    ".mhtml",
    ".txt",
    ".md",
    ".markdown",
    ".rtf"
]
```

### 2. `[scanner.custom_types]`
Maps custom or domain-specific file extensions to supported in-memory extractors.

Supported extractor types:
- `"pdf"`: PDF text extractor via `pypdf`.
- `"docx"`: Microsoft Word document extractor via `python-docx`.
- `"html"`: HTML / MHTML parser via `BeautifulSoup4`.
- `"markdown"`: Markdown structured parser.
- `"text"`: Plain text / log parser.
- `"rtf"`: Rich Text Format extractor via `striprtf`.

```toml
[scanner.custom_types]
# Map reStructuredText to markdown parser
".rst" = "markdown"

# Map application logs to plain text parser
".log" = "text"

# Map LaTeX sources to text parser
".tex" = "text"
```

### 3. `[model]`
Controls the classification model and offline behavior.

```toml
[model]
# Model identifier from Hugging Face Hub (default: "convaiinnovations/laya")
model_id = "convaiinnovations/laya"

# Offline mode policy:
# "auto" (default: checks local cache; runs offline if model is cached)
# "true" (strict offline mode, never attempts network requests)
# "false" (always checks Hugging Face Hub for newer snapshots)
offline = "auto"
```

### 4. `[heuristics]`
Controls rule-based heuristic signals and spaCy NLP analysis.

```toml
[heuristics]
record_signals = true
detailed_signals = false

[heuristics.nlp]
# Lightweight spaCy English model for named entity recognition & header classification
enabled = true
model_name = "en_core_web_sm"

[heuristics.education_academic]
coursework_term_weight = 1.5
preprint_weight = 3.5
journal_weight = 2.5
```

---

## Example JSON Configuration (`ordinale.json`)

If you prefer JSON over TOML:

```json
{
  "scanner": {
    "extensions": [".pdf", ".docx", ".txt", ".md", ".log"],
    "custom_types": {
      ".log": "text",
      ".rst": "markdown"
    }
  },
  "model": {
    "model_id": "convaiinnovations/laya",
    "offline": "auto"
  },
  "heuristics": {
    "nlp": {
      "enabled": true,
      "model_name": "en_core_web_sm"
    }
  }
}
```

---

## Overriding via Command Line

To test or apply a specific configuration profile without modifying local files:

```bash
ordinale --scan "C:/data/documents" --config "C:/configs/custom_rules.toml"
```
