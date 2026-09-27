# Document Classification & Heuristics

Ordinale uses the Laya decision model alongside rule-based heuristic routing and natural language processing (spaCy) to categorize and route documents.

[← Back to README](../README.md) • [Configuration Guide](configuration.md)

---

## 1. In-Memory Snippet Extraction

Before classification, documents pass through `DocumentTextExtractor`:

- **Supported Formats**: `.docx`, `.pdf`, `.html`, `.htm`, `.mhtml`, `.txt`, `.md`, `.markdown`, and `.rtf`.
- **In-Memory Only**: Document text is parsed entirely in RAM. **Ordinale never writes intermediate text snippets or temporary unencrypted plaintext dumps to disk.**
- **Window Size**: Reads only the first 1–2 pages or representative header/body snippets (typically 400 to 1,200 characters), which provides optimal context for classification without consuming excess memory or inference time.

---

## 2. Primary Classification Categories

Ordinale classifies documents across nine primary functional categories:

| Category | Typical Document Types |
| :--- | :--- |
| **`Financial`** | Tax assessments, banking statements, payroll stubs, investment portfolios, loan agreements. |
| **`Receipts & Invoices`** | Point-of-sale purchase receipts, utility bills, vendor invoices, payment confirmations. |
| **`Contracts & Legal`** | Non-disclosure agreements (NDAs), residential leases, employment contracts, terms of service. |
| **`Education & Academic`** | Coursework assignments, homework, syllabi, arXiv preprints, published journal articles. |
| **`Web Articles & Clippings`** | Saved HTML articles, blog snapshots, newsletters, documentation pages. |
| **`Career & Resumes`** | Resumes, curriculum vitae (CVs), cover letters, portfolio summaries. |
| **`Personal & Identity`** | Medical records, insurance policies, vehicle registrations, identity verifications. |
| **`Manuals & Guides`** | Technical user manuals, appliance setup instructions, API documentation, spec sheets. |
| **`Notes & Drafts`** | Scratchpad notes, meeting minutes, rough drafts, action item checklists. |

---

## 3. Heuristic Sub-Routing

For ambiguous or broad categories, Ordinale applies secondary rule-based disambiguation engines:

### Financial Sub-Routing
Documents classified as `Financial` are routed into dedicated sub-folders:
- `Financial/Taxes & Government`: Notice of Assessments (NOA), T4/T5/W-2 tax slips, IRS/CRA correspondence.
- `Financial/Banking`: Bank account statements, wire transfers, credit card agreements.
- `Financial/Investments`: Brokerage trade confirmations, 401(k) / RRSP statements, dividend summaries.
- `Financial/Payroll`: Pay stubs, direct deposit forms, wage statements.
- `Financial/General`: General financial disclosures and loan agreements.

### Academic vs. Coursework Disambiguation
Within `Education & Academic`, Ordinale separates student coursework from published research:
- **Coursework Indicators**: Regex matches for university course codes (e.g. `CS 240`, `BIO:101`, `MATH 135`), assignment headers (`Homework #3`, `Problem Set`, `Syllabus`, `Due Date`), and student identifiers.
- **Academic Paper Indicators**: Preprint servers (`arXiv`, `bioRxiv`, `medRxiv`, `SSRN`), scientific publishers (`IEEE`, `ACM`, `Springer`, `Elsevier`, `Nature`), DOIs, and abstract/citation formatting.

### Heuristics for Notes & Drafts
Identifies rough drafts and scratchpads through:
- File format priors (`.txt`, `.md`, `.rtf`).
- Meeting markers (`Attendees:`, `Agenda:`, `Meeting Minutes`, `Action Items:`).
- Checklist and task patterns (`- [ ]`, `TODO:`, `FIXME:`).
- Informal bullet structures and short, fragmented sentence lengths.

---

## 4. Metadata Scoring: Retention & Privacy

Along with folder classification, Ordinale assigns two metadata signals to each document:

### Retention Scoring
Scores the document's operational lifecycle:
- **`0` (Ephemeral / Prunable)**: Temporary files, scratchpads, low-value receipts, or draft notes that can be pruned during periodic cleanup.
- **`1` (Active Reference)**: Ongoing projects, course notes, active lease agreements, or operational guides.
- **`2` (Permanent Archive)**: Critical documents requiring long-term preservation (e.g., tax filings, birth certificates, deeds, signed contracts).

### Privacy & Sensitivity Flags
Scans extracted snippets for high-risk identity indicators:
- Social Security Numbers (SSN) / Social Insurance Numbers (SIN).
- Bank routing and account numbers.
- Explicit confidentiality notices (e.g., `CONFIDENTIAL AND PROPRIETARY`).

When flagged as sensitive, Ordinale highlights the file in the dry-run table with a warning badge and assigns the triage action `MOVE (SECURE)`.
