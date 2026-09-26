"""Document categorization and triage powered by Laya.

Evaluates document category, government/financial subcategories, retention score,
and privacy sensitivity in a single forward pass without token generation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import time
import warnings

from ordinale.config import (
    CourseCodeHeuristicsConfig,
    EducationAcademicHeuristicsConfig,
    HeuristicsConfig,
    Settings,
)

# Disable symlinks on Windows to prevent WinError 1314 when running without Admin/Developer mode
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Suppress known upstream Laya checkpoint temperature calibration warning
warnings.filterwarnings(
    "ignore",
    message=r".*checkpoint ships invalid temperatures.*",
    category=RuntimeWarning,
)


# Question criteria specifications for Laya
DOCUMENT_QUESTIONS: Dict[str, Any] = {
    "category": {
        "type": "choice",
        "instructions": "Which high-level category does this document belong to?",
        "criteria": {
            "financial": "bank statements, government tax documents, CRA notices, investments, pay stubs, loan agreements",
            "receipts": "purchase receipts, order confirmations, vendor invoices, utility bills, travel expenses",
            "contracts": "contracts, non-disclosure agreements (NDAs), residential leases, employment agreements, terms of service",
            "education_academic": "school homework, university problem sets, lab reports, essay drafts, course syllabi, lecture notes, peer-reviewed research papers, conference preprints (arXiv), academic journal articles, literature reviews",
            "web_snapshots": "saved web pages, HTML articles, blog clippings, online newsletter exports",
            "resumes": "resumes, curriculum vitae (CV), cover letters, job applications, professional bios",
            "personal_id": "medical records, insurance policies, vehicle registration, official government identity forms",
            "manuals": "product user manuals, appliance guides, technical documentation, hardware setup instructions",
            "scratch_notes": "scratchpad notes, quick rough meeting minutes, brain dumps, temporary export files",
        },
    },
    "financial_type": {
        "type": "choice",
        "instructions": "If this document is financial, what specific subcategory does it belong to?",
        "criteria": {
            "taxes_government": "Canada Revenue Agency (CRA) Notice of Assessment, late tax demand letters, installment reminders, T4/T5 tax slips, tax returns",
            "banking": "monthly checking and savings account statements, credit card statements, overdraft notices",
            "investments": "stock brokerage statements, TFSA / RRSP / 401(k) / IRA portfolio reports, trade confirmations",
            "payroll": "employee pay stubs, direct deposit advice, salary earnings statements",
            "general": "loan agreements, credit scores, debt collection, general accounting documents",
        },
    },
    "retention": {
        "type": "score",
        "instructions": "Rate how critical it is to retain this document permanently vs safe to prune eventually (0=disposable/short-lived, 1=active reference, 2=permanent archive).",
        "criteria": [
            "Disposable or short-lived (ephemeral notes, quick web clippings, temporary order receipts)",
            "Reference material (manuals, school assignments, guides, reading papers)",
            "Permanent archive (tax assessments, CRA notices, legal contracts, identity documents, bank statements)",
        ],
    },
    "is_sensitive": {
        "type": "noul",
        "instructions": "Does this document contain sensitive personal, financial, tax (SIN/SSN), or confidential legal information?",
    },
}

# Subfolder destination mappings
CATEGORY_DESTINATIONS: Dict[str, str] = {
    "receipts": "Receipts & Invoices",
    "contracts": "Contracts & Legal",
    "education_academic": "Education & Academic",
    "school": "Education & Academic/School Coursework",
    "academic": "Education & Academic/Research Papers",
    "web_snapshots": "Web Articles & Clippings",
    "resumes": "Career & Resumes",
    "personal_id": "Personal & Identity",
    "manuals": "Manuals & Guides",
    "scratch_notes": "Notes & Drafts",
}

FINANCIAL_SUBDESTINATIONS: Dict[str, str] = {
    "taxes_government": "Financial/Taxes & Government",
    "banking": "Financial/Banking",
    "investments": "Financial/Investments",
    "payroll": "Financial/Payroll",
    "general": "Financial/General",
}

EDUCATION_SUBDESTINATIONS: Dict[str, str] = {
    "school": "Education & Academic/School Coursework",
    "academic": "Education & Academic/Research Papers",
}

# Patterns and lexicons for separating school essays/coursework vs academic research papers
COURSE_CODE_PATTERN = re.compile(
    r"\b[A-Za-z]{2,4}[\s:\-]?\d{3,4}[A-Za-z]?\b", re.IGNORECASE
)

COURSEWORK_TERMS = [
    "homework",
    "problem set",
    "assignment",
    "lab report",
    "lab section",
    "course syllabus",
    "syllabus",
    "professor",
    "due date",
    "student id",
    "student name",
    "essay draft",
    "term paper",
    "term project",
    "midterm",
    "final exam",
    "class notes",
    "proposal",
    "project proposal",
]

PREPRINT_SERVERS = [
    "arxiv",
    "biorxiv",
    "medrxiv",
    "chemrxiv",
    "ssrn",
    "research square",
    "zenodo",
    "osf.io",
    "techrxiv",
]

JOURNAL_PUBLISHERS = [
    "ieee",
    "acm",
    "springer",
    "elsevier",
    "nature",
    "science",
    "plos",
    "pnas",
    "pubmed",
    "wiley",
    "cell press",
    "taylor & francis",
    "frontiers in",
    "mdpi",
    "iop publishing",
    "acm sig",
    "ieee trans",
]

ACADEMIC_PUBLICATION_PATTERNS = [
    re.compile(r"\bdoi:\s*10\.\d{4,9}/", re.IGNORECASE),
    re.compile(r"\bproceedings of\b", re.IGNORECASE),
    re.compile(r"\bjournal of\b", re.IGNORECASE),
    re.compile(r"\btransactions on\b", re.IGNORECASE),
    re.compile(r"\bconference on\b", re.IGNORECASE),
    re.compile(r"\bsymposium on\b", re.IGNORECASE),
    re.compile(r"\bannual meeting of\b", re.IGNORECASE),
    re.compile(r"\bpeer-reviewed\b", re.IGNORECASE),
    re.compile(r"\bet al\.\b", re.IGNORECASE),
    re.compile(r"\bbibliography\b", re.IGNORECASE),
]


COMMON_NON_COURSE_PREFIXES = {
    "year", "tax", "date", "line", "page", "item", "rule", "step", "room",
    "form", "part", "chap", "total", "fund", "suite", "apt", "unit", "post",
    "bill", "acct", "card", "call", "code", "dial", "dept", "dest", "rate",
    "file", "stat", "view", "cost", "plus", "paid", "fees", "gain", "loss",
    "note", "text", "term", "type", "user", "time", "hour", "mins", "secs",
}


NAME_PROXIMITY_PATTERN = re.compile(
    r"(?i)\b(student(\s+name)?|author|by|instructor|professor|teacher|advisor)\b[:\s]+[A-Za-z]"
)
TITLE_PROXIMITY_PATTERN = re.compile(
    r"(?i)\b(title|essay|paper|project|assignment|report|lab|thesis|dissertation)\b[:\s]+[A-Za-z]"
)


def is_valid_course_code(
    code_str: str,
    config: Optional[CourseCodeHeuristicsConfig] = None,
) -> bool:
    """Validates that a regex match is an actual university/school course code and not a date or invoice line."""
    m = re.match(r"^([A-Za-z]{2,4})[\s:\-]?(\d{3,4})([A-Za-z]?)$", code_str.strip())
    if not m:
        return False
    prefix = m.group(1).lower()
    num = m.group(2)
    non_course_prefixes = (
        set(config.non_course_prefixes) if config else COMMON_NON_COURSE_PREFIXES
    )
    if prefix in non_course_prefixes:
        return False
    common_prefixes = (
        set(config.common_prefixes)
        if config
        else {"cs", "comp", "it", "swe", "math", "phys", "chem", "bio", "eng", "hist", "stat", "econ", "psych"}
    )
    if num.startswith(("19", "20")) and prefix not in common_prefixes:
        return False
    return True


def evaluate_course_codes(
    prompt_text: str,
    config: Optional[CourseCodeHeuristicsConfig] = None,
) -> Dict[str, Any]:
    """Detects and scores candidate course codes with common-prefix and proximity heuristics."""
    if config is None:
        config = CourseCodeHeuristicsConfig()

    pattern = re.compile(config.pattern, re.IGNORECASE)
    lines = prompt_text.splitlines()

    candidates: List[Dict[str, Any]] = []
    seen_codes = set()

    for line_idx, line in enumerate(lines):
        for m in pattern.finditer(line):
            code_str = m.group(0).strip()
            norm_code = code_str.upper()
            if not is_valid_course_code(code_str, config):
                continue
            if norm_code in seen_codes:
                continue
            seen_codes.add(norm_code)

            match_prefix = re.match(r"^([A-Za-z]{2,4})", code_str)
            if not match_prefix:
                continue
            prefix = match_prefix.group(1).lower()
            is_common = prefix in config.common_prefixes

            # Context window around the matched line
            win_start = max(0, line_idx - config.proximity_window_lines)
            win_end = min(len(lines), line_idx + config.proximity_window_lines + 1)
            window_lines = lines[win_start:win_end]

            near_name = any(NAME_PROXIMITY_PATTERN.search(ln) for ln in window_lines)
            near_title = any(
                TITLE_PROXIMITY_PATTERN.search(ln)
                or ln.strip().startswith(("# ", "## ", "### "))
                or (ln.strip().startswith(('"', "“")) and ln.strip().endswith(('"', "”")))
                for ln in window_lines
            )

            # Header metadata checks (e.g. if Title: or Author: appears in first 8 lines)
            if line_idx <= 8:
                for top_ln in lines[:8]:
                    top_lower = top_ln.lower()
                    if top_lower.startswith("author:") or top_lower.startswith("student:"):
                        near_name = True
                    if top_lower.startswith("title:"):
                        near_title = True

            isolated = not (near_name or near_title)

            # Scoring calculation
            score = config.base_weight
            details: List[str] = []

            if is_common:
                score += config.common_prefix_bonus
                details.append("+common prefix")
            else:
                score -= config.uncommon_prefix_penalty
                details.append("-uncommon prefix")

            if near_name:
                score += config.proximity_name_bonus
                details.append("+near name")
            if near_title:
                score += config.proximity_title_bonus
                details.append("+near title")
            if isolated:
                score -= config.isolated_penalty
                details.append("-isolated")

            score = max(0.0, score)

            candidates.append({
                "code": code_str,
                "prefix": prefix,
                "is_common": is_common,
                "near_name": near_name,
                "near_title": near_title,
                "isolated": isolated,
                "score": round(score, 2),
                "details": details,
            })

    if not candidates:
        return {
            "valid_codes": [],
            "candidates": [],
            "total_score": 0.0,
        }

    total_score = max(c["score"] for c in candidates)
    if len(candidates) > 1:
        total_score += min(2.0, 0.5 * (len(candidates) - 1))

    return {
        "valid_codes": [c["code"] for c in candidates],
        "candidates": candidates,
        "total_score": round(total_score, 2),
    }


def disambiguate_education_academic(
    prompt_text: str,
    file_name: Optional[str] = None,
    file_extension: Optional[str] = None,
    heuristics: Optional[EducationAcademicHeuristicsConfig] = None,
    record_signals: bool = True,
    detailed_signals: bool = False,
) -> Dict[str, Any]:
    """Disambiguates between school essays/coursework and academic research papers.

    Evaluates:
      1. Course codes & proximity heuristics -> School.
      2. Names of preprint servers (arXiv, bioRxiv, SSRN) & journals/publishers -> Academic.
      3. File format priors: .pdf favors published/prepub academic papers; .docx/.txt favors school coursework.
    """
    if heuristics is None:
        heuristics = EducationAcademicHeuristicsConfig()

    text_lower = prompt_text.lower()

    if not file_name:
        match = re.search(r"File:\s*([^\r\n]+)", prompt_text)
        if match:
            file_name = match.group(1).strip()

    ext = (file_extension or (Path(file_name).suffix.lower() if file_name else "")).lower()

    score_school = 0.0
    score_academic = 0.0
    signals: List[str] = []

    # 1. Course Code Detection & Proximity Evaluation
    course_eval = evaluate_course_codes(prompt_text, config=heuristics.course_codes)
    valid_course_codes = course_eval["valid_codes"]
    course_code_score = course_eval["total_score"]

    if valid_course_codes:
        score_school += course_code_score
        if record_signals:
            if detailed_signals:
                for c in course_eval["candidates"][:3]:
                    signals.append(
                        f"Course code: {c['code']} ({', '.join(c['details'])}, score: {c['score']:.1f})"
                    )
            else:
                signals.append(f"Course code: {valid_course_codes[:3]}")

    # Coursework terms
    coursework_hits = [term for term in heuristics.coursework_terms if term in text_lower]
    if coursework_hits:
        term_weight = min(
            heuristics.coursework_term_max_score,
            len(coursework_hits) * heuristics.coursework_term_weight,
        )
        score_school += term_weight
        if record_signals:
            signals.append(f"Coursework terms: {coursework_hits[:3]}")

    # 2. Preprint servers & Journals
    fn_lower = file_name.lower() if file_name else ""
    found_preprints = [p for p in heuristics.preprint_servers if p in text_lower or p in fn_lower]
    if found_preprints:
        score_academic += heuristics.preprint_weight
        if record_signals:
            signals.append(f"Preprint server: {found_preprints}")

    found_journals = [j for j in heuristics.journal_publishers if j in text_lower or j in fn_lower]
    if found_journals:
        score_academic += heuristics.journal_weight
        if record_signals:
            signals.append(f"Academic publisher/journal: {found_journals}")

    compiled_patterns = [
        p if hasattr(p, "search") else re.compile(p, re.IGNORECASE)
        for p in heuristics.academic_publication_patterns
    ]
    found_academic_markers = [
        pat.pattern for pat in compiled_patterns if pat.search(prompt_text)
    ]
    if found_academic_markers:
        score_academic += min(
            heuristics.academic_marker_max_score,
            len(found_academic_markers) * heuristics.academic_marker_weight,
        )
        if record_signals:
            signals.append(f"Academic markers: {len(found_academic_markers)}")

    # 3. File format prior
    if ext == ".pdf":
        score_academic += heuristics.format_prior_pdf
        if record_signals:
            signals.append("PDF format prior (+academic)")
    elif ext in (".docx", ".doc", ".txt", ".rtf", ".odt", ".md", ".markdown"):
        score_school += heuristics.format_prior_school
        if record_signals:
            signals.append(f"{ext} format prior (+school)")

    diff = score_academic - score_school
    p_academic = 1.0 / (1.0 + math.exp(-diff))
    p_school = 1.0 - p_academic

    if p_academic >= 0.5:
        winner = "academic"
        conf = round(p_academic, 4)
    else:
        winner = "school"
        conf = round(p_school, 4)

    evidence = {
        "course_codes": valid_course_codes,
        "course_code_score": round(course_code_score, 2),
        "course_candidates": course_eval["candidates"],
        "coursework_hits": coursework_hits,
        "preprints": found_preprints,
        "journals": found_journals,
        "academic_markers": len(found_academic_markers),
    }

    return {
        "winner": winner,
        "confidence": conf,
        "probabilities": {
            "academic": round(p_academic, 4),
            "school": round(p_school, 4),
        },
        "signals": signals if record_signals else [],
        "evidence": evidence,
        "scores": {"academic": round(score_academic, 2), "school": round(score_school, 2)},
    }


class CudaDeviceError(RuntimeError):
    """Raised when CUDA is explicitly specified but cannot be used or fails."""
    pass


def autodetect_device() -> str:
    """Autodetects the best available hardware device ('cuda', 'mps', or 'cpu') that can execute tensors."""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                # Verify that the device can successfully allocate a tensor and execute a kernel
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                return "cuda"
            except Exception:
                pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEFAULT_MODEL_ID: str = "convaiinnovations/laya"


def resolve_cached_model_path(
    repo_id_or_path: str = DEFAULT_MODEL_ID,
    subfolder: Optional[str] = None,
) -> Optional[str]:
    """Resolves the local cached directory path for a model if completely cached.

    Args:
        repo_id_or_path: Hugging Face repo ID or local directory path.
        subfolder: Optional subfolder within the repository.

    Returns:
        Local directory path as string if cached, None otherwise.
    """
    if not repo_id_or_path:
        return None

    # 1. Local directory check
    target_path = Path(repo_id_or_path)
    if target_path.exists() and target_path.is_dir():
        if subfolder:
            target_path = target_path / subfolder
            if not target_path.is_dir():
                return None
        cfg_file = target_path / "rl_agent_config.json"
        weights_file = target_path / "model.safetensors"
        if cfg_file.is_file() and weights_file.is_file():
            return str(target_path)
        return None

    # 2. Hugging Face Hub local cache inspection (silent, zero network/progress bar overhead)
    try:
        from huggingface_hub import try_to_load_from_cache

        cfg_filename = f"{subfolder}/rl_agent_config.json" if subfolder else "rl_agent_config.json"
        weights_filename = f"{subfolder}/model.safetensors" if subfolder else "model.safetensors"

        cfg_path = try_to_load_from_cache(repo_id_or_path, cfg_filename)
        weights_path = try_to_load_from_cache(repo_id_or_path, weights_filename)

        if (
            cfg_path
            and isinstance(cfg_path, str)
            and os.path.isfile(cfg_path)
            and weights_path
            and isinstance(weights_path, str)
            and os.path.isfile(weights_path)
        ):
            if subfolder:
                return os.path.dirname(os.path.dirname(cfg_path))
            return os.path.dirname(cfg_path)
    except Exception:
        pass
    return None


def is_model_cached(
    repo_id_or_path: str = DEFAULT_MODEL_ID,
    subfolder: Optional[str] = None,
) -> bool:
    """Checks whether the specified model checkpoint is already completely cached locally.

    Args:
        repo_id_or_path: Hugging Face repo ID or local directory path.
        subfolder: Optional subfolder within the repository.

    Returns:
        True if all required model files are present locally, False otherwise.
    """
    return resolve_cached_model_path(repo_id_or_path, subfolder=subfolder) is not None


def configure_offline_mode(
    repo_id_or_path: str = DEFAULT_MODEL_ID,
    subfolder: Optional[str] = None,
    offline: Optional[Union[bool, str]] = "auto",
) -> bool:
    """Configures environment variables for offline mode if cached or explicitly requested.

    Args:
        repo_id_or_path: Hugging Face repo ID or local path.
        subfolder: Optional subfolder.
        offline: True (force offline), False (force online), or 'auto' (detect if cached).

    Returns:
        True if offline mode was enabled, False if online mode will be used.
    """
    should_be_offline: bool

    if isinstance(offline, str):
        offline_val = offline.strip().lower()
        if offline_val in ("1", "true", "yes", "on"):
            should_be_offline = True
        elif offline_val in ("0", "false", "no", "off"):
            should_be_offline = False
        else:  # "auto" or unrecognized
            should_be_offline = is_model_cached(repo_id_or_path, subfolder=subfolder)
    elif isinstance(offline, bool):
        should_be_offline = offline
    else:
        should_be_offline = is_model_cached(repo_id_or_path, subfolder=subfolder)

    if should_be_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        return False


class DocumentClassifier:
    """Evaluates documents using Laya's typed question engine."""

    def __init__(
        self,
        subfolder: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.55,
        offline: Optional[Union[bool, str]] = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        settings: Optional[Settings] = None,
        heuristics: Optional[HeuristicsConfig] = None,
    ) -> None:
        if settings is not None:
            self.heuristics = heuristics or settings.heuristics
            if model_id == DEFAULT_MODEL_ID and settings.model.model_id:
                model_id = settings.model.model_id
            if subfolder is None and settings.model.subfolder:
                subfolder = settings.model.subfolder
            if offline == "auto" and settings.model.offline:
                offline = settings.model.offline
        else:
            self.heuristics = heuristics or HeuristicsConfig()

        self.subfolder = subfolder
        self.model_id = model_id
        self.offline = offline
        self._explicit_device = device is not None and str(device).lower() != "auto"
        if not self._explicit_device:
            self.device = autodetect_device()
        else:
            self.device = device
        self.confidence_threshold = confidence_threshold
        self._agent = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads and pre-warms the Laya decision model."""
        import laya
        import torch

        # Automatically configure offline mode if cached, unless overridden
        configure_offline_mode(
            repo_id_or_path=self.model_id,
            subfolder=self.subfolder,
            offline=self.offline,
        )

        # If CUDA is explicitly requested, strictly validate CUDA without falling back to CPU
        if self._explicit_device and self.device is not None and "cuda" in str(self.device).lower():
            if not torch.cuda.is_available():
                raise CudaDeviceError(
                    f"CUDA device '{self.device}' was explicitly specified, but torch.cuda.is_available() is False. "
                    "Cannot use CUDA. Exiting without falling back to CPU."
                )
            try:
                # Validate actual GPU kernel execution capability on this device
                test_tensor = torch.zeros(1, device=self.device)
                del test_tensor
            except Exception as e:
                raise CudaDeviceError(
                    f"CUDA device '{self.device}' was explicitly specified, but CUDA execution failed: {e}. "
                    "Exiting without falling back to CPU."
                ) from e

        load_kwargs: Dict[str, Any] = {}
        if self.subfolder is not None:
            load_kwargs["subfolder"] = self.subfolder
        if self.device is not None:
            load_kwargs["device"] = self.device

        model_target = self.model_id
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            cached_dir = resolve_cached_model_path(self.model_id, subfolder=self.subfolder)
            if cached_dir:
                model_target = cached_dir

        self._agent = laya.load(model_target, **load_kwargs)

        # Ensure Laya did not silently downgrade to CPU during initialization
        if self.device is not None and "cuda" in str(self.device).lower():
            agent_device = getattr(self._agent, "device", None)
            if agent_device is not None and getattr(agent_device, "type", "") != "cuda":
                if self._explicit_device:
                    raise CudaDeviceError(
                        f"CUDA was explicitly requested, but model device fell back to '{agent_device}'. "
                        "Exiting without falling back to CPU."
                    )
                self.device = str(getattr(agent_device, "type", "cpu"))

            # Prevent Laya from silently falling back to CPU during prediction
            if hasattr(self._agent, "model") and hasattr(self._agent.model, "to"):
                orig_to = self._agent.model.to
                def strict_to(*args, **kwargs):
                    if self._explicit_device and any(str(a).lower() == "cpu" or getattr(a, "type", "") == "cpu" for a in args):
                        raise CudaDeviceError(
                            "CUDA was explicitly requested, but model attempted to fall back to CPU. "
                            "Exiting without falling back to CPU."
                        )
                    return orig_to(*args, **kwargs)
                self._agent.model.to = strict_to

        # Pre-warm with a tiny dummy call
        try:
            self._agent.predict(
                "Document Title: Sample Invoice. Total due: $50.00",
                {"category": DOCUMENT_QUESTIONS["category"]},
            )
        except Exception as e:
            if self._explicit_device and self.device is not None and "cuda" in str(self.device).lower():
                raise CudaDeviceError(
                    f"CUDA execution failed during model warmup: {e}. Exiting without falling back to CPU."
                ) from e
            raise

        # Verify device remained on CUDA after pre-warming
        if self.device is not None and "cuda" in str(self.device).lower():
            agent_device = getattr(self._agent, "device", None)
            if agent_device is not None and getattr(agent_device, "type", "") != "cuda":
                if self._explicit_device:
                    raise CudaDeviceError(
                        f"CUDA was explicitly requested, but model fell back to '{agent_device}' during inference. "
                        "Exiting without falling back to CPU."
                    )
                self.device = str(getattr(agent_device, "type", "cpu"))

    def classify(self, prompt_text: str) -> Dict[str, Any]:
        """Runs a single forward pass over prompt_text evaluating all typed questions."""
        # Check strict CUDA device before inference
        if self._explicit_device and self.device is not None and "cuda" in str(self.device).lower():
            agent_device = getattr(self._agent, "device", None)
            if agent_device is not None and getattr(agent_device, "type", "") != "cuda":
                raise CudaDeviceError(
                    f"CUDA was explicitly requested, but model device fell back to '{agent_device}'. "
                    "Exiting without falling back to CPU."
                )

        start_time = time.perf_counter()
        try:
            raw_result = self._agent.predict(prompt_text, DOCUMENT_QUESTIONS)
        except Exception as e:
            if self._explicit_device and self.device is not None and "cuda" in str(self.device).lower():
                raise CudaDeviceError(
                    f"CUDA execution failed during inference: {e}. Exiting without falling back to CPU."
                ) from e
            raise
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Verify device remained on CUDA after inference
        if self._explicit_device and self.device is not None and "cuda" in str(self.device).lower():
            agent_device = getattr(self._agent, "device", None)
            if agent_device is not None and getattr(agent_device, "type", "") != "cuda":
                raise CudaDeviceError(
                    f"CUDA was explicitly requested, but model fell back to '{agent_device}' during inference. "
                    "Exiting without falling back to CPU."
                )

        answers = raw_result.get("answers", {})

        # 1. Primary Category
        cat_ans = answers.get("category", {})
        cat_winner = cat_ans.get("choice", "scratch_notes")
        cat_probs = cat_ans.get("probabilities", {})
        cat_conf = cat_ans.get("confidence", 0.0)

        # 2. Financial Subcategory
        fin_ans = answers.get("financial_type", {})
        fin_winner = fin_ans.get("choice", "general")
        fin_probs = fin_ans.get("probabilities", {})
        fin_conf = fin_ans.get("confidence", 0.0)

        # 3. Retention Score (0.0 to 2.0)
        ret_ans = answers.get("retention", {})
        ret_score = float(ret_ans.get("score", 1.0))
        ret_probs = ret_ans.get("probabilities", {})
        ret_conf = ret_ans.get("confidence", 0.0)

        if ret_score >= 1.5:
            ret_label = "Permanent Archive"
        elif ret_score >= 0.7:
            ret_label = "Active Reference"
        else:
            ret_label = "Ephemeral / Prunable"

        # 4. Sensitivity (noul binary probability)
        sens_ans = answers.get("is_sensitive", {})
        sens_prob = float(sens_ans.get("noul", 0.0))
        is_sensitive = sens_prob >= 0.5

        # 5. Compute Destination Path & Subcategory Disambiguation
        education_type = None

        # Check for explicit course code or preprint evidence
        course_cfg = self.heuristics.education_academic.course_codes
        course_eval = evaluate_course_codes(prompt_text, config=course_cfg)
        has_preprints = any(
            srv in prompt_text.lower()
            for srv in self.heuristics.education_academic.preprint_servers
        )

        # If unmistakable academic/coursework pattern is present and Laya was uncertain
        if cat_conf < self.confidence_threshold:
            if (course_eval["valid_codes"] and course_eval["total_score"] >= 2.0) or has_preprints:
                cat_winner = "education_academic"
                cat_conf = 0.90

        if cat_winner == "financial":
            target_subfolder = FINANCIAL_SUBDESTINATIONS.get(fin_winner, "Financial/General")
            if fin_conf > cat_conf:
                cat_conf = round(0.4 * cat_conf + 0.6 * fin_conf, 4)
        elif cat_winner == "education_academic":
            education_type = disambiguate_education_academic(
                prompt_text,
                heuristics=self.heuristics.education_academic,
                record_signals=self.heuristics.record_signals,
                detailed_signals=self.heuristics.detailed_signals,
            )
            target_subfolder = EDUCATION_SUBDESTINATIONS.get(
                education_type["winner"], "Education & Academic"
            )
            # Calibrate confidence if deterministic evidence is present
            evidence = education_type.get("evidence", {})
            has_strong_evidence = (
                evidence.get("course_code_score", 0.0) >= 2.0
                or bool(evidence.get("preprints"))
                or evidence.get("academic_markers", 0) > 0
                or any(
                    s.startswith("Course code:")
                    or s.startswith("Preprint server:")
                    or s.startswith("Academic markers:")
                    for s in education_type.get("signals", [])
                )
            )
            if has_strong_evidence:
                edu_conf = education_type["confidence"]
                cat_conf = max(cat_conf, min(0.95, round(0.2 * cat_conf + 0.8 * edu_conf, 4)))
            elif education_type["confidence"] > 0.8:
                cat_conf = max(cat_conf, min(0.85, round(0.4 * cat_conf + 0.6 * education_type["confidence"], 4)))
        else:
            target_subfolder = CATEGORY_DESTINATIONS.get(cat_winner, "Organized/Other")

        # 6. Safety & Triage Action
        if cat_conf < self.confidence_threshold:
            action = "NEEDS_REVIEW"
            reason = f"Low category confidence ({cat_conf:.2f} < {self.confidence_threshold:.2f})"
        elif is_sensitive and ret_score >= 1.5:
            action = "MOVE_SECURE"
            reason = f"Permanent sensitive record (P(sens)={sens_prob:.1%}, ret={ret_score:.1f})"
        else:
            action = "AUTO_MOVE"
            reason = f"High confidence match ({cat_winner}, P={cat_probs.get(cat_winner, 0.0):.1%})"

        return {
            "prompt_text": prompt_text,
            "category": {
                "winner": cat_winner,
                "confidence": cat_conf,
                "probabilities": cat_probs,
            },
            "financial_type": {
                "winner": fin_winner,
                "confidence": fin_conf,
                "probabilities": fin_probs,
            } if cat_winner == "financial" else None,
            "education_type": education_type,
            "retention": {
                "score": ret_score,
                "label": ret_label,
                "confidence": ret_conf,
                "probabilities": ret_probs,
            },
            "sensitivity": {
                "is_sensitive": is_sensitive,
                "probability": sens_prob,
            },
            "target_subfolder": target_subfolder,
            "triage_action": {
                "action": action,
                "reason": reason,
            },
            "latency_ms": latency_ms,
        }
