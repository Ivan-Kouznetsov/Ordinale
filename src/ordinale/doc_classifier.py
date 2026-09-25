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
from typing import Any, Dict, List, Optional
import warnings

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


def is_valid_course_code(code_str: str) -> bool:
    """Validates that a regex match is an actual university/school course code and not a date or invoice line."""
    m = re.match(r"^([A-Za-z]{2,4})[\s:\-]?(\d{3,4})([A-Za-z]?)$", code_str.strip())
    if not m:
        return False
    prefix = m.group(1).lower()
    num = m.group(2)
    if prefix in COMMON_NON_COURSE_PREFIXES:
        return False
    if num.startswith(("19", "20")) and prefix not in {
        "cs", "math", "phys", "chem", "bio", "eng", "hist", "stat", "econ", "psych"
    }:
        return False
    return True


def disambiguate_education_academic(
    prompt_text: str,
    file_name: Optional[str] = None,
    file_extension: Optional[str] = None,
) -> Dict[str, Any]:
    """Disambiguates between school essays/coursework and academic research papers.

    Evaluates:
      1. Course codes (e.g. CS240, BIO:101, ENGL-102) & coursework terms -> School.
      2. Names of preprint servers (arXiv, bioRxiv, SSRN) & journals/publishers -> Academic.
      3. File format priors: .pdf favors published/prepub academic papers; .docx/.txt favors school coursework.
    """
    text_lower = prompt_text.lower()

    if not file_name:
        match = re.search(r"File:\s*([^\r\n]+)", prompt_text)
        if match:
            file_name = match.group(1).strip()

    ext = (file_extension or (Path(file_name).suffix.lower() if file_name else "")).lower()

    score_school = 0.0
    score_academic = 0.0
    signals: List[str] = []

    # 1. Course Code Detection (\w{2}(\s|:|-)\d{3,4})
    course_codes = COURSE_CODE_PATTERN.findall(prompt_text)
    valid_course_codes = [c for c in course_codes if is_valid_course_code(c)]
    if valid_course_codes:
        score_school += 3.5
        signals.append(f"Course code: {valid_course_codes[:3]}")

    # Coursework terms
    coursework_hits = [term for term in COURSEWORK_TERMS if term in text_lower]
    if coursework_hits:
        term_weight = min(4.0, len(coursework_hits) * 1.5)
        score_school += term_weight
        signals.append(f"Coursework terms: {coursework_hits[:3]}")

    # 2. Preprint servers & Journals
    fn_lower = file_name.lower() if file_name else ""
    found_preprints = [p for p in PREPRINT_SERVERS if p in text_lower or p in fn_lower]
    if found_preprints:
        score_academic += 3.5
        signals.append(f"Preprint server: {found_preprints}")

    found_journals = [j for j in JOURNAL_PUBLISHERS if j in text_lower or j in fn_lower]
    if found_journals:
        score_academic += 2.5
        signals.append(f"Academic publisher/journal: {found_journals}")

    found_academic_markers = [
        pat.pattern for pat in ACADEMIC_PUBLICATION_PATTERNS if pat.search(prompt_text)
    ]
    if found_academic_markers:
        score_academic += min(3.0, len(found_academic_markers) * 1.5)
        signals.append(f"Academic markers: {len(found_academic_markers)}")

    # 3. File format prior
    if ext == ".pdf":
        score_academic += 1.5
        signals.append("PDF format prior (+academic)")
    elif ext in (".docx", ".doc", ".txt", ".rtf", ".odt", ".md", ".markdown"):
        score_school += 1.5
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

    return {
        "winner": winner,
        "confidence": conf,
        "probabilities": {
            "academic": round(p_academic, 4),
            "school": round(p_school, 4),
        },
        "signals": signals,
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


class DocumentClassifier:
    """Evaluates documents using Laya's typed question engine."""

    def __init__(
        self,
        subfolder: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        self.subfolder = subfolder
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

        self._agent = laya.load("convaiinnovations/laya", **load_kwargs)

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
        course_codes = [c for c in COURSE_CODE_PATTERN.findall(prompt_text) if is_valid_course_code(c)]
        has_arxiv = "arxiv" in prompt_text.lower()

        # If unmistakable academic/coursework pattern is present and Laya was uncertain
        if cat_conf < self.confidence_threshold:
            if course_codes or has_arxiv:
                cat_winner = "education_academic"
                cat_conf = 0.90

        if cat_winner == "financial":
            target_subfolder = FINANCIAL_SUBDESTINATIONS.get(fin_winner, "Financial/General")
            if fin_conf > cat_conf:
                cat_conf = round(0.4 * cat_conf + 0.6 * fin_conf, 4)
        elif cat_winner == "education_academic":
            education_type = disambiguate_education_academic(prompt_text)
            target_subfolder = EDUCATION_SUBDESTINATIONS.get(
                education_type["winner"], "Education & Academic"
            )
            # Calibrate confidence if deterministic evidence is present
            has_strong_evidence = any(
                s.startswith("Course code:") or s.startswith("Preprint server:") or s.startswith("Academic markers:")
                for s in education_type.get("signals", [])
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
