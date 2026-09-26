"""NLP preprocessing and entity extraction engine powered by spaCy.

Provides lazy-loaded spaCy pipeline management, filename tokenization/preprocessing,
structural text segmentation (front-page/header block), and entity recognition
(PERSON, ORG, University/College markers, Course Codes) to power enhanced heuristics.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default keywords to identify academic institutions / universities in ORG entities or text
DEFAULT_UNIVERSITY_KEYWORDS = [
    "university",
    "college",
    "institute of technology",
    "polytechnic",
    "faculty of",
    "department of",
    "school of",
    "academy",
    "campus",
]

# Patterns for filename delimiter and camelCase splitting
CAMEL_CASE_PATTERN = re.compile(r"([a-z])([A-Z])")
NUMBER_LETTER_PATTERN = re.compile(r"([A-Za-z])(\d+)|(\d+)([A-Za-z])")
DELIMITER_PATTERN = re.compile(r"[_\-.\s]+")


class SpacyManager:
    """Manages lazy loading and caching of the spaCy pipeline."""

    _nlp_instance: Any = None
    _loaded_model_name: Optional[str] = None
    _disabled_pipes: Tuple[str, ...] = ("parser", "lemmatizer", "textcat")

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if spacy package is importable."""
        try:
            import spacy  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def is_model_available(cls, model_name: str = "en_core_web_sm") -> bool:
        """Checks if the given spaCy model is installed and loadable."""
        if not cls.is_available():
            return False
        import spacy
        return spacy.util.is_package(model_name)

    @classmethod
    def get_nlp(
        cls,
        model_name: str = "en_core_web_sm",
        disable: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Returns a cached, lazy-loaded spaCy Language pipeline.

        Falls back gracefully to a blank English model if the requested model
        is not installed, or returns None if spacy is not installed.
        """
        if cls._nlp_instance is not None and cls._loaded_model_name == model_name:
            return cls._nlp_instance

        if not cls.is_available():
            return None

        import spacy

        pipes_to_disable = list(disable) if disable is not None else list(cls._disabled_pipes)

        try:
            nlp = spacy.load(model_name, disable=pipes_to_disable)
            cls._nlp_instance = nlp
            cls._loaded_model_name = model_name
            return nlp
        except Exception as exc:
            logger.warning(
                "Could not load spaCy model '%s' (%s). Falling back to blank English pipeline.",
                model_name,
                exc,
            )
            try:
                nlp = spacy.blank("en")
                cls._nlp_instance = nlp
                cls._loaded_model_name = "blank:en"
                return nlp
            except Exception:
                return None


class FilenamePreprocessor:
    """Cleans, normalizes, and tokenizes filenames for NLP and heuristic extraction."""

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Transforms raw filename into clean, readable text.

        Examples:
          - "CS101_John_Doe_Essay.docx" -> "CS 101 John Doe Essay"
          - "DistributedSystemsAssignment-2.pdf" -> "Distributed Systems Assignment 2"
          - "smith_jane_term_paper.txt" -> "smith jane term paper"
        """
        # Strip path and extension
        stem = Path(filename).stem

        # Split CamelCase (e.g. FinalProject -> Final Project)
        cleaned = CAMEL_CASE_PATTERN.sub(r"\1 \2", stem)

        # Split letter/digit boundaries (e.g. CS101 -> CS 101, Assignment2 -> Assignment 2)
        def _split_num_let(m: re.Match) -> str:
            if m.group(1) and m.group(2):
                return f"{m.group(1)} {m.group(2)}"
            elif m.group(3) and m.group(4):
                return f"{m.group(3)} {m.group(4)}"
            return m.group(0)

        cleaned = NUMBER_LETTER_PATTERN.sub(_split_num_let, cleaned)

        # Replace underscores, hyphens, and multiple dots with space
        cleaned = DELIMITER_PATTERN.sub(" ", cleaned).strip()

        return cleaned

    @classmethod
    def extract_filename_entities(
        cls,
        filename: str,
        nlp: Optional[Any] = None,
        common_coursework_terms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extracts candidate entities and keyword cues from a filename."""
        clean_text = cls.clean_filename(filename)
        lower_clean = clean_text.lower()

        terms = common_coursework_terms or [
            "essay",
            "assignment",
            "homework",
            "hw",
            "lab",
            "report",
            "project",
            "term paper",
            "thesis",
            "dissertation",
            "midterm",
            "final",
            "syllabus",
            "draft",
        ]

        found_terms = [t for t in terms if t in lower_clean]

        # Extract potential course codes from filename
        course_code_pat = re.compile(r"\b[A-Za-z]{2,4}\s*\d{3,4}[A-Za-z]?\b", re.IGNORECASE)
        found_codes = [m.group(0) for m in course_code_pat.finditer(clean_text)]

        # Extract PERSON entities using spaCy if available
        persons: List[str] = []
        orgs: List[str] = []

        if nlp is not None and hasattr(nlp, "pipe_names") and "ner" in nlp.pipe_names:
            doc = nlp(clean_text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    persons.append(ent.text.strip())
                elif ent.label_ == "ORG":
                    orgs.append(ent.text.strip())

        return {
            "raw_filename": filename,
            "cleaned_text": clean_text,
            "coursework_terms": found_terms,
            "course_codes": found_codes,
            "persons": persons,
            "orgs": orgs,
        }


class DocumentStructuralSegmenter:
    """Segments document text into structural sections (front-page header, body, references)."""

    DEFAULT_MAX_HEADER_LINES = 35
    DEFAULT_MAX_HEADER_CHARS = 1600

    @classmethod
    def segment_text(
        cls,
        text: str,
        max_header_lines: int = DEFAULT_MAX_HEADER_LINES,
        max_header_chars: int = DEFAULT_MAX_HEADER_CHARS,
    ) -> Dict[str, str]:
        """Segments text into front_page / header block and body."""
        if not text:
            return {"front_page": "", "body": "", "references": ""}

        lines = text.splitlines()
        header_lines: List[str] = []
        body_lines: List[str] = []
        ref_lines: List[str] = []

        in_refs = False
        ref_pattern = re.compile(
            r"^(references|bibliography|works cited|literature cited)\s*$",
            re.IGNORECASE,
        )

        char_count = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not in_refs and ref_pattern.match(stripped):
                in_refs = True

            if in_refs:
                ref_lines.append(line)
            elif idx < max_header_lines and char_count < max_header_chars:
                header_lines.append(line)
                char_count += len(line) + 1
            else:
                body_lines.append(line)

        return {
            "front_page": "\n".join(header_lines).strip(),
            "body": "\n".join(body_lines).strip(),
            "references": "\n".join(ref_lines).strip(),
        }


class DocumentEntityExtractor:
    """Extracts Named Entities and composite signals using spaCy and domain knowledge."""

    def __init__(
        self,
        nlp: Optional[Any] = None,
        university_keywords: Optional[List[str]] = None,
    ) -> None:
        self.nlp = nlp
        self.university_keywords = (
            [k.lower() for k in university_keywords]
            if university_keywords is not None
            else DEFAULT_UNIVERSITY_KEYWORDS
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts PERSON, ORG, and institutional entities from the given text."""
        persons: List[str] = []
        orgs: List[str] = []
        universities: List[str] = []
        dates: List[str] = []

        if not text.strip():
            return {
                "persons": [],
                "orgs": [],
                "universities": [],
                "dates": [],
            }

        active_nlp = self.nlp or SpacyManager.get_nlp()

        if active_nlp is not None and hasattr(active_nlp, "pipe_names") and "ner" in active_nlp.pipe_names:
            doc = active_nlp(text[:4000])  # Cap at first 4000 chars for speed
            for ent in doc.ents:
                ent_text = ent.text.strip(" \t\r\n,.;:-")
                if not ent_text:
                    continue
                if ent.label_ == "PERSON":
                    # Filter out short single-letter false positives
                    if len(ent_text) > 1 and ent_text not in persons:
                        persons.append(ent_text)
                elif ent.label_ == "ORG":
                    if ent_text not in orgs:
                        orgs.append(ent_text)
                        # Check if ORG matches higher education keywords
                        lower_ent = ent_text.lower()
                        if any(kw in lower_ent for kw in self.university_keywords):
                            universities.append(ent_text)
                elif ent.label_ == "DATE":
                    if ent_text not in dates:
                        dates.append(ent_text)

        # Fallback / augment: regex search for universities if NER missed them
        text_lower = text.lower()
        for kw in self.university_keywords:
            if kw in text_lower:
                # Find matching line or phrase
                for match in re.finditer(rf"\b([A-Z][A-Za-z\s&'-]+{kw}[A-Za-z\s&'-]*)\b", text, re.IGNORECASE):
                    val = match.group(0).strip()
                    if val and val not in universities and len(val) < 60:
                        universities.append(val)

        return {
            "persons": persons,
            "orgs": orgs,
            "universities": universities,
            "dates": dates,
        }

    def analyze_front_page(
        self,
        front_page_text: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs comprehensive front-page and filename analysis for student heuristics.

        Returns detected persons, universities, course codes, and synergy indicators.
        """
        active_nlp = self.nlp or SpacyManager.get_nlp()

        # 1. Filename entity analysis
        fn_data = (
            FilenamePreprocessor.extract_filename_entities(filename, nlp=active_nlp)
            if filename
            else {
                "raw_filename": "",
                "cleaned_text": "",
                "coursework_terms": [],
                "course_codes": [],
                "persons": [],
                "orgs": [],
            }
        )

        # 2. Front-page text entity analysis
        text_entities = self.extract_entities(front_page_text)

        # Merge persons (front page + filename)
        all_persons = list(dict.fromkeys(text_entities["persons"] + fn_data["persons"]))
        all_universities = text_entities["universities"]

        # 3. Synergy detection
        has_person = len(all_persons) > 0
        has_university = len(all_universities) > 0

        return {
            "filename_data": fn_data,
            "persons": all_persons,
            "universities": all_universities,
            "dates": text_entities["dates"],
            "has_person": has_person,
            "has_university": has_university,
        }
