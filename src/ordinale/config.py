"""Configuration management for Ordinale.

Handles loading settings from TOML or JSON configuration files, discovering
workspace/user configuration paths, extension normalization, and default settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

DEFAULT_SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".docx": "docx",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".mhtml": "html",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rtf": "rtf",
}

CONFIG_FILENAMES = (
    "ordinale.toml",
    "ordinale.json",
    ".ordinalerc.toml",
    ".ordinalerc.json",
    ".ordinalerc",
)

DEFAULT_SETTINGS_PATH: Path = Path(__file__).resolve().parent / "default_settings.toml"


def normalize_extension(ext: str) -> str:
    """Normalizes a file extension string (e.g. ' PDF ' -> '.pdf')."""
    cleaned = ext.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


@dataclass
class ScannerConfig:
    """Scanner configuration options."""

    extensions: List[str] = field(
        default_factory=lambda: list(DEFAULT_SUPPORTED_EXTENSIONS.keys())
    )
    custom_types: Dict[str, str] = field(default_factory=dict)

    def get_effective_supported_extensions(self) -> Dict[str, str]:
        """Returns the full mapping of extension to parser type."""
        mapping: Dict[str, str] = {}
        for ext in self.extensions:
            norm_ext = normalize_extension(ext)
            if not norm_ext:
                continue
            if norm_ext in self.custom_types:
                mapping[norm_ext] = self.custom_types[norm_ext]
            elif norm_ext in DEFAULT_SUPPORTED_EXTENSIONS:
                mapping[norm_ext] = DEFAULT_SUPPORTED_EXTENSIONS[norm_ext]
            else:
                mapping[norm_ext] = "text"

        for ext, parser in self.custom_types.items():
            norm_ext = normalize_extension(ext)
            if norm_ext:
                mapping[norm_ext] = parser

        return mapping


@dataclass
class CourseCodeHeuristicsConfig:
    """Heuristic settings for detecting and scoring course codes."""

    pattern: str = r"\b[A-Za-z]{2,4}[\s:\-]?\d{3,4}[A-Za-z]?\b"
    common_prefixes: List[str] = field(
        default_factory=lambda: [
            "cs", "comp", "it", "swe", "cis", "se", "ds",
            "math", "stat", "phys", "chem", "bio", "biol",
            "eng", "engl", "hist", "econ", "psyc", "psych",
            "phil", "poli", "posc", "soc", "anth", "comm",
            "bus", "mgmt", "fin", "acct", "mktg", "art",
            "mus", "span", "fren", "germ", "chin", "lang",
            "med", "nurs", "law", "ed", "educ", "geog",
            "geol", "astr", "ee", "ece", "me", "ce", "che",
            "is", "inf", "sci", "lit", "ling", "arch", "env",
        ]
    )
    non_course_prefixes: List[str] = field(
        default_factory=lambda: [
            "year", "tax", "date", "line", "page", "item", "rule", "step", "room",
            "form", "part", "chap", "total", "fund", "suite", "apt", "unit", "post",
            "bill", "acct", "card", "call", "code", "dial", "dept", "dest", "rate",
            "file", "stat", "view", "cost", "plus", "paid", "fees", "gain", "loss",
            "note", "text", "term", "type", "user", "time", "hour", "mins", "secs",
        ]
    )
    proximity_window_lines: int = 3
    base_weight: float = 3.0
    common_prefix_bonus: float = 1.0
    uncommon_prefix_penalty: float = 1.5
    proximity_name_bonus: float = 1.0
    proximity_title_bonus: float = 1.0
    isolated_penalty: float = 1.0


@dataclass
class EducationAcademicHeuristicsConfig:
    """Heuristic settings for education and academic disambiguation."""

    course_codes: CourseCodeHeuristicsConfig = field(default_factory=CourseCodeHeuristicsConfig)
    coursework_terms: List[str] = field(
        default_factory=lambda: [
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
    )
    preprint_servers: List[str] = field(
        default_factory=lambda: [
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
    )
    journal_publishers: List[str] = field(
        default_factory=lambda: [
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
    )
    academic_publication_patterns: List[str] = field(
        default_factory=lambda: [
            r"\bdoi:\s*10\.\d{4,9}/",
            r"\bproceedings of\b",
            r"\bjournal of\b",
            r"\btransactions on\b",
            r"\bconference on\b",
            r"\bsymposium on\b",
            r"\bannual meeting of\b",
            r"\bpeer-reviewed\b",
            r"\bet al\.\b",
            r"\bbibliography\b",
        ]
    )
    coursework_term_weight: float = 1.5
    coursework_term_max_score: float = 4.0
    preprint_weight: float = 3.5
    journal_weight: float = 2.5
    academic_marker_weight: float = 1.5
    academic_marker_max_score: float = 3.0
    format_prior_pdf: float = 1.5
    format_prior_school: float = 1.5


@dataclass
class HeuristicsConfig:
    """Root container for heuristic rules and scoring."""

    record_signals: bool = True
    detailed_signals: bool = False
    education_academic: EducationAcademicHeuristicsConfig = field(
        default_factory=EducationAcademicHeuristicsConfig
    )


@dataclass
class ModelConfig:
    """Model and offline detection configuration options."""

    model_id: str = "convaiinnovations/laya"
    subfolder: Optional[str] = None
    offline: str = "auto"


@dataclass
class Settings:
    """Root configuration settings container."""

    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    heuristics: HeuristicsConfig = field(default_factory=HeuristicsConfig)


def get_default_settings() -> Settings:
    """Returns a Settings instance with default values."""
    return Settings(
        scanner=ScannerConfig(),
        model=ModelConfig(),
        heuristics=HeuristicsConfig(),
    )


def find_settings_file(
    explicit_path: Optional[Path | str] = None,
    search_dir: Optional[Path | str] = None,
    include_default: bool = True,
) -> Optional[Path]:
    """Finds a settings file from explicit path, current/search directory, user home directory,
    or falls back to the default settings file.

    Order of precedence:
    1. explicit_path (if provided and exists)
    2. search_dir (or current working directory)
    3. User config directory (~/.config/ordinale or %APPDATA%/ordinale)
    4. Bundled default settings file (default_settings.toml) if include_default is True
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists() and p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"Specified configuration file not found: {explicit_path}")

    start_dir = Path(search_dir).resolve() if search_dir else Path.cwd().resolve()

    curr: Optional[Path] = start_dir
    while curr is not None:
        for fname in CONFIG_FILENAMES:
            candidate = curr / fname
            if candidate.is_file():
                return candidate.resolve()
        parent = curr.parent
        if parent == curr:
            break
        curr = parent

    user_candidates: List[Path] = []
    if os.name == "nt":
        app_data = os.environ.get("APPDATA")
        if app_data:
            user_candidates.extend(
                [
                    Path(app_data) / "ordinale" / "config.toml",
                    Path(app_data) / "ordinale" / "config.json",
                ]
            )

    home = Path.home()
    user_candidates.extend(
        [
            home / ".config" / "ordinale" / "config.toml",
            home / ".config" / "ordinale" / "config.json",
        ]
    )

    for cand in user_candidates:
        if cand.is_file():
            return cand.resolve()

    if include_default and DEFAULT_SETTINGS_PATH.is_file():
        return DEFAULT_SETTINGS_PATH.resolve()

    return None


def _parse_dict_to_settings(data: Dict[str, Any]) -> Settings:
    """Constructs a validated Settings object from a raw dictionary."""
    scanner_data = data.get("scanner", {})
    if not isinstance(scanner_data, dict):
        scanner_data = {}

    extensions_raw = scanner_data.get("extensions")
    if extensions_raw is not None:
        if isinstance(extensions_raw, (list, tuple, set)):
            extensions = [
                normalize_extension(str(x))
                for x in extensions_raw
                if normalize_extension(str(x))
            ]
        else:
            extensions = list(DEFAULT_SUPPORTED_EXTENSIONS.keys())
    else:
        extensions = list(DEFAULT_SUPPORTED_EXTENSIONS.keys())

    custom_types_raw = scanner_data.get("custom_types", {})
    custom_types: Dict[str, str] = {}
    if isinstance(custom_types_raw, dict):
        for k, v in custom_types_raw.items():
            norm_k = normalize_extension(str(k))
            if norm_k:
                custom_types[norm_k] = str(v).lower()

    model_data = data.get("model", {})
    if not isinstance(model_data, dict):
        model_data = {}
    model_cfg = ModelConfig(
        model_id=str(model_data.get("model_id", "convaiinnovations/laya")),
        subfolder=model_data.get("subfolder"),
        offline=str(model_data.get("offline", "auto")),
    )

    heuristics_data = data.get("heuristics", {})
    if not isinstance(heuristics_data, dict):
        heuristics_data = {}

    edu_data = heuristics_data.get("education_academic", {})
    if not isinstance(edu_data, dict):
        edu_data = {}

    course_data = edu_data.get("course_codes", {})
    if not isinstance(course_data, dict):
        course_data = {}

    default_course = CourseCodeHeuristicsConfig()
    course_cfg = CourseCodeHeuristicsConfig(
        pattern=str(course_data.get("pattern", default_course.pattern)),
        common_prefixes=[
            str(x).strip().lower()
            for x in course_data.get("common_prefixes", default_course.common_prefixes)
            if str(x).strip()
        ],
        non_course_prefixes=[
            str(x).strip().lower()
            for x in course_data.get("non_course_prefixes", default_course.non_course_prefixes)
            if str(x).strip()
        ],
        proximity_window_lines=int(
            course_data.get("proximity_window_lines", default_course.proximity_window_lines)
        ),
        base_weight=float(course_data.get("base_weight", default_course.base_weight)),
        common_prefix_bonus=float(
            course_data.get("common_prefix_bonus", default_course.common_prefix_bonus)
        ),
        uncommon_prefix_penalty=float(
            course_data.get("uncommon_prefix_penalty", default_course.uncommon_prefix_penalty)
        ),
        proximity_name_bonus=float(
            course_data.get("proximity_name_bonus", default_course.proximity_name_bonus)
        ),
        proximity_title_bonus=float(
            course_data.get("proximity_title_bonus", default_course.proximity_title_bonus)
        ),
        isolated_penalty=float(
            course_data.get("isolated_penalty", default_course.isolated_penalty)
        ),
    )

    default_edu = EducationAcademicHeuristicsConfig()
    edu_cfg = EducationAcademicHeuristicsConfig(
        course_codes=course_cfg,
        coursework_terms=[
            str(x) for x in edu_data.get("coursework_terms", default_edu.coursework_terms)
        ],
        preprint_servers=[
            str(x).strip().lower()
            for x in edu_data.get("preprint_servers", default_edu.preprint_servers)
            if str(x).strip()
        ],
        journal_publishers=[
            str(x).strip().lower()
            for x in edu_data.get("journal_publishers", default_edu.journal_publishers)
            if str(x).strip()
        ],
        academic_publication_patterns=[
            str(x)
            for x in edu_data.get(
                "academic_publication_patterns", default_edu.academic_publication_patterns
            )
        ],
        coursework_term_weight=float(
            edu_data.get("coursework_term_weight", default_edu.coursework_term_weight)
        ),
        coursework_term_max_score=float(
            edu_data.get("coursework_term_max_score", default_edu.coursework_term_max_score)
        ),
        preprint_weight=float(
            edu_data.get("preprint_weight", default_edu.preprint_weight)
        ),
        journal_weight=float(
            edu_data.get("journal_weight", default_edu.journal_weight)
        ),
        academic_marker_weight=float(
            edu_data.get("academic_marker_weight", default_edu.academic_marker_weight)
        ),
        academic_marker_max_score=float(
            edu_data.get("academic_marker_max_score", default_edu.academic_marker_max_score)
        ),
        format_prior_pdf=float(
            edu_data.get("format_prior_pdf", default_edu.format_prior_pdf)
        ),
        format_prior_school=float(
            edu_data.get("format_prior_school", default_edu.format_prior_school)
        ),
    )

    default_heuristics = HeuristicsConfig()
    heuristics_cfg = HeuristicsConfig(
        record_signals=bool(
            heuristics_data.get("record_signals", default_heuristics.record_signals)
        ),
        detailed_signals=bool(
            heuristics_data.get("detailed_signals", default_heuristics.detailed_signals)
        ),
        education_academic=edu_cfg,
    )

    return Settings(
        scanner=ScannerConfig(
            extensions=extensions,
            custom_types=custom_types,
        ),
        model=model_cfg,
        heuristics=heuristics_cfg,
    )


def load_settings(
    config_path: Optional[Path | str] = None,
    search_dir: Optional[Path | str] = None,
) -> Settings:
    """Loads and parses settings from a file or falls back to defaults.

    Args:
        config_path: Explicit path to configuration file (optional).
        search_dir: Directory from which to search for configuration (optional).

    Returns:
        Settings instance.
    """
    resolved_file = find_settings_file(explicit_path=config_path, search_dir=search_dir)
    if resolved_file is None:
        return get_default_settings()

    content = resolved_file.read_text(encoding="utf-8")
    suffix = resolved_file.suffix.lower()

    data: Dict[str, Any] = {}
    if suffix in (".json",) or resolved_file.name.endswith(".json"):
        data = json.loads(content)
    elif suffix in (".toml",) or resolved_file.name.endswith(".toml"):
        if tomllib is None:
            raise ImportError(
                "TOML parsing requires Python 3.11+ or the 'tomli' package. "
                "Use a JSON configuration file instead or install tomli."
            )
        data = tomllib.loads(content)
    else:
        try:
            data = json.loads(content)
        except Exception:
            if tomllib is not None:
                data = tomllib.loads(content)
            else:
                raise ValueError(
                    f"Unsupported configuration file format for '{resolved_file}'. "
                    "Expected JSON or TOML format."
                )

    return _parse_dict_to_settings(data)
