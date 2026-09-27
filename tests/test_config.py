"""Unit tests for configuration management in ordinale.config."""

import json
from pathlib import Path
import pytest

from ordinale.config import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    ScannerConfig,
    Settings,
    find_settings_file,
    get_default_settings,
    load_settings,
    normalize_extension,
)
from ordinale.extractor import DocumentTextExtractor
from ordinale.organizer_engine import OrganizerEngine


def test_normalize_extension():
    assert normalize_extension("pdf") == ".pdf"
    assert normalize_extension(".PDF") == ".pdf"
    assert normalize_extension("  .DocX  ") == ".docx"
    assert normalize_extension("txt") == ".txt"
    assert normalize_extension("") == ""
    assert normalize_extension("   ") == ""


def test_get_default_settings():
    settings = get_default_settings()
    assert isinstance(settings, Settings)
    assert set(settings.scanner.extensions) == set(DEFAULT_SUPPORTED_EXTENSIONS.keys())
    effective = settings.scanner.get_effective_supported_extensions()
    assert effective[".pdf"] == "pdf"
    assert effective[".docx"] == "docx"


def test_load_settings_missing_explicit_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_settings(config_path=missing_file)


def test_load_settings_from_json(tmp_path: Path):
    config_file = tmp_path / "ordinale.json"
    data = {
        "scanner": {
            "extensions": ["pdf", ".DOCX", "rst"],
            "custom_types": {
                ".rst": "markdown",
                "custom": "text",
            },
        }
    }
    config_file.write_text(json.dumps(data), encoding="utf-8")

    settings = load_settings(config_path=config_file)
    assert settings.scanner.extensions == [".pdf", ".docx", ".rst"]
    assert settings.scanner.custom_types == {".rst": "markdown", ".custom": "text"}

    effective = settings.scanner.get_effective_supported_extensions()
    assert effective[".pdf"] == "pdf"
    assert effective[".docx"] == "docx"
    assert effective[".rst"] == "markdown"
    assert effective[".custom"] == "text"


def test_load_settings_from_toml(tmp_path: Path):
    toml_file = tmp_path / "ordinale.toml"
    toml_content = """
[scanner]
extensions = [".pdf", ".txt", ".log"]

[scanner.custom_types]
".log" = "text"
"""
    toml_file.write_text(toml_content, encoding="utf-8")

    settings = load_settings(config_path=toml_file)
    assert settings.scanner.extensions == [".pdf", ".txt", ".log"]
    assert settings.scanner.custom_types == {".log": "text"}

    effective = settings.scanner.get_effective_supported_extensions()
    assert effective[".pdf"] == "pdf"
    assert effective[".txt"] == "text"
    assert effective[".log"] == "text"


def test_find_settings_file_in_search_dir(tmp_path: Path):
    config_file = tmp_path / "ordinale.json"
    config_file.write_text(json.dumps({"scanner": {"extensions": [".pdf"]}}), encoding="utf-8")

    sub_dir = tmp_path / "subfolder"
    sub_dir.mkdir()

    found = find_settings_file(search_dir=sub_dir)
    assert found == config_file.resolve()

    settings = load_settings(search_dir=sub_dir)
    assert settings.scanner.extensions == [".pdf"]


def test_extractor_with_custom_settings():
    scanner_cfg = ScannerConfig(
        extensions=[".pdf", ".custom"],
        custom_types={".custom": "text"},
    )
    extractor = DocumentTextExtractor(
        supported_extensions=scanner_cfg.get_effective_supported_extensions()
    )

    assert extractor.is_supported("file.pdf")
    assert extractor.is_supported("file.custom")
    assert not extractor.is_supported("file.docx")
    assert not extractor.is_supported("file.txt")


def test_organizer_engine_with_settings(tmp_path: Path):
    settings = Settings(
        scanner=ScannerConfig(extensions=[".pdf", ".notes"])
    )
    engine = OrganizerEngine(settings=settings)

    (tmp_path / "doc1.pdf").write_text("pdf dummy", encoding="utf-8")
    (tmp_path / "doc2.docx").write_text("docx dummy", encoding="utf-8")
    (tmp_path / "doc3.notes").write_text("notes dummy", encoding="utf-8")

    scanned = engine.scan_directory(tmp_path)
    names = [p.name for p in scanned]

    assert "doc1.pdf" in names
    assert "doc3.notes" in names
    assert "doc2.docx" not in names


def test_organizer_engine_scan_directory_extensions_override(tmp_path: Path):
    engine = OrganizerEngine()

    (tmp_path / "doc1.pdf").write_text("pdf dummy", encoding="utf-8")
    (tmp_path / "doc2.docx").write_text("docx dummy", encoding="utf-8")
    (tmp_path / "doc3.txt").write_text("txt dummy", encoding="utf-8")

    # Ad-hoc override: scan only .pdf
    scanned = engine.scan_directory(tmp_path, extensions=[".pdf"])
    names = [p.name for p in scanned]

    assert names == ["doc1.pdf"]


def test_default_settings_file_exists_and_contains_all_extensions():
    from ordinale.config import DEFAULT_SETTINGS_PATH, DEFAULT_SUPPORTED_EXTENSIONS

    assert DEFAULT_SETTINGS_PATH.is_file()
    settings = load_settings(config_path=DEFAULT_SETTINGS_PATH)
    assert set(settings.scanner.extensions) == set(DEFAULT_SUPPORTED_EXTENSIONS.keys())


def test_load_settings_fallback_to_default_file():
    from ordinale.config import DEFAULT_SUPPORTED_EXTENSIONS

    # Without passing any path, load_settings should resolve default settings
    settings = load_settings()
    assert set(settings.scanner.extensions) == set(DEFAULT_SUPPORTED_EXTENSIONS.keys())
    assert ".pdf" in settings.scanner.extensions
    assert ".docx" in settings.scanner.extensions
    assert ".md" in settings.scanner.extensions


def test_model_config_defaults_and_custom(tmp_path: Path):
    from ordinale.config import ModelConfig

    settings = get_default_settings()
    assert isinstance(settings.model, ModelConfig)
    assert settings.model.model_id == "convaiinnovations/laya"
    assert settings.model.subfolder is None
    assert settings.model.offline == "auto"

    # Test loading model config from JSON
    config_file = tmp_path / "ordinale.json"
    config_file.write_text(
        json.dumps({
            "model": {
                "model_id": "custom/model",
                "subfolder": "multilingual",
                "offline": "true",
            }
        }),
        encoding="utf-8",
    )
    loaded = load_settings(config_path=config_file)
    assert loaded.model.model_id == "custom/model"
    assert loaded.model.subfolder == "multilingual"
    assert loaded.model.offline == "true"


def test_heuristics_config_defaults():
    from ordinale.config import (
        CourseCodeHeuristicsConfig,
        EducationAcademicHeuristicsConfig,
        HeuristicsConfig,
    )

    settings = get_default_settings()
    assert isinstance(settings.heuristics, HeuristicsConfig)
    assert settings.heuristics.record_signals is True
    assert settings.heuristics.detailed_signals is False

    edu = settings.heuristics.education_academic
    assert isinstance(edu, EducationAcademicHeuristicsConfig)
    assert "homework" in edu.coursework_terms
    assert "arxiv" in edu.preprint_servers

    assert settings.heuristics.nlp.enabled is True
    assert settings.heuristics.nlp.model_name == "en_core_web_sm"

    names = edu.names
    assert names.front_page_person_bonus == 1.5
    assert names.university_bonus == 1.5
    assert names.student_header_synergy_bonus == 2.5
    assert "university" in names.university_keywords

    course = edu.course_codes
    assert isinstance(course, CourseCodeHeuristicsConfig)
    assert "it" in course.common_prefixes
    assert "cs" in course.common_prefixes
    assert "nw" not in course.common_prefixes
    assert "tax" in course.non_course_prefixes


def test_load_heuristics_from_json(tmp_path: Path):
    config_file = tmp_path / "ordinale.json"
    data = {
        "heuristics": {
            "record_signals": False,
            "detailed_signals": True,
            "education_academic": {
                "course_codes": {
                    "common_prefixes": ["it", "nw", "eng"],
                    "proximity_name_bonus": 2.5,
                },
                "preprint_servers": ["customarxiv"],
            },
        }
    }
    config_file.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_settings(config_path=config_file)

    assert loaded.heuristics.record_signals is False
    assert loaded.heuristics.detailed_signals is True
    assert loaded.heuristics.education_academic.course_codes.common_prefixes == ["it", "nw", "eng"]
    assert loaded.heuristics.education_academic.course_codes.proximity_name_bonus == 2.5
    assert loaded.heuristics.education_academic.preprint_servers == ["customarxiv"]


def test_load_heuristics_from_toml(tmp_path: Path):
    toml_file = tmp_path / "ordinale.toml"
    toml_content = """
[heuristics]
record_signals = false
detailed_signals = true

[heuristics.education_academic]
preprint_servers = ["myarchive"]

[heuristics.education_academic.course_codes]
common_prefixes = ["it", "swe"]
common_prefix_bonus = 2.0
"""
    toml_file.write_text(toml_content, encoding="utf-8")
    loaded = load_settings(config_path=toml_file)

    assert loaded.heuristics.record_signals is False
    assert loaded.heuristics.detailed_signals is True
    assert loaded.heuristics.education_academic.preprint_servers == ["myarchive"]
    assert loaded.heuristics.education_academic.course_codes.common_prefixes == ["it", "swe"]
    assert loaded.heuristics.education_academic.course_codes.common_prefix_bonus == 2.0


def test_load_nlp_and_name_heuristics_custom_toml(tmp_path: Path):
    toml_file = tmp_path / "ordinale.toml"
    toml_content = """
[heuristics.nlp]
enabled = false
model_name = "custom_model"

[heuristics.education_academic.names]
front_page_person_bonus = 3.0
university_bonus = 2.0
student_header_synergy_bonus = 4.0
filename_name_bonus = 2.0
university_keywords = ["polytechnic", "academy"]
"""
    toml_file.write_text(toml_content, encoding="utf-8")
    loaded = load_settings(config_path=toml_file)

    assert loaded.heuristics.nlp.enabled is False
    assert loaded.heuristics.nlp.model_name == "custom_model"
    names = loaded.heuristics.education_academic.names
    assert names.front_page_person_bonus == 3.0
    assert names.university_bonus == 2.0
    assert names.student_header_synergy_bonus == 4.0
    assert names.filename_name_bonus == 2.0
    assert names.university_keywords == ["polytechnic", "academy"]
