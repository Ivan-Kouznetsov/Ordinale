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
