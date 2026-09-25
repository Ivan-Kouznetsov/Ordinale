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


def get_default_settings() -> Settings:
    """Returns a Settings instance with default values."""
    return Settings(scanner=ScannerConfig(), model=ModelConfig())


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

    return Settings(
        scanner=ScannerConfig(
            extensions=extensions,
            custom_types=custom_types,
        ),
        model=model_cfg,
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
