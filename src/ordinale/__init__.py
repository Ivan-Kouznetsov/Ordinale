from ordinale.config import (
    DEFAULT_SETTINGS_PATH,
    ModelConfig,
    ScannerConfig,
    Settings,
    find_settings_file,
    load_settings,
    normalize_extension,
)
from ordinale.doc_classifier import (
    CATEGORY_DESTINATIONS,
    DEFAULT_MODEL_ID,
    EDUCATION_SUBDESTINATIONS,
    FINANCIAL_SUBDESTINATIONS,
    CudaDeviceError,
    DocumentClassifier,
    autodetect_device,
    configure_offline_mode,
    is_model_cached,
    resolve_cached_model_path,
)
from ordinale.extractor import DocumentTextExtractor, ExtractedDocument
from ordinale.organizer_engine import OrganizationPlan, OrganizerEngine

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "ModelConfig",
    "ScannerConfig",
    "Settings",
    "find_settings_file",
    "load_settings",
    "normalize_extension",
    "autodetect_device",
    "configure_offline_mode",
    "is_model_cached",
    "resolve_cached_model_path",
    "DEFAULT_MODEL_ID",
    "CATEGORY_DESTINATIONS",
    "EDUCATION_SUBDESTINATIONS",
    "FINANCIAL_SUBDESTINATIONS",
    "CudaDeviceError",
    "DocumentClassifier",
    "DocumentTextExtractor",
    "ExtractedDocument",
    "OrganizationPlan",
    "OrganizerEngine",
]
