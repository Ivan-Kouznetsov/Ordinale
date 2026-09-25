from ordinale.config import (
    DEFAULT_SETTINGS_PATH,
    ScannerConfig,
    Settings,
    find_settings_file,
    load_settings,
    normalize_extension,
)
from ordinale.doc_classifier import (
    CATEGORY_DESTINATIONS,
    EDUCATION_SUBDESTINATIONS,
    FINANCIAL_SUBDESTINATIONS,
    CudaDeviceError,
    DocumentClassifier,
    autodetect_device,
)
from ordinale.extractor import DocumentTextExtractor, ExtractedDocument
from ordinale.organizer_engine import OrganizationPlan, OrganizerEngine

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "ScannerConfig",
    "Settings",
    "find_settings_file",
    "load_settings",
    "normalize_extension",
    "autodetect_device",
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
