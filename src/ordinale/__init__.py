"""Ordinale - Local document categorization and organizer powered by Laya."""

from ordinale.doc_classifier import (
    CATEGORY_DESTINATIONS,
    EDUCATION_SUBDESTINATIONS,
    FINANCIAL_SUBDESTINATIONS,
    CudaDeviceError,
    DocumentClassifier,
)
from ordinale.extractor import DocumentTextExtractor, ExtractedDocument
from ordinale.organizer_engine import OrganizationPlan, OrganizerEngine

__version__ = "0.1.0"

__all__ = [
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
