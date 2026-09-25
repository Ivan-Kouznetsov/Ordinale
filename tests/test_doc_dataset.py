"""Tests for sample_documents.json integrity and coverage."""

import json
from pathlib import Path
import pytest

from ordinale.doc_classifier import CATEGORY_DESTINATIONS, FINANCIAL_SUBDESTINATIONS, EDUCATION_SUBDESTINATIONS


@pytest.fixture
def sample_documents() -> list[dict]:
    dataset_path = Path(__file__).parent / "fixtures" / "sample_documents.json"
    assert dataset_path.exists(), f"sample_documents.json not found at {dataset_path}"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_non_empty(sample_documents: list[dict]):
    assert len(sample_documents) >= 12, "Dataset should have at least 12 diverse document samples"


def test_unique_ids(sample_documents: list[dict]):
    ids = [item["id"] for item in sample_documents]
    assert len(ids) == len(set(ids)), "All document sample IDs must be unique"


def test_required_fields(sample_documents: list[dict]):
    required = {"id", "filename", "title", "content", "expected_category"}
    for item in sample_documents:
        missing = required - set(item.keys())
        assert not missing, f"Item {item.get('id')} is missing fields: {missing}"
        assert item["filename"].strip()
        assert item["content"].strip()


def test_valid_categories(sample_documents: list[dict]):
    all_categories = set(CATEGORY_DESTINATIONS.keys()) | {"financial"}
    for item in sample_documents:
        cat = item["expected_category"]
        assert cat in all_categories, f"Invalid category '{cat}' in item {item['id']}"


def test_financial_subcategories(sample_documents: list[dict]):
    financial_items = [d for d in sample_documents if d["expected_category"] == "financial"]
    assert len(financial_items) >= 4, "Should have multiple financial samples"

    # Must specifically include taxes_government for CRA notices
    tax_items = [d for d in financial_items if d.get("expected_financial_type") == "taxes_government"]
    assert len(tax_items) >= 2, "Should have government/CRA tax assessment and demand letter samples"

    for item in financial_items:
        fin_type = item.get("expected_financial_type")
        assert fin_type in FINANCIAL_SUBDESTINATIONS, f"Invalid financial subcategory '{fin_type}' in {item['id']}"


def test_education_subcategories(sample_documents: list[dict]):
    education_items = [d for d in sample_documents if d["expected_category"] == "education_academic"]
    assert len(education_items) >= 3, "Should have at least 3 education/academic samples"

    school_items = [d for d in education_items if d.get("expected_education_type") == "school"]
    academic_items = [d for d in education_items if d.get("expected_education_type") == "academic"]

    assert len(school_items) >= 2, "Should have school coursework samples"
    assert len(academic_items) >= 1, "Should have academic research paper samples"

    for item in education_items:
        edu_type = item.get("expected_education_type")
        assert edu_type in EDUCATION_SUBDESTINATIONS, f"Invalid education subcategory '{edu_type}' in {item['id']}"
