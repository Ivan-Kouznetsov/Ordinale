"""Unit tests for DocumentClassifier and decision logic."""

from unittest.mock import MagicMock, patch
import pytest

from ordinale.doc_classifier import (
    DOCUMENT_QUESTIONS,
    CATEGORY_DESTINATIONS,
    FINANCIAL_SUBDESTINATIONS,
    EDUCATION_SUBDESTINATIONS,
    DocumentClassifier,
    disambiguate_education_academic,
)


def test_schema_keys():
    assert "category" in DOCUMENT_QUESTIONS
    assert "financial_type" in DOCUMENT_QUESTIONS
    assert "retention" in DOCUMENT_QUESTIONS
    assert "is_sensitive" in DOCUMENT_QUESTIONS


def test_category_schema_completeness():
    cat = DOCUMENT_QUESTIONS["category"]
    assert cat["type"] == "choice"
    criteria = cat["criteria"]
    expected_categories = {
        "financial",
        "receipts",
        "contracts",
        "education_academic",
        "web_snapshots",
        "resumes",
        "personal_id",
        "manuals",
        "scratch_notes",
    }
    assert set(criteria.keys()) == expected_categories


def test_financial_type_schema():
    fin = DOCUMENT_QUESTIONS["financial_type"]
    assert fin["type"] == "choice"
    criteria = fin["criteria"]
    assert "taxes_government" in criteria
    assert "CRA" in criteria["taxes_government"] or "Canada Revenue Agency" in criteria["taxes_government"]
    assert "banking" in criteria
    assert "investments" in criteria
    assert "payroll" in criteria


def test_destinations_mapped():
    for cat in CATEGORY_DESTINATIONS:
        assert CATEGORY_DESTINATIONS[cat].strip()

    assert FINANCIAL_SUBDESTINATIONS["taxes_government"] == "Financial/Taxes & Government"
    assert FINANCIAL_SUBDESTINATIONS["banking"] == "Financial/Banking"
    assert EDUCATION_SUBDESTINATIONS["school"] == "Education & Academic/School Coursework"
    assert EDUCATION_SUBDESTINATIONS["academic"] == "Education & Academic/Research Papers"


@patch("laya.load")
def test_classify_mocked_cra_notice(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "financial",
                "confidence": 0.95,
                "probabilities": {"financial": 0.95, "receipts": 0.03},
            },
            "financial_type": {
                "choice": "taxes_government",
                "confidence": 0.92,
                "probabilities": {"taxes_government": 0.92, "banking": 0.05},
            },
            "retention": {
                "score": 1.95,
                "confidence": 0.90,
                "probabilities": {"2": 0.95, "1": 0.05, "0": 0.0},
            },
            "is_sensitive": {
                "noul": 0.88,
                "confidence": 0.88,
            },
        }
    }

    classifier = DocumentClassifier()
    res = classifier.classify("File: CRA_Notice_of_Assessment.pdf\nTax Year 2025 Net Income $88,200")

    assert res["category"]["winner"] == "financial"
    assert res["financial_type"]["winner"] == "taxes_government"
    assert res["target_subfolder"] == "Financial/Taxes & Government"
    assert res["retention"]["label"] == "Permanent Archive"
    assert res["sensitivity"]["is_sensitive"] is True
    assert res["triage_action"]["action"] == "MOVE_SECURE"


@patch("laya.load")
def test_classify_mocked_school_assignment(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "education_academic",
                "confidence": 0.92,
                "probabilities": {"education_academic": 0.92, "contracts": 0.05},
            },
            "financial_type": {
                "choice": "general",
                "confidence": 0.1,
                "probabilities": {"general": 0.5},
            },
            "retention": {
                "score": 1.1,
                "confidence": 0.80,
                "probabilities": {"1": 0.85, "2": 0.10, "0": 0.05},
            },
            "is_sensitive": {
                "noul": 0.05,
                "confidence": 0.90,
            },
        }
    }

    classifier = DocumentClassifier()
    res = classifier.classify("File: CS101_Homework_3.docx\nImplement Binary Search Trees. Due Date: Oct 10.")

    assert res["category"]["winner"] == "education_academic"
    assert res["education_type"] is not None
    assert res["education_type"]["winner"] == "school"
    assert res["target_subfolder"] == "Education & Academic/School Coursework"
    assert res["retention"]["label"] == "Active Reference"
    assert res["sensitivity"]["is_sensitive"] is False
    assert res["triage_action"]["action"] == "AUTO_MOVE"


def test_disambiguate_education_academic_course_codes():
    # Test course codes pattern: \w{2}(\s|:|-)\d{3,4}
    res_cs = disambiguate_education_academic("File: essay.docx\nCourse: CS 240 Algorithms")
    assert res_cs["winner"] == "school"

    res_bio = disambiguate_education_academic("File: lab.docx\nBIO:101 Lab Report Section 4")
    assert res_bio["winner"] == "school"

    res_engl = disambiguate_education_academic("File: draft.txt\nENGL-102 Essay Draft")
    assert res_engl["winner"] == "school"


def test_disambiguate_education_academic_journals_and_preprints():
    # Test preprint servers (arXiv, bioRxiv)
    res_arxiv = disambiguate_education_academic("File: paper.pdf\narXiv:2301.00012 Deep Learning")
    assert res_arxiv["winner"] == "academic"

    # Test journals and publishers
    res_nature = disambiguate_education_academic("File: article.pdf\nPublished in Nature Biotechnology")
    assert res_nature["winner"] == "academic"

    res_ieee = disambiguate_education_academic("File: report.pdf\nProceedings of IEEE Conference on Robotics")
    assert res_ieee["winner"] == "academic"


def test_disambiguate_education_academic_format_priors():
    # Neutral text with .pdf format prior
    res_pdf = disambiguate_education_academic("File: manuscript.pdf\nAbstract: An empirical evaluation.")
    assert res_pdf["winner"] == "academic"

    # Neutral text with .docx format prior
    res_docx = disambiguate_education_academic("File: submission.docx\nDraft by Student")
    assert res_docx["winner"] == "school"


@patch("laya.load")
def test_classify_mocked_low_confidence(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "scratch_notes",
                "confidence": 0.40,
                "probabilities": {"scratch_notes": 0.40, "manuals": 0.35},
            },
            "financial_type": {"choice": "general", "confidence": 0.2, "probabilities": {}},
            "retention": {"score": 0.2, "confidence": 0.5, "probabilities": {}},
            "is_sensitive": {"noul": 0.1, "confidence": 0.5},
        }
    }

    classifier = DocumentClassifier(confidence_threshold=0.60)
    res = classifier.classify("File: untitled.txt\nsome random words 12345")

    assert res["triage_action"]["action"] == "NEEDS_REVIEW"


def test_cuda_unavailable_raises_without_cpu_fallback():
    from ordinale.doc_classifier import CudaDeviceError
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(CudaDeviceError) as exc_info:
            DocumentClassifier(device="cuda")
        assert "torch.cuda.is_available() is False" in str(exc_info.value)
        assert "Exiting without falling back to CPU" in str(exc_info.value)


@patch("laya.load")
def test_cuda_downgrade_to_cpu_raises_immediately(mock_load):
    from ordinale.doc_classifier import CudaDeviceError
    mock_agent = MagicMock()
    # Simulate Laya returning an agent on CPU when CUDA was asked
    mock_agent.device = MagicMock()
    mock_agent.device.type = "cpu"
    mock_load.return_value = mock_agent

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.zeros"):
        with pytest.raises(CudaDeviceError) as exc_info:
            DocumentClassifier(device="cuda")
        assert "fell back" in str(exc_info.value).lower()

