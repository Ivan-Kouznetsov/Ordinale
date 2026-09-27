"""Unit tests for DocumentClassifier and decision logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ordinale.config import (
    CourseCodeHeuristicsConfig,
    EducationAcademicHeuristicsConfig,
    HeuristicsConfig,
    ScratchNotesHeuristicsConfig,
    WebSnapshotsHeuristicsConfig,
)
from ordinale.doc_classifier import (
    DOCUMENT_QUESTIONS,
    CATEGORY_DESTINATIONS,
    FINANCIAL_SUBDESTINATIONS,
    EDUCATION_SUBDESTINATIONS,
    DocumentClassifier,
    disambiguate_education_academic,
    evaluate_course_codes,
    evaluate_scratch_notes,
    evaluate_web_snapshots,
    is_valid_course_code,
    is_model_cached,
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


def test_course_code_common_vs_uncommon_prefix_it_vs_nw():
    # IT300 is more likely to be a course than NW300
    prompt_it = "Content: IT300 final project requirements."
    prompt_nw = "Content: NW300 final project requirements."

    res_it = disambiguate_education_academic(prompt_it)
    res_nw = disambiguate_education_academic(prompt_nw)

    # IT300 gets common prefix bonus; NW300 gets uncommon prefix penalty
    assert res_it["scores"]["school"] > res_nw["scores"]["school"]
    assert res_it["probabilities"]["school"] > res_nw["probabilities"]["school"]

    eval_it = evaluate_course_codes(prompt_it)
    eval_nw = evaluate_course_codes(prompt_nw)

    assert eval_it["candidates"][0]["is_common"] is True
    assert eval_nw["candidates"][0]["is_common"] is False
    assert eval_it["total_score"] > eval_nw["total_score"]


def test_course_code_proximity_to_name_and_title():
    # If near name or essay title: increase confidence; otherwise (isolated) decrease it
    prompt_near = (
        "File: doc.txt\n"
        "Student Name: John Doe\n"
        "Title: Distributed Computing Systems\n"
        "Course: IT 300\n"
    )
    prompt_isolated = (
        "File: doc.txt\n"
        "A long paragraph discussing random data.\n"
        "There is an isolated code IT 300 listed without context.\n"
        "More arbitrary content follows.\n"
    )

    eval_near = evaluate_course_codes(prompt_near)
    eval_isolated = evaluate_course_codes(prompt_isolated)

    assert eval_near["candidates"][0]["near_name"] is True
    assert eval_near["candidates"][0]["near_title"] is True
    assert eval_near["candidates"][0]["isolated"] is False

    assert eval_isolated["candidates"][0]["near_name"] is False
    assert eval_isolated["candidates"][0]["near_title"] is False
    assert eval_isolated["candidates"][0]["isolated"] is True

    # Higher score and confidence when in proximity to name/title
    assert eval_near["total_score"] > eval_isolated["total_score"]

    res_near = disambiguate_education_academic(prompt_near)
    res_isolated = disambiguate_education_academic(prompt_isolated)
    assert res_near["scores"]["school"] > res_isolated["scores"]["school"]


def test_optional_signals_in_disambiguation():
    prompt = "Student: Alice\nTitle: Operating Systems\nIT 300"

    # 1. Signals disabled (record_signals=False)
    res_no_sig = disambiguate_education_academic(prompt, record_signals=False)
    assert res_no_sig["signals"] == []
    # Evidence is still computed internally
    assert res_no_sig["evidence"]["course_codes"] == ["IT 300"]

    # 2. Concise signals (record_signals=True, detailed_signals=False)
    res_concise = disambiguate_education_academic(prompt, record_signals=True, detailed_signals=False)
    assert len(res_concise["signals"]) > 0
    assert any("Course code: ['IT 300']" in s for s in res_concise["signals"])

    # 3. Detailed signals (record_signals=True, detailed_signals=True)
    res_detailed = disambiguate_education_academic(prompt, record_signals=True, detailed_signals=True)
    assert any("+common prefix" in s and "+near name" in s for s in res_detailed["signals"])


def test_custom_heuristics_override_prefix():
    # When NW is explicitly configured as a common prefix in custom settings
    custom_cfg = EducationAcademicHeuristicsConfig(
        course_codes=CourseCodeHeuristicsConfig(common_prefixes=["nw", "cs"])
    )
    prompt = "Content: NW300 assignment."
    res = disambiguate_education_academic(prompt, heuristics=custom_cfg)
    assert res["evidence"]["course_candidates"][0]["is_common"] is True


def test_student_header_synergy_boosts_school_coursework():
    # Student name + university name + course code on front page
    prompt = (
        "File: final_project.pdf\n"
        "Harvard University\n"
        "Alice Smith\n"
        "CS 181 Final Project\n"
        "December 2024\n"
        "\n"
        "Abstract / Introduction: In this project, we explore deep neural networks..."
    )
    res = disambiguate_education_academic(prompt, file_name="final_project.pdf")

    assert res["winner"] == "school"
    assert res["evidence"]["has_student_synergy"] is True
    assert any("Student header synergy:" in s for s in res["signals"])
    assert any("Harvard" in u for u in res["evidence"]["universities"])
    assert any("Alice Smith" in p for p in res["evidence"]["persons"])


def test_academic_paper_with_university_affiliation():
    # Research paper with university affiliation and academic publisher / DOI / proceedings
    prompt = (
        "File: paper.pdf\n"
        "Proceedings of IEEE Conference on Computer Vision\n"
        "Stanford University\n"
        "John Miller, Sarah Connor\n"
        "doi: 10.1109/CVPR.2024.123456\n"
        "\n"
        "Abstract: We propose a novel transformer architecture..."
    )
    res = disambiguate_education_academic(prompt, file_name="paper.pdf")

    assert res["winner"] == "academic"
    assert res["evidence"]["has_student_synergy"] is False
    assert any("Affiliated institution:" in s for s in res["signals"])


def test_course_code_proximity_via_nlp_person_without_prefix():
    # Adjacent line has raw person name 'Alice Smith' without 'Student:' prefix
    prompt = (
        "Distributed Systems Term Project\n"
        "Alice Smith\n"
        "CS 350\n"
    )
    eval_res = evaluate_course_codes(prompt)
    assert len(eval_res["candidates"]) > 0
    assert eval_res["candidates"][0]["near_name"] is True



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


def is_cuda_supported_on_system() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        return True
    except Exception:
        return False


@pytest.mark.requires_model
@pytest.mark.skipif(not is_cuda_supported_on_system(), reason="Requires a computer supporting CUDA execution")
@pytest.mark.skipif(not is_model_cached(), reason="Requires Laya model to be cached locally without downloading from Hugging Face")
def test_autodetect_and_use_cuda():
    classifier = DocumentClassifier(device=None)
    assert classifier.device == "cuda"
    assert getattr(classifier._agent, "device", None) is not None
    assert getattr(classifier._agent.device, "type", "") == "cuda"


def test_is_model_cached_local_dir(tmp_path: Path):
    from ordinale.doc_classifier import is_model_cached

    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    assert is_model_cached(str(model_dir)) is False

    (model_dir / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    assert is_model_cached(str(model_dir)) is False

    (model_dir / "model.safetensors").write_text("dummy", encoding="utf-8")
    assert is_model_cached(str(model_dir)) is True

    # With subfolder
    sub_dir = model_dir / "sub"
    assert is_model_cached(str(model_dir), subfolder="sub") is False
    sub_dir.mkdir()
    (sub_dir / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (sub_dir / "model.safetensors").write_text("dummy", encoding="utf-8")
    assert is_model_cached(str(model_dir), subfolder="sub") is True


def test_is_model_cached_hub_snapshot(tmp_path: Path):
    from ordinale.doc_classifier import is_model_cached, resolve_cached_model_path

    cfg_file = str(tmp_path / "rl_agent_config.json")
    weights_file = str(tmp_path / "model.safetensors")
    with open(cfg_file, "w") as f:
        f.write("{}")
    with open(weights_file, "w") as f:
        f.write("dummy")

    def mock_cache(repo, filename):
        if filename == "rl_agent_config.json":
            return cfg_file
        if filename == "model.safetensors":
            return weights_file
        return None

    with patch("huggingface_hub.try_to_load_from_cache", side_effect=mock_cache):
        assert is_model_cached("my_org/my_model") is True
        assert resolve_cached_model_path("my_org/my_model") == str(tmp_path)

    with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
        assert is_model_cached("my_org/my_model") is False
        assert resolve_cached_model_path("my_org/my_model") is None


def test_configure_offline_mode():
    import os
    from ordinale.doc_classifier import configure_offline_mode

    # Force offline True
    assert configure_offline_mode(offline=True) is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    # Force online False
    assert configure_offline_mode(offline=False) is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ

    # Auto mode when cached
    with patch("ordinale.doc_classifier.is_model_cached", return_value=True):
        assert configure_offline_mode(offline="auto") is True
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    # Auto mode when not cached
    with patch("ordinale.doc_classifier.is_model_cached", return_value=False):
        assert configure_offline_mode(offline="auto") is False
        assert "HF_HUB_OFFLINE" not in os.environ


@patch("laya.load")
@patch("ordinale.doc_classifier.configure_offline_mode")
def test_document_classifier_passes_offline(mock_configure, mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    DocumentClassifier(offline=True)
    mock_configure.assert_called_with(
        repo_id_or_path="convaiinnovations/laya",
        subfolder=None,
        offline=True,
    )


def test_evaluate_scratch_notes_text_and_doc_format_priors():
    # Plain text format receives text format prior (1.5)
    txt_eval = evaluate_scratch_notes("Just some short thoughts.", file_name="thoughts.txt")
    assert txt_eval["scores"]["format_prior"] == 1.5
    assert txt_eval["scores"]["competitor_penalty"] == 0.0

    # Markdown format receives text format prior (1.5)
    md_eval = evaluate_scratch_notes("Short bullet outline.", file_name="outline.md")
    assert md_eval["scores"]["format_prior"] == 1.5

    # docx and rtf receive doc format prior (1.0)
    docx_eval = evaluate_scratch_notes("Project kickoff rambling.", file_name="kickoff.docx")
    assert docx_eval["scores"]["format_prior"] == 1.0

    rtf_eval = evaluate_scratch_notes("Quick memo text.", file_name="memo_quick.rtf")
    assert rtf_eval["scores"]["format_prior"] == 1.0


def test_evaluate_scratch_notes_meeting_markers():
    content = """File: sync.txt
Content Snippet:
Meeting notes:
Attendees: Alice, Bob, Charlie
Action items:
- finalize architecture doc
- test latency
"""
    res = evaluate_scratch_notes(content, file_name="sync.txt", record_signals=True)
    assert res["is_strong_candidate"] is True
    assert res["scores"]["content_markers"] >= 1.5
    assert any("Content note markers:" in s for s in res["signals"])
    assert any("Bullet list structure:" in s for s in res["signals"])


def test_evaluate_scratch_notes_checklists_and_bullets():
    content = """File: sprint.md
Content Snippet:
- [ ] Implement caching layer
- [x] Review pull request
TODO: write unit tests
"""
    res = evaluate_scratch_notes(content, file_name="sprint.md", record_signals=True)
    assert res["is_strong_candidate"] is True
    assert res["scores"]["checklists"] == 1.5
    assert any("Checklist/task syntax:" in s for s in res["signals"])


def test_evaluate_scratch_notes_filename_cues():
    res_notes = evaluate_scratch_notes("Discussion points for team.", file_name="meeting_notes.docx")
    assert res_notes["scores"]["filename_bonus"] == 2.0
    assert "meeting" in res_notes["evidence"]["filename_matches"] or "notes" in res_notes["evidence"]["filename_matches"]

    res_draft = evaluate_scratch_notes("Early draft of chapter 1.", file_name="draft_v1.rtf")
    assert res_draft["scores"]["filename_bonus"] == 2.0
    assert "draft" in res_draft["evidence"]["filename_matches"]


def test_evaluate_scratch_notes_competitor_exclusion():
    # If a .txt or .docx contains clear tax/CRA/invoice terms, it must NOT be marked as notes
    tax_content = """File: cra_letter.txt
Content Snippet:
CANADA REVENUE AGENCY
Notice of Assessment
Social Insurance Number: 123-456-789
Tax year: 2025
Net income: $75,000
"""
    res_tax = evaluate_scratch_notes(tax_content, file_name="cra_letter.txt")
    assert res_tax["is_strong_candidate"] is False
    assert res_tax["scores"]["format_prior"] == 0.0
    assert res_tax["scores"]["competitor_penalty"] > 0
    assert len(res_tax["evidence"]["competitor_matches"]) > 0

    # Invoice
    inv_content = """File: bill.docx
Content Snippet:
Invoice summary
Invoice number: 994812
Billing period: August 2026
Total due: $145.20
"""
    res_inv = evaluate_scratch_notes(inv_content, file_name="bill.docx")
    assert res_inv["is_strong_candidate"] is False
    assert res_inv["scores"]["format_prior"] == 0.0


def test_evaluate_scratch_notes_disabled():
    cfg = ScratchNotesHeuristicsConfig(enabled=False)
    res = evaluate_scratch_notes("meeting notes: TODO items", file_name="notes.txt", heuristics=cfg)
    assert res["score"] == 0.0
    assert res["is_strong_candidate"] is False


@patch("laya.load")
def test_classify_rescues_notes_low_confidence(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "manuals",
                "confidence": 0.45,
                "probabilities": {"manuals": 0.45, "scratch_notes": 0.40},
            },
            "financial_type": {"choice": "general", "confidence": 0.1, "probabilities": {}},
            "retention": {"score": 0.3, "confidence": 0.7, "probabilities": {}},
            "is_sensitive": {"noul": 0.05, "confidence": 0.8},
        }
    }

    classifier = DocumentClassifier(confidence_threshold=0.60)
    prompt = """File: meeting_minutes.docx
Content Snippet:
Meeting notes:
Attendees: Ivan, Dave
Action items:
- fix bug 404
- deploy to staging
"""
    res = classifier.classify(prompt)
    assert res["category"]["winner"] == "scratch_notes"
    assert res["category"]["confidence"] >= 0.80
    assert res["target_subfolder"] == "Notes & Drafts"
    assert res["triage_action"]["action"] == "AUTO_MOVE"
    assert res["notes_type"] is not None


@patch("laya.load")
def test_classify_corroborates_notes_confidence(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "scratch_notes",
                "confidence": 0.70,
                "probabilities": {"scratch_notes": 0.70, "web_snapshots": 0.20},
            },
            "financial_type": {"choice": "general", "confidence": 0.1, "probabilities": {}},
            "retention": {"score": 0.1, "confidence": 0.8, "probabilities": {}},
            "is_sensitive": {"noul": 0.05, "confidence": 0.8},
        }
    }

    classifier = DocumentClassifier(confidence_threshold=0.60)
    prompt = """File: scratchpad_ideas.txt
Content Snippet:
Quick meeting notes
- sync with Dave tomorrow 10am
- order more coffee
"""
    res = classifier.classify(prompt)
    assert res["category"]["winner"] == "scratch_notes"
    # Corroborated confidence boosted from 0.70
    assert res["category"]["confidence"] > 0.70
    assert res["target_subfolder"] == "Notes & Drafts"
    assert res["triage_action"]["action"] == "AUTO_MOVE"


def test_evaluate_scratch_notes_commands_and_scripts():
    prompt = """File: run categorization script.txt
Content Snippet:
python -m ordinale.doc_organizer --scan C:\\Users\\ivank\\Desktop --target C:\\Users\\ivank\\Desktop\\Organized_Documents
"""
    res = evaluate_scratch_notes(prompt, file_name="run categorization script.txt", record_signals=True)
    assert res["is_strong_candidate"] is True
    assert res["scores"]["commands"] == 2.0
    assert res["scores"]["filename_bonus"] == 2.0
    assert "script" in res["evidence"]["filename_matches"] or "run" in res["evidence"]["filename_matches"]
    assert any("Command/script syntax:" in s for s in res["signals"])


def test_evaluate_web_snapshots_html_and_article_headers():
    html_prompt = """File: article.html
Content Snippet:
<!DOCTYPE html>
<html>
<body>
<article>
URL: https://distributed-systems.org/raft
Published on: 2026-01-15
8 min read
Raft consensus algorithm provides safety under network partitions.
</article>
</body>
</html>
"""
    web_res = evaluate_web_snapshots(html_prompt, file_name="article.html", record_signals=True)
    assert web_res["is_html"] is True
    assert web_res["has_web_headers"] is True
    assert web_res["score"] >= 5.0
    assert any("HTML web format" in s for s in web_res["signals"])
    assert any("Web article headers:" in s for s in web_res["signals"])

    # Non-web plain text penalty
    txt_prompt = "File: notes.txt\nContent Snippet:\njust some thoughts"
    txt_res = evaluate_web_snapshots(txt_prompt, file_name="notes.txt", record_signals=True)
    assert txt_res["is_html"] is False
    assert txt_res["has_web_headers"] is False
    assert txt_res["score"] < 0
    assert any("Plain text non-web penalty" in s for s in txt_res["signals"])


@patch("laya.load")
def test_classify_rescues_web_snapshots_to_scratch_notes(mock_load):
    # Reproduces the real user trigger: 'run categorization script.txt' classified as web_snapshots with 0.1193
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "web_snapshots",
                "confidence": 0.1193,
                "probabilities": {"web_snapshots": 0.1193, "scratch_notes": 0.1105},
            },
            "financial_type": {"choice": "general", "confidence": 0.1, "probabilities": {}},
            "retention": {"score": 0.2, "confidence": 0.5, "probabilities": {}},
            "is_sensitive": {"noul": 0.0, "confidence": 0.9},
        }
    }

    classifier = DocumentClassifier(confidence_threshold=0.55)
    prompt = """File: run categorization script.txt
Content Snippet:
python -m ordinale.doc_organizer --scan C:\\Users\\ivank\\Desktop --target C:\\Users\\ivank\\Desktop\\Organized_Documents
"""
    res = classifier.classify(prompt)
    assert res["category"]["winner"] == "scratch_notes"
    assert res["target_subfolder"] == "Notes & Drafts"
    assert res["category"]["confidence"] >= 0.75
    assert res["triage_action"]["action"] == "AUTO_MOVE"


@patch("laya.load")
def test_classify_preserves_genuine_web_snapshots(mock_load):
    mock_agent = MagicMock()
    mock_load.return_value = mock_agent

    mock_agent.predict.return_value = {
        "answers": {
            "category": {
                "choice": "web_snapshots",
                "confidence": 0.85,
                "probabilities": {"web_snapshots": 0.85, "scratch_notes": 0.05},
            },
            "financial_type": {"choice": "general", "confidence": 0.1, "probabilities": {}},
            "retention": {"score": 0.5, "confidence": 0.8, "probabilities": {}},
            "is_sensitive": {"noul": 0.0, "confidence": 0.9},
        }
    }

    classifier = DocumentClassifier(confidence_threshold=0.55)
    prompt = """File: raft_consensus.html
Content Snippet:
<!DOCTYPE html>
<html>
<body>
<article>
URL: https://example.com/raft
Published on: June 2026
In Search of an Understandable Consensus Algorithm.
</article>
</body>
</html>
"""
    res = classifier.classify(prompt)
    assert res["category"]["winner"] == "web_snapshots"
    assert res["target_subfolder"] == "Web Articles & Clippings"
    assert res["triage_action"]["action"] == "AUTO_MOVE"
