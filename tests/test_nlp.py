"""Unit tests for ordinale.nlp module and spaCy entity extraction."""

from unittest.mock import patch
import pytest

from ordinale.nlp import (
    DocumentEntityExtractor,
    DocumentStructuralSegmenter,
    FilenamePreprocessor,
    SpacyManager,
)


def test_clean_filename_snake_and_kebab():
    raw = "CS101_John_Doe_Final-Essay.docx"
    clean = FilenamePreprocessor.clean_filename(raw)
    assert clean == "CS 101 John Doe Final Essay"

    raw2 = "math240-homework-3-smith.pdf"
    clean2 = FilenamePreprocessor.clean_filename(raw2)
    assert clean2 == "math 240 homework 3 smith"


def test_clean_filename_camelcase():
    raw = "DistributedSystemsTermPaper.pdf"
    clean = FilenamePreprocessor.clean_filename(raw)
    assert clean == "Distributed Systems Term Paper"


def test_extract_filename_entities():
    filename = "CS101_John_Doe_Final_Project.docx"
    nlp = SpacyManager.get_nlp()
    res = FilenamePreprocessor.extract_filename_entities(filename, nlp=nlp)

    assert "project" in res["coursework_terms"]
    assert any("CS 101" in code or "CS101" in code for code in res["course_codes"])
    assert res["cleaned_text"] == "CS 101 John Doe Final Project"


def test_structural_segmenter():
    sample_text = (
        "Distributed Systems and Cloud Computing\n"
        "Student: Alice Smith\n"
        "Course: CS 450\n"
        "University of Washington\n"
        "\n"
        "Abstract\n"
        "This project explores raft consensus.\n"
        "\n"
        "References\n"
        "1. Lamport, L. (1998). The Part-Time Parliament."
    )

    segmented = DocumentStructuralSegmenter.segment_text(sample_text, max_header_lines=5)
    assert "Student: Alice Smith" in segmented["front_page"]
    assert "University of Washington" in segmented["front_page"]
    assert "References" not in segmented["front_page"]
    assert "Lamport" in segmented["references"]


def test_document_entity_extractor():
    front_page = (
        "Computer Science Department\n"
        "Harvard University\n"
        "Title: Deep Learning for Vision\n"
        "Student Name: Bob Jones\n"
        "Instructor: Prof. Sarah Jenkins\n"
        "Due Date: October 15, 2024\n"
        "Course: CS 181\n"
    )

    extractor = DocumentEntityExtractor()
    entities = extractor.extract_entities(front_page)

    assert any("Bob Jones" in p or "Sarah Jenkins" in p for p in entities["persons"])
    assert any("Harvard" in u for u in entities["universities"])


def test_analyze_front_page_synergy():
    front_page = (
        "Operating Systems Principles\n"
        "Author: David Miller\n"
        "University of California, Berkeley\n"
        "Course: CS 162\n"
    )
    extractor = DocumentEntityExtractor()
    analysis = extractor.analyze_front_page(front_page, filename="Miller_CS162_Lab1.pdf")

    assert analysis["has_person"] is True
    assert analysis["has_university"] is True
    assert any("David Miller" in p or "Miller" in p for p in analysis["persons"])
    assert any("California" in u or "University" in u for u in analysis["universities"])
    assert "lab" in analysis["filename_data"]["coursework_terms"]


def test_spacy_manager_fallback_on_missing_model():
    with patch("spacy.load", side_effect=OSError("Model not found")):
        nlp = SpacyManager.get_nlp(model_name="non_existent_model_12345")
        # Should fall back to blank:en gracefully
        assert nlp is not None
