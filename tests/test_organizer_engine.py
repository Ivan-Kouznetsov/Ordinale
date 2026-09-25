"""Unit tests for OrganizerEngine collision resolution, planning, execution, and undo."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from ordinale.organizer_engine import OrganizerEngine, OrganizationPlan


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify.return_value = {
        "category": {"winner": "financial", "confidence": 0.94},
        "financial_type": {"winner": "taxes_government", "confidence": 0.91},
        "retention": {"label": "Permanent Archive"},
        "sensitivity": {"is_sensitive": True},
        "target_subfolder": "Financial/Taxes & Government",
        "triage_action": {"action": "MOVE_SECURE", "reason": "High confidence tax document"},
    }
    return classifier


def test_scan_directory(tmp_path: Path):
    engine = OrganizerEngine()
    (tmp_path / "doc1.docx").write_text("dummy", encoding="utf-8")
    (tmp_path / "doc2.pdf").write_text("dummy", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("dummy", encoding="utf-8")
    (tmp_path / ".hidden.pdf").write_text("dummy", encoding="utf-8")
    (tmp_path / "app.exe").write_text("dummy", encoding="utf-8")

    files = engine.scan_directory(tmp_path)
    names = [f.name for f in files]

    assert "doc1.docx" in names
    assert "doc2.pdf" in names
    assert "notes.txt" in names
    assert ".hidden.pdf" not in names
    assert "app.exe" not in names


def test_scan_directory_recursive_and_excludes(tmp_path: Path):
    engine = OrganizerEngine()

    # Top-level doc
    (tmp_path / "root_doc.txt").write_text("root", encoding="utf-8")

    # Subfolder doc
    sub_dir = tmp_path / "nested" / "deep"
    sub_dir.mkdir(parents=True)
    (sub_dir / "nested_doc.pdf").write_text("nested", encoding="utf-8")

    # Hidden folder doc (should be ignored)
    hidden_dir = tmp_path / ".git" / "hooks"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "git_doc.txt").write_text("git", encoding="utf-8")

    # Excluded folder doc (e.g. target root)
    excluded_dir = tmp_path / "Organized_Documents"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "already_organized.docx").write_text("organized", encoding="utf-8")

    # Test recursive=True (default) with exclude_dirs
    files = engine.scan_directory(tmp_path, recursive=True, exclude_dirs=[excluded_dir])
    names = [f.name for f in files]

    assert "root_doc.txt" in names
    assert "nested_doc.pdf" in names
    assert "git_doc.txt" not in names
    assert "already_organized.docx" not in names

    # Test recursive=False
    top_only_files = engine.scan_directory(tmp_path, recursive=False)
    top_names = [f.name for f in top_only_files]
    assert "root_doc.txt" in top_names
    assert "nested_doc.pdf" not in top_names


def test_collision_duplicate(tmp_path: Path):
    engine = OrganizerEngine()
    src_file = tmp_path / "source" / "doc.txt"
    src_file.parent.mkdir()
    src_file.write_text("Identical content", encoding="utf-8")

    target_dir = tmp_path / "dest"
    target_dir.mkdir()
    dest_file = target_dir / "doc.txt"
    dest_file.write_text("Identical content", encoding="utf-8")

    resolved_path, is_dup = engine.resolve_target_collision(src_file, target_dir)
    assert is_dup is True
    assert resolved_path == dest_file


def test_collision_disambiguate(tmp_path: Path):
    engine = OrganizerEngine()
    src_file = tmp_path / "source" / "doc.txt"
    src_file.parent.mkdir()
    src_file.write_text("New content", encoding="utf-8")

    target_dir = tmp_path / "dest"
    target_dir.mkdir()
    dest_file = target_dir / "doc.txt"
    dest_file.write_text("Old different content", encoding="utf-8")

    resolved_path, is_dup = engine.resolve_target_collision(src_file, target_dir)
    assert is_dup is False
    assert resolved_path.name == "doc (1).txt"


def test_execute_and_undo_lifecycle(tmp_path: Path, mock_classifier):
    engine = OrganizerEngine(classifier=mock_classifier)
    source_dir = tmp_path / "downloads"
    source_dir.mkdir()
    target_root = tmp_path / "organized"

    file1 = source_dir / "CRA_Assessment.pdf"
    file1.write_text("CRA Notice 2025 Tax Content", encoding="utf-8")

    # 1. Plan
    files = [file1]
    plans = engine.plan_organization(files, target_root, classifier=mock_classifier)
    assert len(plans) == 1
    assert plans[0].category == "financial"
    assert plans[0].subcategory == "taxes_government"
    assert "Financial" in str(plans[0].target_path)
    assert "Taxes & Government" in str(plans[0].target_path)

    # 2. Execute
    exec_res = engine.execute_plans(plans, target_root)
    assert exec_res["moved_count"] == 1
    assert not file1.exists()

    expected_dest = target_root / "Financial" / "Taxes & Government" / "CRA_Assessment.pdf"
    assert expected_dest.exists()

    # 3. Undo
    undo_res = engine.undo_last_batch(target_root)
    assert undo_res["status"] == "success"
    assert undo_res["restored_count"] == 1
    assert file1.exists()
    assert not expected_dest.exists()


def test_relative_subfolder_preservation(tmp_path: Path, mock_classifier):
    """Verifies that files from different subfolders preserve their relative folders under the target category."""
    engine = OrganizerEngine(classifier=mock_classifier)
    source_dir = tmp_path / "source"
    target_root = tmp_path / "organized"

    inbox_dir = source_dir / "inbox"
    inbox_dir.mkdir(parents=True)
    file_inbox = inbox_dir / "receipt.pdf"
    file_inbox.write_text("Invoice 1", encoding="utf-8")

    downloads_dir = source_dir / "downloads"
    downloads_dir.mkdir(parents=True)
    file_downloads = downloads_dir / "receipt.pdf"
    file_downloads.write_text("Invoice 2", encoding="utf-8")

    files = [file_inbox, file_downloads]
    plans = engine.plan_organization(
        files,
        target_root,
        source_dir=source_dir,
        classifier=mock_classifier,
        preserve_folders=True,
    )

    assert len(plans) == 2
    # Verify neither file collided or was renamed to receipt (1).pdf
    assert plans[0].target_path.name == "receipt.pdf"
    assert plans[1].target_path.name == "receipt.pdf"

    # Verify both preserved their respective source folders
    assert plans[0].target_path == target_root / "Financial" / "Taxes & Government" / "inbox" / "receipt.pdf"
    assert plans[1].target_path == target_root / "Financial" / "Taxes & Government" / "downloads" / "receipt.pdf"


def test_plan_organization_parallel_workers(tmp_path: Path, mock_classifier):
    """Verifies parallel execution with multiple worker threads returns identical results and preserves order."""
    engine = OrganizerEngine(classifier=mock_classifier)
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    target_root = tmp_path / "organized"

    created_files = []
    for i in range(10):
        f = source_dir / f"doc_{i:02d}.txt"
        f.write_text(f"Content {i}", encoding="utf-8")
        created_files.append(f)

    progress_events = []

    def on_progress(done, total, p):
        progress_events.append((done, total, p.name))

    plans = engine.plan_organization(
        created_files,
        target_root,
        classifier=mock_classifier,
        max_workers=4,
        progress_callback=on_progress,
    )

    assert len(plans) == 10
    # Original order must be preserved
    for i, plan in enumerate(plans):
        assert plan.source_path == created_files[i]
        assert plan.target_path.name == f"doc_{i:02d}.txt"

    # Progress callback was invoked for each file
    assert len(progress_events) == 10
    assert progress_events[-1][0] == 10  # done == total


def test_batch_claimed_targets_same_folder_collision(tmp_path: Path, mock_classifier):
    """Verifies that in-batch identical names without folder preservation are disambiguated safely."""
    engine = OrganizerEngine(classifier=mock_classifier)
    source_dir = tmp_path / "source"
    target_root = tmp_path / "organized"

    dir1 = source_dir / "folder_a"
    dir1.mkdir(parents=True)
    f1 = dir1 / "statement.pdf"
    f1.write_text("Unique content 1", encoding="utf-8")

    dir2 = source_dir / "folder_b"
    dir2.mkdir(parents=True)
    f2 = dir2 / "statement.pdf"
    f2.write_text("Unique content 2", encoding="utf-8")

    dir3 = source_dir / "folder_c"
    dir3.mkdir(parents=True)
    f3 = dir3 / "statement.pdf"
    f3.write_text("Unique content 1", encoding="utf-8")  # Exact duplicate of f1

    # Flatten into category without preserving folders
    plans = engine.plan_organization(
        [f1, f2, f3],
        target_root,
        classifier=mock_classifier,
        preserve_folders=False,
    )

    assert len(plans) == 3
    # f1 gets statement.pdf
    assert plans[0].target_path.name == "statement.pdf"
    assert plans[0].is_duplicate is False

    # f2 has different content -> gets statement (1).pdf
    assert plans[1].target_path.name == "statement (1).pdf"
    assert plans[1].is_duplicate is False

    # f3 has identical content to f1 -> marked as duplicate of statement.pdf
    assert plans[2].target_path.name == "statement.pdf"
    assert plans[2].is_duplicate is True

