"""Unit tests for doc_organizer.py CLI."""

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from ordinale.doc_organizer import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["doc_organizer.py", "--help"]):
            main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--scan" in captured.out
    assert "--target" in captured.out
    assert "--execute" in captured.out
    assert "--undo" in captured.out
    assert "--samples" in captured.out
    assert "--no-recursive" in captured.out
    assert "--workers" in captured.out
    assert "--sequential" in captured.out
    assert "--no-preserve-folders" in captured.out
    assert "--config" in captured.out
    assert "--offline" in captured.out
    assert "--online" in captured.out
    assert "--extensions" not in captured.out


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.display_benchmark_samples")
def test_cli_default_samples(mock_display, mock_classifier):
    with patch("sys.argv", ["doc_organizer.py"]):
        main()
    assert mock_display.called


@patch("ordinale.doc_organizer.OrganizerEngine")
@patch("ordinale.doc_organizer.run_undo")
def test_cli_undo(mock_undo, mock_engine):
    with patch("sys.argv", ["doc_organizer.py", "--undo", "--target", "MyOrganized"]):
        main()
    assert mock_undo.called


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.run_scan_and_organize")
def test_cli_scan_args(mock_run_scan, mock_classifier):
    with patch("sys.argv", ["doc_organizer.py", "--scan", "my_folder", "--no-recursive", "--workers", "6"]):
        main()
    assert mock_run_scan.called
    _, kwargs = mock_run_scan.call_args
    assert kwargs["recursive"] is False
    assert kwargs["max_workers"] == 6
    assert kwargs["preserve_folders"] is True


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.run_scan_and_organize")
def test_cli_sequential_and_no_preserve_folders(mock_run_scan, mock_classifier):
    with patch("sys.argv", ["doc_organizer.py", "--scan", "my_folder", "--sequential", "--no-preserve-folders"]):
        main()
    assert mock_run_scan.called
    _, kwargs = mock_run_scan.call_args
    assert kwargs["max_workers"] == 1
    assert kwargs["preserve_folders"] is False


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.run_scan_and_organize")
def test_cli_config(mock_run_scan, mock_classifier, tmp_path: Path):
    cfg_file = tmp_path / "ordinale.json"
    cfg_file.write_text('{"scanner": {"extensions": [".pdf", ".docx"]}}', encoding="utf-8")

    with patch("sys.argv", [
        "doc_organizer.py",
        "--scan", "my_folder",
        "--config", str(cfg_file),
    ]):
        main()

    assert mock_run_scan.called
    engine_arg = mock_run_scan.call_args[1]["engine"]
    assert engine_arg.settings.scanner.extensions == [".pdf", ".docx"]


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.display_benchmark_samples")
def test_cli_offline_flag(mock_display, mock_classifier):
    with patch("sys.argv", ["doc_organizer.py", "--offline"]):
        main()
    assert mock_classifier.called
    _, kwargs = mock_classifier.call_args
    assert kwargs.get("offline") is True


@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.display_benchmark_samples")
def test_cli_online_flag(mock_display, mock_classifier):
    with patch("sys.argv", ["doc_organizer.py", "--online"]):
        main()
    assert mock_classifier.called
    _, kwargs = mock_classifier.call_args
    assert kwargs.get("offline") is False


def test_cli_conflicting_offline_online_flags():
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["doc_organizer.py", "--offline", "--online"]):
            main()
    assert exc_info.value.code == 1


@patch("ordinale.doc_organizer.is_model_cached", return_value=True)
@patch("ordinale.doc_organizer.DocumentClassifier")
@patch("ordinale.doc_organizer.display_benchmark_samples")
def test_cli_default_auto_offline(mock_display, mock_classifier, mock_is_cached):
    with patch("sys.argv", ["doc_organizer.py"]):
        main()
    assert mock_classifier.called
    _, kwargs = mock_classifier.call_args
    assert kwargs.get("offline") == "auto"
