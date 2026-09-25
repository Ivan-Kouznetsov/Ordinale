"""Command-line interface for the Laya Document Organizer.

Provides dry-run visual triage previews, safe automated file routing,
interactive category correction, and instant 1-command rollback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional
import warnings

# Suppress known upstream Laya checkpoint temperature calibration warning
warnings.filterwarnings(
    "ignore",
    message=r".*checkpoint ships invalid temperatures.*",
    category=RuntimeWarning,
)

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ordinale.config import load_settings, normalize_extension
from ordinale.doc_classifier import CudaDeviceError, DocumentClassifier
from ordinale.extractor import DocumentTextExtractor
from ordinale.organizer_engine import OrganizerEngine, OrganizationPlan

console = Console()


def display_benchmark_samples(classifier: DocumentClassifier, samples_file: Path) -> None:
    """Categorizes all synthetic sample documents and renders performance metrics."""
    if not samples_file.exists():
        repo_fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / samples_file.name
        if repo_fixtures.exists():
            samples_file = repo_fixtures
        else:
            console.print(f"[bold red]Samples file '{samples_file}' not found![/bold red]")
            sys.exit(1)

    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    console.rule(f"[bold cyan]Benchmarking {len(samples)} Sample Documents with Laya[/bold cyan]")

    table = Table(
        title="Document Categorization & Government/Financial Routing Benchmark",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("ID / Filename", style="bold", width=22)
    table.add_column("Predicted Category", justify="center", width=22)
    table.add_column("Expected", justify="center", style="dim", width=16)
    table.add_column("Subcategory / Path", style="cyan", width=26)
    table.add_column("Retention", justify="center", width=18)
    table.add_column("Sensitive?", justify="center", width=12)
    table.add_column("Action", style="yellow", width=16)
    table.add_column("Latency", justify="right", style="dim", width=10)

    correct_matches = 0
    total_latency = 0.0

    for item in samples:
        doc_id = item.get("id", "DOC")
        filename = item.get("filename", "unknown")
        expected_cat = item.get("expected_category", "")
        expected_sub = item.get("expected_financial_type") or item.get("expected_education_type")

        prompt = f"File: {filename}\nTitle: {item.get('title', '')}\nContent Snippet:\n{item.get('content', '')}"
        res = classifier.classify(prompt)

        cat_winner = res["category"]["winner"]
        cat_conf = res["category"]["confidence"]
        subcat = res["financial_type"]["winner"] if res.get("financial_type") else (
            res["education_type"]["winner"] if res.get("education_type") else None
        )
        retention = res["retention"]["label"]
        is_sensitive = res["sensitivity"]["is_sensitive"]
        sens_prob = res["sensitivity"]["probability"]
        action = res["triage_action"]["action"]
        subfolder = res["target_subfolder"]
        latency = res["latency_ms"]

        total_latency += latency
        is_match = (cat_winner == expected_cat) if expected_cat else True
        if is_match and expected_sub:
            is_match = (subcat == expected_sub)
        if is_match:
            correct_matches += 1

        cat_style = "bold green" if is_match else "bold red"
        cat_str = f"[{cat_style}]{cat_winner}[/]\n(conf={cat_conf:.2f})"

        sens_str = f"[bold red]YES ({sens_prob:.0%})[/]" if is_sensitive else f"[dim]No ({sens_prob:.0%})[/]"

        id_col = f"{doc_id}\n[dim]{filename[:20]}[/]"
        table.add_row(
            id_col,
            cat_str,
            f"{expected_cat}\n[dim]({expected_sub or '-'})[/]",
            f"{subfolder}",
            retention,
            sens_str,
            action,
            f"{latency:.1f} ms",
        )

    console.print(table)

    avg_latency = total_latency / len(samples) if samples else 0.0
    accuracy = (correct_matches / len(samples)) * 100.0 if samples else 0.0

    console.print()
    console.print(
        Panel(
            f"[bold]Total Documents Tested:[/] {len(samples)}  |  "
            f"[bold]Category Accuracy:[/] [green]{accuracy:.1f}%[/green] ({correct_matches}/{len(samples)})  |  "
            f"[bold]Avg Latency:[/] [cyan]{avg_latency:.1f} ms/doc[/cyan]",
            title="Benchmark Performance",
            border_style="green",
        )
    )


def run_scan_and_organize(
    engine: OrganizerEngine,
    source_dir: Path,
    target_root: Path,
    execute: bool = False,
    interactive: bool = False,
    recursive: bool = True,
    max_workers: Optional[int] = None,
    preserve_folders: bool = True,
    extensions: Optional[List[str]] = None,
) -> None:
    """Scans directory, displays preview, and executes moves if requested."""
    scan_mode = "recursively" if recursive else "top-level only"
    console.rule(f"[bold cyan]Scanning '{source_dir}' for Documents ({scan_mode})[/bold cyan]")

    files = engine.scan_directory(
        source_dir,
        recursive=recursive,
        exclude_dirs=[target_root],
        extensions=extensions,
    )
    if not files:
        exts_list = sorted(list(engine.extractor.supported_extensions.keys()))
        exts_str = ", ".join(exts_list)
        console.print(f"[yellow]No supported document files ({exts_str}) found in '{source_dir}'.[/yellow]")
        return

    workers_desc = f"{max_workers} worker(s)" if max_workers else "auto parallel workers"
    console.print(f"[bold green]Found {len(files)} document(s). Analyzing in-memory with Laya ({workers_desc})...[/bold green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Analyzing documents...", total=len(files))

        def _update_progress(completed: int, total: int, file_path: Path) -> None:
            progress.update(
                task_id,
                completed=completed,
                description=f"[cyan]Analyzing ({file_path.name[:25]})...",
            )

        try:
            plans: List[OrganizationPlan] = engine.plan_organization(
                files=files,
                target_root=target_root,
                source_dir=source_dir,
                max_workers=max_workers,
                preserve_folders=preserve_folders,
                progress_callback=_update_progress,
            )
        except Exception as err:
            if isinstance(err, CudaDeviceError) or "cuda" in str(err).lower():
                console.print(f"\n[bold red]Fatal CUDA Error:[/] {err}")
                console.print("[yellow]Exiting immediately without falling back to CPU because CUDA was specified.[/yellow]")
                sys.exit(1)
            raise

    table = Table(
        title=f"Document Triage Plan (Destination Root: '{target_root}')",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Source Document", style="bold", width=30)
    table.add_column("Predicted Destination", style="cyan", width=36)
    table.add_column("Confidence", justify="center", width=12)
    table.add_column("Retention", justify="center", width=18)
    table.add_column("Sensitive", justify="center", width=11)
    table.add_column("Status / Action", style="yellow", width=18)

    for idx, plan in enumerate(plans, 1):
        rel_target = plan.target_path.relative_to(target_root)
        try:
            rel_source = plan.source_path.relative_to(source_dir)
        except ValueError:
            rel_source = plan.source_path.name

        conf_str = f"{plan.confidence:.0%}"
        conf_style = "green" if plan.confidence >= 0.70 else "yellow" if plan.confidence >= 0.50 else "red"

        sens_badge = "[bold red]YES[/]" if plan.is_sensitive else "[dim]No[/]"

        if plan.is_duplicate:
            status = "[dim strike]DUPLICATE (Skip)[/]"
        elif plan.action == "NEEDS_REVIEW":
            status = "[bold yellow]REVIEW NEEDED[/]"
        elif plan.action == "MOVE_SECURE":
            status = "[bold red]MOVE (SECURE)[/]"
        else:
            status = "[green]MOVE[/]"

        table.add_row(
            str(idx),
            str(rel_source),
            str(rel_target),
            f"[{conf_style}]{conf_str}[/]",
            plan.retention,
            sens_badge,
            status,
        )

    console.print(table)
    console.print()

    # If dry-run mode
    if not execute:
        console.print(
            Panel(
                "[bold yellow]DRY-RUN MODE:[/] No files were moved.\n"
                f"To execute this reorganization, rerun with: [bold green]--execute[/bold green]\n"
                f"Example: [dim]python doc_organizer.py --scan \"{source_dir}\" --target \"{target_root}\" --execute[/dim]",
                title="Safe Preview Complete",
                border_style="yellow",
            )
        )
        return

    # In execute mode
    if interactive:
        for plan in plans:
            if plan.action == "NEEDS_REVIEW":
                try:
                    display_name = plan.source_path.relative_to(source_dir)
                except ValueError:
                    display_name = plan.source_path.name
                console.print(f"\n[bold yellow]Low confidence file:[/] {display_name}")
                console.print(f"Proposed destination: {plan.target_path.relative_to(target_root)}")
                choice = Prompt.ask(
                    "Action",
                    choices=["accept", "skip"],
                    default="accept",
                )
                if choice == "skip":
                    plan.is_duplicate = True  # Marks as skipped

    proceed = Confirm.ask(f"[bold red]Proceed with moving {len(plans)} document(s) to '{target_root}'?[/bold red]")
    if not proceed:
        console.print("[dim]Aborted by user. No files moved.[/dim]")
        return

    res = engine.execute_plans(plans, target_root)
    console.print(
        Panel(
            f"[bold green]Execution Succeeded![/bold green]\n"
            f"- Moved Files: [bold]{res['moved_count']}[/bold]\n"
            f"- Skipped Duplicates: [bold]{res['skipped_duplicates']}[/bold]\n"
            f"- Batch ID: [dim]{res['batch_id']}[/dim]\n\n"
            f"[dim]To revert this batch at any time, run: python doc_organizer.py --target \"{target_root}\" --undo[/dim]",
            title="Batch Complete",
            border_style="green",
        )
    )


def run_undo(engine: OrganizerEngine, target_root: Path) -> None:
    """Restores the last batch of moved documents back to their source directories."""
    console.rule(f"[bold yellow]Rolling Back Last Move Batch in '{target_root}'[/bold yellow]")
    res = engine.undo_last_batch(target_root)
    if res.get("status") == "success":
        console.print(
            Panel(
                f"[bold green]Undo Successful![/bold green]\n"
                f"- Reverted Batch ID: [cyan]{res['batch_id']}[/cyan]\n"
                f"- Restored Files: [bold]{res['restored_count']}[/bold]\n"
                f"- Missing / Unrestored: {res['missing_count']}",
                title="Rollback Complete",
                border_style="green",
            )
        )
    else:
        console.print(f"[bold red]Rollback Failed:[/] {res.get('message')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laya Document Organizer: local, fast, privacy-preserving document triage and filing."
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Run categorization benchmark against sample_documents.json.",
    )
    parser.add_argument(
        "--samples-file",
        type=str,
        default="tests/fixtures/sample_documents.json",
        help="Path to JSON file containing sample documents.",
    )
    parser.add_argument(
        "--scan",
        type=str,
        help="Source directory containing documents to scan (.docx, .pdf, .html, .txt, .md).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target root folder for organized directories (defaults to <source>/Organized_Documents).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform actual file moves (by default runs in safe dry-run mode).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for confirmation on low-confidence files.",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the last batch of moves recorded in the target directory manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu', 'cuda', 'mps'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel worker threads for analysis (defaults to auto: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential (single-threaded) execution.",
    )
    parser.add_argument(
        "--no-preserve-folders",
        action="store_true",
        help="Do not preserve source folder hierarchy under category destinations.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan top-level files in source directory, excluding subdirectories.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force full offline mode (reads directly from local cache without checking HuggingFace Hub).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings file (.toml or .json).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated list of file extensions to scan (e.g. '.pdf,.docx,.txt').",
    )

    args = parser.parse_args()

    # Load settings from config file if found/specified
    try:
        settings = load_settings(config_path=args.config)
    except Exception as err:
        console.print(f"[bold red]Configuration Error:[/] {err}")
        sys.exit(1)

    # CLI flag overrides configured extensions
    if args.extensions:
        parsed_exts = [
            normalize_extension(e)
            for e in args.extensions.split(",")
            if normalize_extension(e)
        ]
        settings.scanner.extensions = parsed_exts

    # Default action if nothing specified
    if not (args.samples or args.scan or args.undo):
        args.samples = True

    # If running undo without loading model
    if args.undo:
        target_dir = Path(args.target) if args.target else Path("Organized_Documents")
        engine = OrganizerEngine(settings=settings)
        run_undo(engine, target_dir)
        return

    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Load classifier
    try:
        classifier = DocumentClassifier(device=args.device)
    except (CudaDeviceError, RuntimeError) as err:
        console.print(f"[bold red]Device Error:[/] {err}")
        if args.device and "cuda" in str(args.device).lower():
            console.print("[yellow]Exiting immediately without falling back to CPU because CUDA was specified.[/yellow]")
        sys.exit(1)

    engine = OrganizerEngine(classifier=classifier, settings=settings)

    try:
        if args.samples:
            display_benchmark_samples(classifier, Path(args.samples_file))
        elif args.scan:
            source_dir = Path(args.scan)
            target_dir = Path(args.target) if args.target else source_dir / "Organized_Documents"
            workers = 1 if args.sequential else args.workers
            run_scan_and_organize(
                engine=engine,
                source_dir=source_dir,
                target_root=target_dir,
                execute=args.execute,
                interactive=args.interactive,
                recursive=not args.no_recursive,
                max_workers=workers,
                preserve_folders=not args.no_preserve_folders,
            )
    except (CudaDeviceError, RuntimeError) as err:
        if (args.device and "cuda" in str(args.device).lower()) or "cuda" in str(err).lower():
            console.print(f"\n[bold red]Fatal CUDA Error:[/] {err}")
            console.print("[yellow]Exiting immediately without falling back to CPU because CUDA was specified.[/yellow]")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
