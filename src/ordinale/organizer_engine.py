"""Safe file movement and manifest rollback engine for Document Organizer.

Manages collision resolution, SHA-256 deduplication, atomic directory creation,
and persistent manifest tracking to allow instantaneous single-command undo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ordinale.config import Settings, normalize_extension
from ordinale.doc_classifier import CudaDeviceError, DocumentClassifier
from ordinale.extractor import DocumentTextExtractor, ExtractedDocument


@dataclass
class OrganizationPlan:
    """Planned move operation for a single document."""

    source_path: Path
    target_path: Path
    file_sha256: str
    category: str
    subcategory: Optional[str]
    confidence: float
    retention: str
    is_sensitive: bool
    action: str
    reason: str
    is_duplicate: bool = False


class OrganizerEngine:
    """Orchestrates document scanning, categorization, safe movement, and rollback."""

    MANIFEST_FILENAME = ".organizer_manifest.json"

    def __init__(
        self,
        extractor: Optional[DocumentTextExtractor] = None,
        classifier: Optional[DocumentClassifier] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings
        if extractor is not None:
            self.extractor = extractor
        elif settings is not None:
            self.extractor = DocumentTextExtractor(
                supported_extensions=settings.scanner.get_effective_supported_extensions()
            )
        else:
            self.extractor = DocumentTextExtractor()
        self.classifier = classifier

    def scan_directory(
        self,
        source_dir: Path | str,
        recursive: bool = True,
        exclude_dirs: Optional[List[Path | str]] = None,
        extensions: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """Discovers all supported document files in the given directory.

        Args:
            source_dir: Directory containing documents to scan.
            recursive: If True, recursively traverses all subdirectories.
            exclude_dirs: Optional list of directories to exclude (e.g. target output root).
            extensions: Optional override list of extensions to look for.

        Returns:
            Sorted list of discovered document Paths.
        """
        src = Path(source_dir).resolve()
        if not src.exists() or not src.is_dir():
            return []

        resolved_excludes = [Path(d).resolve() for d in (exclude_dirs or [])]
        allowed_exts: Optional[Set[str]] = None
        if extensions is not None:
            allowed_exts = {normalize_extension(e) for e in extensions if normalize_extension(e)}

        def _is_match(path: Path) -> bool:
            if allowed_exts is not None:
                return path.suffix.lower() in allowed_exts and self.extractor.is_supported(path)
            return self.extractor.is_supported(path)

        def _is_excluded(path: Path) -> bool:
            p_res = path.resolve()
            for exc in resolved_excludes:
                if p_res == exc or p_res.is_relative_to(exc):
                    return True
            return False

        found_files: List[Path] = []

        if recursive:
            for root_str, dirnames, filenames in os.walk(src):
                root_path = Path(root_str)

                # Skip root itself if excluded
                if _is_excluded(root_path):
                    dirnames.clear()
                    continue

                # Filter out hidden directories and excluded directories in-place
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not d.startswith(".") and not _is_excluded(root_path / d)
                ]

                for fname in filenames:
                    if fname.startswith("."):
                        continue
                    file_path = root_path / fname
                    if _is_match(file_path):
                        found_files.append(file_path)
        else:
            for entry in src.iterdir():
                if entry.is_file() and not entry.name.startswith("."):
                    if _is_match(entry):
                        found_files.append(entry)

        return sorted(found_files, key=lambda p: str(p).lower())

    @staticmethod
    def calculate_sha256(path: Path) -> str:
        """Computes SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def resolve_target_collision(
        self,
        source_file: Path,
        target_dir: Path,
        claimed_targets: Optional[Dict[Path, str]] = None,
        source_hash: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """
        Determines target file path. If file exists with identical hash, flags as duplicate.
        Otherwise appends a numeric suffix (e.g. 'doc (1).docx') to prevent overwriting.

        Checks both existing filesystem files and an in-memory `claimed_targets` dictionary
        (target_path -> sha256) to ensure collision safety during concurrent or batch planning.
        """
        if source_hash is None:
            source_hash = self.calculate_sha256(source_file)

        candidate = target_dir / source_file.name

        def _is_match_or_taken(path: Path) -> Tuple[bool, bool]:
            """Returns (is_exact_dup, is_taken)."""
            # Check in-memory claimed targets first
            if claimed_targets is not None and path in claimed_targets:
                if claimed_targets[path] == source_hash:
                    return True, True
                return False, True

            # Check filesystem
            if path.exists():
                try:
                    if self.calculate_sha256(path) == source_hash:
                        return True, True
                except (OSError, IOError):
                    pass
                return False, True

            return False, False

        is_dup, is_taken = _is_match_or_taken(candidate)
        if not is_taken:
            if claimed_targets is not None:
                claimed_targets[candidate] = source_hash
            return candidate, False
        if is_dup:
            return candidate, True

        # Disambiguate filename
        stem = source_file.stem
        suffix = source_file.suffix
        counter = 1
        while True:
            candidate = target_dir / f"{stem} ({counter}){suffix}"
            is_dup, is_taken = _is_match_or_taken(candidate)
            if not is_taken:
                if claimed_targets is not None:
                    claimed_targets[candidate] = source_hash
                return candidate, False
            if is_dup:
                return candidate, True
            counter += 1

    def plan_organization(
        self,
        files: List[Path],
        target_root: Path | str,
        source_dir: Optional[Path | str] = None,
        classifier: Optional[DocumentClassifier] = None,
        max_workers: Optional[int] = None,
        preserve_folders: bool = True,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    ) -> List[OrganizationPlan]:
        """Extracts content, classifies, and plans safe destination paths for each file.

        Args:
            files: List of document Paths to analyze.
            target_root: Destination root directory.
            source_dir: Optional source directory root to preserve relative subfolders.
            classifier: DocumentClassifier instance (defaults to self.classifier).
            max_workers: Concurrency limit for ThreadPoolExecutor. Defaults to min(8, cpu_count).
            preserve_folders: If True and source_dir is provided, preserves subfolder paths
                relative to source_dir under the category folder.
            progress_callback: Optional callback fn(completed_count, total_count, file_path)
                invoked as files finish analysis.

        Returns:
            List of OrganizationPlan objects in the same order as input files.
        """
        active_classifier = classifier or self.classifier
        if active_classifier is None:
            raise ValueError("DocumentClassifier must be provided to plan organization.")

        root = Path(target_root)
        if not files:
            return []

        def _analyze_single(
            file_path: Path,
        ) -> Tuple[Path, str, ExtractedDocument, Dict[str, Any], Optional[Exception]]:
            try:
                doc = self.extractor.extract(file_path)
                file_hash = self.calculate_sha256(file_path)
                res = active_classifier.classify(doc.prompt_text)
                return (file_path, file_hash, doc, res, None)
            except Exception as exc:
                exc_str = str(exc).lower()
                is_cuda_err = (
                    isinstance(exc, CudaDeviceError)
                    or "cuda" in exc_str
                    or "device fell back" in exc_str
                )
                if is_cuda_err and getattr(active_classifier, "device", None) and "cuda" in str(active_classifier.device).lower():
                    # If CUDA was explicitly requested and failed, propagate immediately - do not swallow or fall back
                    raise

                fallback_doc = ExtractedDocument(
                    file_path=file_path,
                    file_name=file_path.name,
                    file_type="unknown",
                    text_snippet="",
                    extraction_error=str(exc),
                )
                return (file_path, "", fallback_doc, {}, exc)

        total_files = len(files)
        if max_workers is None:
            max_workers = min(8, max(1, os.cpu_count() or 1))
        effective_workers = max(1, min(max_workers, total_files))

        analyzed_results: List[
            Optional[Tuple[Path, str, ExtractedDocument, Dict[str, Any], Optional[Exception]]]
        ] = [None] * total_files
        completed_count = 0

        if effective_workers > 1:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_idx = {
                    executor.submit(_analyze_single, f): idx for idx, f in enumerate(files)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    res = future.result()
                    analyzed_results[idx] = res
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_files, res[0])
        else:
            for idx, file_path in enumerate(files):
                res = _analyze_single(file_path)
                analyzed_results[idx] = res
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_files, file_path)

        # Sequential destination path allocation & collision resolution
        plans: List[OrganizationPlan] = []
        claimed_targets: Dict[Path, str] = {}
        resolved_source_dir = Path(source_dir).resolve() if source_dir else None

        for item in analyzed_results:
            assert item is not None
            file_path, file_hash, doc, res, exc = item

            if exc is not None or not res:
                dest_dir = root / "Organized/Failed_Extraction"
                resolved_target, is_dup = self.resolve_target_collision(
                    file_path, dest_dir, claimed_targets=claimed_targets, source_hash=file_hash or "0"
                )
                plans.append(
                    OrganizationPlan(
                        source_path=file_path,
                        target_path=resolved_target,
                        file_sha256=file_hash or "",
                        category="unknown",
                        subcategory=None,
                        confidence=0.0,
                        retention="Needs Review",
                        is_sensitive=False,
                        action="NEEDS_REVIEW",
                        reason=f"Analysis failed: {exc}",
                        is_duplicate=is_dup,
                    )
                )
                continue

            cat_winner = res["category"]["winner"]
            cat_conf = res["category"]["confidence"]
            subcat = None
            if res.get("financial_type"):
                subcat = res["financial_type"]["winner"]
            elif res.get("education_type"):
                subcat = res["education_type"]["winner"]
            retention_label = res["retention"]["label"]
            is_sensitive = res["sensitivity"]["is_sensitive"]
            subfolder = res["target_subfolder"]
            triage_action = res["triage_action"]

            dest_dir = root / subfolder

            # If preserve_folders is True and source_dir provided, keep relative subdirectories
            if preserve_folders and resolved_source_dir:
                try:
                    rel_to_src = file_path.resolve().relative_to(resolved_source_dir)
                    rel_parent = rel_to_src.parent
                    if rel_parent != Path("."):
                        dest_dir = dest_dir / rel_parent
                except ValueError:
                    pass

            resolved_target, is_dup = self.resolve_target_collision(
                file_path, dest_dir, claimed_targets=claimed_targets, source_hash=file_hash
            )

            plan = OrganizationPlan(
                source_path=file_path,
                target_path=resolved_target,
                file_sha256=file_hash,
                category=cat_winner,
                subcategory=subcat,
                confidence=cat_conf,
                retention=retention_label,
                is_sensitive=is_sensitive,
                action=triage_action["action"],
                reason=triage_action["reason"],
                is_duplicate=is_dup,
            )
            plans.append(plan)

        return plans

    def execute_plans(
        self,
        plans: List[OrganizationPlan],
        target_root: Path | str,
    ) -> Dict[str, Any]:
        """
        Executes moves, creates target directories, and logs to manifest ledger.
        Skips exact duplicates to prevent redundant moves.
        """
        root = Path(target_root)
        batch_id = time.strftime("%Y%m%d-%H%M%S")
        moved_actions = []

        for plan in plans:
            if plan.is_duplicate:
                continue

            plan.target_path.parent.mkdir(parents=True, exist_ok=True)
            # Move file
            shutil.move(str(plan.source_path), str(plan.target_path))

            moved_actions.append(
                {
                    "source_path": str(plan.source_path.resolve()),
                    "target_path": str(plan.target_path.resolve()),
                    "file_sha256": plan.file_sha256,
                    "category": plan.category,
                    "subcategory": plan.subcategory,
                    "confidence": plan.confidence,
                    "retention": plan.retention,
                    "is_sensitive": plan.is_sensitive,
                }
            )

        # Update manifest
        manifest_path = root / self.MANIFEST_FILENAME
        manifest_data = self._read_manifest(manifest_path)
        manifest_data.append(
            {
                "batch_id": batch_id,
                "timestamp": time.time(),
                "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_moved": len(moved_actions),
                "actions": moved_actions,
            }
        )
        self._write_manifest(manifest_path, manifest_data)

        return {
            "batch_id": batch_id,
            "moved_count": len(moved_actions),
            "skipped_duplicates": sum(1 for p in plans if p.is_duplicate),
        }

    def undo_last_batch(self, target_root: Path | str) -> Dict[str, Any]:
        """Rolls back the most recent move batch recorded in target_root manifest."""
        root = Path(target_root)
        manifest_path = root / self.MANIFEST_FILENAME
        manifest_data = self._read_manifest(manifest_path)

        if not manifest_data:
            return {"status": "error", "message": "No move history found in manifest."}

        last_batch = manifest_data.pop()
        batch_id = last_batch["batch_id"]
        actions = last_batch.get("actions", [])

        restored_count = 0
        missing_count = 0

        for action in reversed(actions):
            current_target = Path(action["target_path"])
            orig_source = Path(action["source_path"])

            if current_target.exists():
                orig_source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_target), str(orig_source))
                restored_count += 1
            else:
                missing_count += 1

        # Write back updated manifest
        self._write_manifest(manifest_path, manifest_data)

        return {
            "status": "success",
            "batch_id": batch_id,
            "restored_count": restored_count,
            "missing_count": missing_count,
        }

    def _read_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        if not manifest_path.exists():
            return []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_manifest(self, manifest_path: Path, data: List[Dict[str, Any]]) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
