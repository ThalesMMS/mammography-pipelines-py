#!/usr/bin/env python3
"""Check that no protected medical data is tracked inside the repository."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
)


def check_git_status() -> bool:
    """Return ``True`` if the current directory is a Git repository."""
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:  # git not installed
        return False


def check_gitignore() -> bool:
    """Ensure the ``.gitignore`` file excludes medical data directories and files."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        return False

    content = gitignore_path.read_text(encoding="utf-8")
    required_patterns = [
        "archive/",
        "*.dcm",
        "*.dicom",
        "data/raw/",
        "patient_data/",
        "medical_records/",
    ]
    return any(pattern in content for pattern in required_patterns)


def check_tracked_files() -> List[str]:
    """Return the list of files tracked by Git."""
    try:
        result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def check_medical_file_extensions(tracked_files: List[str]) -> List[str]:
    """Identify tracked files with medical imaging extensions."""
    medical_extensions = (".dcm", ".dicom", ".DCM", ".DICOM")
    return [path for path in tracked_files if path.endswith(medical_extensions)]


def check_medical_directories(tracked_files: List[str]) -> List[str]:
    """Detect tracked files located under sensitive directories."""
    medical_dirs = ("archive/", "data/raw/", "patient_data/", "medical_records/")
    return [path for path in tracked_files if path.startswith(medical_dirs)]


def check_file_sizes(tracked_files: List[str]) -> List[Tuple[str, float]]:
    """Return tracked files larger than 10 MB (they may contain medical data)."""
    large_files: List[Tuple[str, float]] = []
    for file_path in tracked_files:
        if not os.path.exists(file_path):
            continue
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 10:
            large_files.append((file_path, size_mb))
    return large_files


def check_archive_directory() -> Tuple[bool, List[str]]:
    """Verify whether ``archive/`` exists and list DICOM files inside it."""
    archive_dir = Path("archive")
    if not archive_dir.exists():
        return False, []

    dicom_files: List[Path] = []
    for extension in (".dcm", ".dicom", ".DCM", ".DICOM"):
        dicom_files.extend(archive_dir.rglob(f"*{extension}"))

    return True, [str(path) for path in dicom_files]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan the repository for tracked medical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RESEARCH_DISCLAIMER,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero code when medical files are detected",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about tracked files",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Medical Data Safety Check")
    print("=" * 80)
    print(RESEARCH_DISCLAIMER.strip())
    print("=" * 80)

    is_git_repo = check_git_status()
    has_gitignore = check_gitignore()

    print("
🔍 BASIC CHECKS")
    print(f"Git repository detected: {'✅' if is_git_repo else '❌'}")
    print(f".gitignore protects medical data: {'✅' if has_gitignore else '❌'}")

    if not is_git_repo:
        print("
⚠️ Not a Git repository — tracked-file checks skipped.")
        return 0

    tracked_files = check_tracked_files()
    print(f"Tracked files: {len(tracked_files)}")

    medical_files = check_medical_file_extensions(tracked_files)
    medical_dirs = check_medical_directories(tracked_files)
    large_files = check_file_sizes(tracked_files)
    archive_exists, archive_files = check_archive_directory()

    print("
🛡️ MEDICAL DATA SAFETY")
    print(f"DICOM files tracked: {len(medical_files)}")
    print(f"Sensitive directories tracked: {len(medical_dirs)}")
    print(f"Files larger than 10 MB: {len(large_files)}")
    print(f"Local archive directory present: {'✅' if archive_exists else '❌'}")

    if args.verbose:
        if medical_files:
            print("
Tracked DICOM files:")
            for file_path in medical_files:
                print(f"  - {file_path}")
        if medical_dirs:
            print("
Tracked files inside sensitive directories:")
            for file_path in medical_dirs:
                print(f"  - {file_path}")
        if large_files:
            print("
Large tracked files (>10 MB):")
            for file_path, size_mb in large_files:
                print(f"  - {file_path} ({size_mb:.2f} MB)")
        if archive_files:
            print("
Local archive DICOM files (not tracked):")
            for file_path in archive_files[:10]:
                print(f"  - {file_path}")
            if len(archive_files) > 10:
                print(f"  ... {len(archive_files) - 10} more")

    has_issues = bool(medical_files or medical_dirs)

    if has_issues:
        print("
❌ Attention: remove medical data from version control immediately.")
        if args.strict:
            return 1
    else:
        print("
✅ No medical data tracked in Git. Keep archive/ outside version control.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
