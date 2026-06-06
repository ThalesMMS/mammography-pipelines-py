"""Small helpers for redacting identifiers and validating filesystem paths."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Union


def fingerprint_value(value: object, label: str = "id") -> str:
    """Return a stable non-identifying token for a sensitive value."""
    text = str(value)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"<{label}:{digest}>"


def redact_path(path: Union[str, Path], label: str = "path") -> str:
    """Return a stable non-identifying token for a potentially sensitive path."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    return f"{fingerprint_value(path_obj, label)}{suffix}"


def resolve_path(path: Union[str, Path], *, must_exist: bool = False) -> Path:
    """Resolve a user-provided path without following missing final paths."""
    return Path(path).expanduser().resolve(strict=must_exist)


def safe_child_path(base_dir: Union[str, Path], child_name: Union[str, Path]) -> Path:
    """Join a single user/data-derived child name under base_dir without escape."""
    child_text = os.fspath(child_name)
    if not child_text or Path(child_text).is_absolute():
        raise ValueError("Child path must be a relative name")

    base_path = resolve_path(base_dir, must_exist=False)
    child_path = (base_path / child_text).resolve(strict=False)
    child_path.relative_to(base_path)
    return child_path


def resolve_within_base(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve path and reject values outside base_dir."""
    base_path = resolve_path(base_dir, must_exist=False)
    candidate = resolve_path(path, must_exist=must_exist)
    candidate.relative_to(base_path)
    return candidate
