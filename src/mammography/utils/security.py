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
    path_text = os.fspath(path)
    if "\x00" in path_text:
        raise ValueError("Path must not contain null bytes")
    normalized = os.path.abspath(os.path.normpath(os.path.expanduser(path_text)))
    resolved = Path(normalized)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def safe_child_path(base_dir: Union[str, Path], child_name: Union[str, Path]) -> Path:
    """Join a single user/data-derived child name under base_dir without escape."""
    child_text = os.fspath(child_name)
    if "\x00" in child_text:
        raise ValueError("Path must not contain null bytes")
    if not child_text or Path(child_text).is_absolute():
        raise ValueError("Child path must be a relative name")

    base_path = resolve_path(base_dir, must_exist=False)
    child_path = resolve_within_base(child_text, base_path)
    return child_path


def resolve_within_base(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve path and reject values outside base_dir."""
    del must_exist
    base_text = os.fspath(base_dir)
    path_text = os.fspath(path)
    if "\x00" in base_text or "\x00" in path_text:
        raise ValueError("Path must not contain null bytes")

    base_path_text = os.path.normpath(os.path.realpath(os.path.expanduser(base_text)))
    expanded_path = os.path.expanduser(path_text)
    if os.path.isabs(expanded_path):
        joined_path = expanded_path
    else:
        joined_path = os.path.join(base_path_text, expanded_path)

    candidate_text = os.path.normpath(os.path.realpath(joined_path))
    base_prefix = base_path_text
    if not base_prefix.endswith(os.sep):
        base_prefix = f"{base_prefix}{os.sep}"
    if candidate_text != base_path_text and not candidate_text.startswith(base_prefix):
        raise ValueError("Path must stay within base directory")
    if os.path.commonpath([base_path_text, candidate_text]) != base_path_text:
        raise ValueError("Path must stay within base directory")
    return Path(candidate_text)
