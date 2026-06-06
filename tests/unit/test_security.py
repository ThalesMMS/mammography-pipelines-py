from pathlib import Path

import pytest

from mammography.utils.security import (
    resolve_path,
    resolve_within_base,
    safe_child_path,
)


def test_resolve_within_base_accepts_relative_child(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    resolved = resolve_within_base("child/file.txt", root)

    assert resolved == root / "child" / "file.txt"


def test_resolve_within_base_rejects_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(ValueError):
        resolve_within_base("../outside.txt", root)


def test_resolve_within_base_rejects_absolute_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.touch()

    with pytest.raises(ValueError):
        resolve_within_base(outside, root, must_exist=True)


def test_resolve_within_base_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sibling = tmp_path / "root-other"
    root.mkdir()
    sibling.mkdir()
    outside = sibling / "file.txt"
    outside.touch()

    with pytest.raises(ValueError):
        resolve_within_base(outside, root, must_exist=True)


def test_resolve_within_base_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").touch()
    (root / "link").symlink_to(outside)

    with pytest.raises(ValueError):
        resolve_within_base("link/secret.txt", root, must_exist=True)


def test_resolve_within_base_normalizes_missing_child(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    resolved = resolve_within_base("missing.txt", root, must_exist=True)

    assert resolved == root / "missing.txt"


def test_resolve_path_rejects_null_bytes() -> None:
    with pytest.raises(ValueError):
        resolve_path("bad\x00path")


def test_safe_child_path_rejects_null_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, "bad\x00path")
