"""Tests for mammography/__init__.py import shim.

Verifies that the source-checkout shim correctly inserts the src-layout
package path so that ``python -m mammography.cli`` works from the repo root
before editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path


def test_shim_inserts_src_package_in_path() -> None:
    """The shim should add src/mammography to __path__ when the directory exists."""
    import mammography  # noqa: F401

    # The __path__ list should contain the src/mammography directory
    mammography_paths = list(mammography.__path__)
    assert len(mammography_paths) >= 1

    # At least one entry should point to src/mammography
    src_mammography = Path(__file__).resolve().parents[2] / "src" / "mammography"
    assert any(
        Path(p).resolve() == src_mammography
        for p in mammography_paths
    ), (
        f"Expected src/mammography ({src_mammography}) to be in mammography.__path__ "
        f"but got: {mammography_paths}"
    )


def test_shim_allows_submodule_import() -> None:
    """After the shim runs, importing a submodule through the top-level package should work."""
    # If the shim is working, this import should succeed
    try:
        from mammography import cli as _cli  # noqa: F401
        assert _cli is not None
    except ImportError as exc:
        raise AssertionError(
            "Import of mammography.cli failed — the src-layout shim is not working correctly"
        ) from exc


def test_shim_does_not_duplicate_path_entries() -> None:
    """Re-importing mammography should not keep inserting the same path entry."""
    import importlib
    import mammography

    src_mammography = Path(__file__).resolve().parents[2] / "src" / "mammography"
    src_entry = str(src_mammography.resolve())
    initial_paths = [str(Path(p).resolve()) for p in mammography.__path__]
    initial_count = initial_paths.count(src_entry)

    # Simulate repeated access (re-importing won't re-run module body, but length stays stable)
    importlib.reload(mammography)

    reloaded_paths = [str(Path(p).resolve()) for p in mammography.__path__]
    assert len(reloaded_paths) == len(set(reloaded_paths)), (
        f"Duplicate mammography.__path__ entries found: {reloaded_paths}"
    )
    assert reloaded_paths.count(src_entry) == initial_count, (
        f"src/mammography entry count changed after reload: {initial_paths} -> {reloaded_paths}"
    )


def test_shim_path_entry_is_string() -> None:
    """All __path__ entries must be strings, not Path objects, for importlib compatibility."""
    import mammography

    for entry in mammography.__path__:
        assert isinstance(entry, str), (
            f"Expected string in __path__, got {type(entry)!r}: {entry!r}"
        )


def test_shim_src_package_constant_points_to_correct_location() -> None:
    """The _SRC_PACKAGE constant should point to the correct absolute path."""
    import mammography

    src_package = mammography._SRC_PACKAGE
    # It should be an absolute Path
    assert src_package.is_absolute(), f"_SRC_PACKAGE is not absolute: {src_package}"
    # It should be named 'mammography' (the src-layout package directory)
    assert src_package.name == "mammography"
    # Its parent should be the 'src' directory
    assert src_package.parent.name == "src"
