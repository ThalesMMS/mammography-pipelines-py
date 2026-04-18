"""Source-checkout import shim for the src-layout package.

This keeps `python -m mammography.cli` working from the repository root before
the project is installed in editable mode.
"""
from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mammography"
if _SRC_PACKAGE.is_dir():
    if not any(Path(path).resolve() == _SRC_PACKAGE for path in __path__):
        __path__.insert(0, str(_SRC_PACKAGE))
