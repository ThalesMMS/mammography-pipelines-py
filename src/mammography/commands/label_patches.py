#!/usr/bin/env python3
#
# label_patches.py
# mammography-pipelines
#
# Wrapper to launch the patch marking GUI for drawing ROIs on mammograms.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Wrapper to launch the patch marking Streamlit UI."""
from __future__ import annotations

from typing import Sequence

from mammography.apps.patch_marking.streamlit_app import run as run_streamlit


def main(argv: Sequence[str] | None = None) -> int:
    return run_streamlit(argv)


if __name__ == "__main__":
    raise SystemExit(main())
