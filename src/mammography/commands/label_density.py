#!/usr/bin/env python3
#
# label_density.py
# mammography-pipelines
#
# Wrapper to launch the density classifier GUI.
#
"""Wrapper to launch the density classifier Streamlit UI."""
from __future__ import annotations

from typing import Sequence

from mammography.apps.density_classifier.streamlit_app import run as run_streamlit


def main(argv: Sequence[str] | None = None) -> int:
    return run_streamlit(argv)


if __name__ == "__main__":
    raise SystemExit(main())
