#!/usr/bin/env python3
#
# web.py
# mammography-pipelines
#
# Wrapper to launch the web-based UI dashboard.
#
"""Wrapper to launch the web-based UI dashboard Streamlit app."""
from __future__ import annotations

from typing import Sequence

from mammography.apps.web_ui.streamlit_app import run as run_streamlit


def main(argv: Sequence[str] | None = None) -> int:
    return run_streamlit(argv)


if __name__ == "__main__":
    raise SystemExit(main())
