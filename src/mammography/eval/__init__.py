"""Evaluation utilities for the mammography pipelines."""

__version__ = "1.0.0"

RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
"""
)


def get_disclaimer() -> str:
    """Return the mandatory research disclaimer."""
    return RESEARCH_DISCLAIMER
