"""ResNet50_Test – Breast Density Exploration research package."""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"
__description__ = "Breast Density Exploration via ResNet-50 Embeddings"

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
