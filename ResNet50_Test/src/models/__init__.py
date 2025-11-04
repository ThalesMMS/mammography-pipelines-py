"""
Models module for ResNet50_test
Educational Research Project - NOT for clinical use

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Este módulo contém modelos de aprendizado de máquina para
extração de embeddings e clustering de imagens mamográficas.
"""

__version__ = "1.0.0"

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
"""

def get_disclaimer() -> str:
    """Retorna o disclaimer de pesquisa obrigatório."""
    return RESEARCH_DISCLAIMER
