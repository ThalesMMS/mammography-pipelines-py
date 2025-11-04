#!/usr/bin/env python3
"""
Setup script for mammography analysis CLI commands.

This script creates command-line entry points for all pipeline components
including preprocessing, embedding extraction, clustering, and analysis.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- CLI setup enables easy access to all pipeline components
- Entry points provide unified interface for pipeline operations
- Configuration management enables reproducible experiments

Author: Research Team
Version: 1.0.0
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
"""

def main():
    """Main entry point for CLI setup."""
    print(RESEARCH_DISCLAIMER)
    print()
    
    # Import and run the main CLI
    from src.cli.main import app
    app()

if __name__ == "__main__":
    main()
