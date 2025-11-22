#!/usr/bin/env python3
#
# train_rl.py
# mammography-pipelines-py
#
# CLI wrapper that launches the RL refinement stub from the repository root.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""CLI wrapper to launch the RL refinement stub from repo root."""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.rl.train import main

if __name__ == "__main__":
    sys.exit(main())
