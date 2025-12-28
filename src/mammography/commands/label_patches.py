#!/usr/bin/env python3
#
# label_patches.py
# mammography-pipelines
#
# Wrapper to launch the patch marking GUI for drawing ROIs on mammograms.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Wrapper to launch the patch marking GUI from the repository root."""
from mammography.apps.patch_marking.main import main

if __name__ == "__main__":
    main()
