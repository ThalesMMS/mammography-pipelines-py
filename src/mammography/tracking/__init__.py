#!/usr/bin/env python3
#
# __init__.py
# mammography-pipelines
#
# Local experiment tracking with SQLite backend for offline/air-gapped environments.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Local experiment tracking for mammography pipelines."""

from mammography.tracking.local_tracker import LocalTracker

__all__ = ["LocalTracker"]
