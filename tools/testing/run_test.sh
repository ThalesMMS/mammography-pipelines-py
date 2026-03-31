#!/bin/bash
.test_venv/bin/python -m pytest tests/integration/test_view_specific_training.py -v
echo "Exit code: $?"
