#!/usr/bin/env python3
"""Test script to verify view extraction in CSV loader"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Test the helper function first
from mammography.data.csv_loader import _extract_view_from_dicom

# Test that the function is callable
print("Testing _extract_view_from_dicom function...")
print(f"Function exists: {_extract_view_from_dicom is not None}")

# Now test the full dataframe loading
print("\nTesting load_dataset_dataframe...")
from mammography.data.csv_loader import load_dataset_dataframe

try:
    df = load_dataset_dataframe(
        str(REPO_ROOT / "classificacao.csv"),
        str(REPO_ROOT),
        dataset='archive',
    )
    if 'view' in df.columns:
        print("✓ OK - 'view' column exists in dataframe")
        print(f"  Dataframe shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  View values: {df['view'].value_counts()}")
    else:
        print("✗ FAIL - 'view' column not found in dataframe")
        print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ FAIL - Error loading dataframe: {e}")
    import traceback
    traceback.print_exc()
