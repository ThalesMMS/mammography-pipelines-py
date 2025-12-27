#
# test.py
# mammography-pipelines
#
# Tiny smoke test that opens a sample DICOM and prints its pixel dimensions.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Tiny smoke test that prints the dimensions of a sample DICOM file."""

import pydicom

# Load the DICOM file without requiring pixel data handlers for speed.
ds = pydicom.dcmread("src/img.dcm", force=True)

# Access image dimensions (standard DICOM tags).
rows = ds.Rows        # (0028,0010)
columns = ds.Columns  # (0028,0011)

print(f"Image dimensions: {columns} x {rows} pixels")
