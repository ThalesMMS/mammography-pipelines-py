#!/usr/bin/env python3
"""Test script to verify all imports work correctly after refactoring."""
import sys
sys.path.insert(0, './src')

try:
    from mammography.io.dicom import apply_windowing, extract_window_parameters
    print("✅ Central io.dicom imports successful")
except Exception as e:
    print(f"❌ Failed to import from io.dicom: {e}")
    sys.exit(1)

try:
    from mammography.apps.density_classifier.dicom_loader import load_dicom_task
    print("✅ Density classifier imports successful")
except Exception as e:
    print(f"❌ Failed to import from density_classifier: {e}")
    sys.exit(1)

try:
    from mammography.apps.patch_marking.dicom_loader import DicomImageLoader
    print("✅ Patch marking imports successful")
except Exception as e:
    print(f"❌ Failed to import from patch_marking: {e}")
    sys.exit(1)

print("\n✅ All imports successful - refactoring verified!")
