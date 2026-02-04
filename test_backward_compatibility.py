"""
Backward compatibility verification script.

This script tests that the refactored DICOM reading functions maintain
backward compatibility with legacy code that doesn't pass the dataset parameter.
"""

import tempfile
import os
import sys
import pydicom
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mammography.io.dicom import create_mammography_image_from_dicom, MammographyImage


def create_test_dicom_file():
    """Create a temporary valid DICOM file for testing."""
    dataset = pydicom.Dataset()

    # Required fields for mammography
    dataset.PatientID = "COMPAT_TEST_001"
    dataset.StudyInstanceUID = "1.2.840.12345.123456789"
    dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
    dataset.SOPInstanceUID = "1.2.840.12345.456789123"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta = file_meta

    # Mammography-specific fields
    dataset.Manufacturer = "SIEMENS"
    dataset.PixelSpacing = [0.1, 0.1]
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = dataset.BitsStored - 1
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = 2048
    dataset.Columns = 1536

    # Projection information
    dataset.ViewPosition = "CC"
    dataset.ImageLaterality = "L"

    # Create pixel data
    dataset.PixelData = np.random.randint(
        0, 4095, (2048, 1536), dtype=np.uint16
    ).tobytes()

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.dcm', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        dataset.save_as(temp_path, enforce_file_format=True)
    except TypeError:
        dataset.save_as(temp_path, write_like_original=False)

    return temp_path


def test_create_mammography_image_without_dataset():
    """Test that create_mammography_image_from_dicom works with only file_path."""
    print("Test 1: create_mammography_image_from_dicom with only file_path")
    print("-" * 70)

    temp_path = create_test_dicom_file()

    try:
        # Call with only file_path (legacy usage)
        image = create_mammography_image_from_dicom(temp_path)

        # Verify the image was created correctly
        assert image is not None, "Image should not be None"
        assert isinstance(image, MammographyImage), "Should return MammographyImage instance"
        assert image.patient_id == "COMPAT_TEST_001", "Patient ID should be preserved"
        assert image.manufacturer == "SIEMENS", "Manufacturer should be preserved"
        assert image.projection_type == "CC", "Projection type should be preserved"
        assert image.laterality == "L", "Laterality should be preserved"

        print("✓ create_mammography_image_from_dicom(file_path) works correctly")
        print(f"  - Created MammographyImage instance")
        print(f"  - Patient ID: {image.patient_id}")
        print(f"  - Manufacturer: {image.manufacturer}")
        print(f"  - Projection: {image.projection_type}")
        print(f"  - Laterality: {image.laterality}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_validate_dicom_file_without_dataset():
    """Test that validate_dicom_file works without dataset parameter."""
    print("\nTest 2: MammographyImage.validate_dicom_file without dataset parameter")
    print("-" * 70)

    temp_path = create_test_dicom_file()

    try:
        # Create image using legacy method (no dataset)
        image = create_mammography_image_from_dicom(temp_path)

        # Call validate_dicom_file without dataset parameter (legacy usage)
        is_valid = image.validate_dicom_file()

        assert is_valid == True, "Validation should succeed for valid DICOM"

        print("✓ validate_dicom_file() works correctly without dataset parameter")
        print(f"  - Validation result: {is_valid}")
        print(f"  - File was read internally and validated successfully")
        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_create_mammography_image_with_dataset():
    """Test that create_mammography_image_from_dicom works WITH dataset (new usage)."""
    print("\nTest 3: create_mammography_image_from_dicom WITH dataset (new usage)")
    print("-" * 70)

    temp_path = create_test_dicom_file()

    try:
        # Read dataset first (new optimized usage)
        dataset = pydicom.dcmread(temp_path)

        # Call with dataset parameter
        image = create_mammography_image_from_dicom(temp_path, dataset=dataset)

        # Verify the image was created correctly
        assert image is not None, "Image should not be None"
        assert isinstance(image, MammographyImage), "Should return MammographyImage instance"
        assert image.patient_id == "COMPAT_TEST_001", "Patient ID should be preserved"

        print("✓ create_mammography_image_from_dicom(file_path, dataset=dataset) works correctly")
        print(f"  - Created MammographyImage instance using pre-loaded dataset")
        print(f"  - Patient ID: {image.patient_id}")
        print(f"  - This usage avoids redundant file reads (optimized)")
        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    """Run all backward compatibility tests."""
    print("=" * 70)
    print("BACKWARD COMPATIBILITY VERIFICATION")
    print("=" * 70)
    print()

    results = []

    # Test 1: create_mammography_image_from_dicom with only file_path
    results.append(("create_mammography_image_from_dicom(file_path)",
                   test_create_mammography_image_without_dataset()))

    # Test 2: validate_dicom_file without dataset parameter
    results.append(("validate_dicom_file()",
                   test_validate_dicom_file_without_dataset()))

    # Test 3: create_mammography_image_from_dicom with dataset (new usage)
    results.append(("create_mammography_image_from_dicom(file_path, dataset)",
                   test_create_mammography_image_with_dataset()))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All backward compatibility tests PASSED")
        print("✓ Legacy code will continue to work without modifications")
        print("✓ New optimized usage is available for performance improvements")
        return 0
    else:
        print("✗ Some tests FAILED - backward compatibility is broken!")
        return 1


if __name__ == "__main__":
    exit(main())
