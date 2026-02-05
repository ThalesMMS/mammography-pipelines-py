"""
Quick verification script for DICOM codec and pixel array fixes.
Run this to verify the implementation without pytest.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
    import pydicom
    from mammography.io.dicom import (
        dicom_to_pil_rgb,
        is_dicom_path,
        _is_mono1,
        _to_float32,
        _apply_rescale,
        robust_window,
        extract_window_parameters,
        apply_windowing,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_is_dicom_path():
    """Test DICOM path detection."""
    assert is_dicom_path("file.dcm") is True
    assert is_dicom_path("file.DICOM") is True
    assert is_dicom_path("file.png") is False
    print("✓ test_is_dicom_path passed")


def test_to_float32():
    """Test array conversion."""
    arr = np.array([1, 2, 3], dtype=np.uint16)
    result = _to_float32(arr)
    assert result.dtype == np.float32
    print("✓ test_to_float32 passed")


def test_robust_window():
    """Test robust windowing."""
    arr = np.array([0, 100, 200, 300, 400, 500], dtype=np.float32)
    result = robust_window(arr, p_low=0.0, p_high=100.0)
    assert result.min() == 0.0
    assert result.max() == 1.0
    print("✓ test_robust_window passed")


def test_apply_windowing():
    """Test windowing application."""
    image = np.array([0, 50, 100, 150, 200], dtype=np.float32)
    result = apply_windowing(image, 100.0, 100.0, "MONOCHROME2")
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    print("✓ test_apply_windowing passed")


def test_dicom_creation_and_reading():
    """Test creating and reading a DICOM file."""
    # Create test DICOM
    dataset = pydicom.Dataset()
    dataset.PatientID = "TEST_001"
    dataset.StudyInstanceUID = "1.2.3"
    dataset.SeriesInstanceUID = "1.2.3.4"
    dataset.SOPInstanceUID = "1.2.3.4.5"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta = file_meta

    dataset.Rows = 64
    dataset.Columns = 64
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.random.randint(0, 1000, (64, 64), dtype=np.uint16).tobytes()

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        dcm_path = Path(tmpdir) / "test.dcm"
        try:
            dataset.save_as(str(dcm_path), enforce_file_format=True)
        except TypeError:
            dataset.save_as(str(dcm_path), write_like_original=False)

        # Test reading with dicom_to_pil_rgb
        try:
            result = dicom_to_pil_rgb(str(dcm_path))
            assert result.mode == "RGB"
            assert result.size == (64, 64)
            print("✓ test_dicom_creation_and_reading passed")
        except Exception as e:
            print(f"✗ test_dicom_creation_and_reading failed: {e}")
            raise


def test_error_handling():
    """Test error handling for invalid files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test missing file
        try:
            dicom_to_pil_rgb("nonexistent.dcm")
            print("✗ Should have raised RuntimeError for missing file")
        except RuntimeError as e:
            if "Falha ao ler" in str(e):
                print("✓ test_error_handling (missing file) passed")
            else:
                print(f"✗ Wrong error message: {e}")

        # Test invalid file
        invalid_path = Path(tmpdir) / "invalid.dcm"
        invalid_path.write_text("Not a DICOM file")
        try:
            dicom_to_pil_rgb(str(invalid_path))
            print("✗ Should have raised RuntimeError for invalid file")
        except RuntimeError as e:
            if "Falha ao ler" in str(e):
                print("✓ test_error_handling (invalid file) passed")
            else:
                print(f"✗ Wrong error message: {e}")


def test_extract_window_parameters():
    """Test window parameter extraction."""
    dataset = pydicom.Dataset()
    dataset.WindowCenter = 2048
    dataset.WindowWidth = 4096
    dataset.PhotometricInterpretation = "MONOCHROME2"

    arr = np.zeros((10, 10), dtype=np.float32)
    wc, ww, photometric = extract_window_parameters(dataset, arr)

    assert wc == 2048.0
    assert ww == 4096.0
    assert photometric == "MONOCHROME2"
    print("✓ test_extract_window_parameters passed")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("DICOM Codec and Pixel Array Access - Verification Tests")
    print("=" * 60 + "\n")

    tests = [
        test_is_dicom_path,
        test_to_float32,
        test_robust_window,
        test_apply_windowing,
        test_extract_window_parameters,
        test_dicom_creation_and_reading,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("✅ All verification tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
