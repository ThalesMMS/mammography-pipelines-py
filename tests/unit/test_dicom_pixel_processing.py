from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from mammography.io.dicom.pixel_processing import (
    _apply_rescale,
    _is_mono1,
    _to_float32,
    allow_invalid_decimal_strings_context,
    apply_windowing,
    extract_window_parameters,
    robust_window,
)


def test_is_mono1_is_case_insensitive() -> None:
    ds = SimpleNamespace(PhotometricInterpretation="monochrome1")

    assert _is_mono1(ds) is True


def test_to_float32_preserves_float32_array_object() -> None:
    arr = np.array([1, 2, 3], dtype=np.float32)

    result = _to_float32(arr)

    assert result is arr
    assert result.dtype == np.float32


def test_apply_rescale_uses_slope_and_intercept() -> None:
    ds = SimpleNamespace(RescaleSlope="2", RescaleIntercept="-10")
    arr = np.array([10, 20], dtype=np.float32)

    result = _apply_rescale(ds, arr)

    np.testing.assert_array_equal(result, np.array([10, 30], dtype=np.float32))


def test_apply_rescale_raises_when_direct_and_lut_fail() -> None:
    ds = SimpleNamespace(RescaleSlope="invalid", RescaleIntercept="invalid")
    arr = np.array([10, 20], dtype=np.float32)

    with pytest.raises(ValueError, match="Failed to apply DICOM rescale"):
        _apply_rescale(ds, arr)


def test_invalid_decimal_context_restores_pydicom_ds() -> None:
    original_ds = pydicom.valuerep.DS

    with allow_invalid_decimal_strings_context():
        assert pydicom.valuerep.DS("not-a-decimal") == "not-a-decimal"

    assert pydicom.valuerep.DS is original_ds


def test_invalid_decimal_context_restores_after_nested_exit() -> None:
    original_ds = pydicom.valuerep.DS

    with allow_invalid_decimal_strings_context():
        patched_ds = pydicom.valuerep.DS
        with allow_invalid_decimal_strings_context():
            assert pydicom.valuerep.DS is patched_ds
        assert pydicom.valuerep.DS is patched_ds

    assert pydicom.valuerep.DS is original_ds


def test_robust_window_returns_zero_for_constant_array() -> None:
    arr = np.full((3, 3), 7, dtype=np.float32)

    result = robust_window(arr)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.zeros_like(arr))


def test_robust_window_clips_outliers() -> None:
    arr = np.array([0, 1, 2, 3, 10_000], dtype=np.float32)

    result = robust_window(arr, p_low=0, p_high=80)

    assert result.min() == 0
    assert result.max() == 1


def test_extract_window_parameters_uses_first_multivalue() -> None:
    ds = pydicom.Dataset()
    ds.WindowCenter = [40, 50]
    ds.WindowWidth = [400, 500]
    ds.PhotometricInterpretation = "MONOCHROME1"

    wc, ww, photometric = extract_window_parameters(ds, np.array([0, 1]))

    assert wc == 40.0
    assert ww == 400.0
    assert photometric == "MONOCHROME1"


def test_extract_window_parameters_falls_back_to_pixels_for_invalid_tags() -> None:
    ds = SimpleNamespace()
    ds.WindowCenter = "not-a-number"
    ds.WindowWidth = "not-a-number"
    arr = np.array([10, 30], dtype=np.float32)

    wc, ww, photometric = extract_window_parameters(ds, arr)

    assert wc == 20.0
    assert ww == 20.0
    assert photometric == "MONOCHROME2"


def test_apply_windowing_inverts_monochrome1() -> None:
    image = np.array([0, 50, 100], dtype=np.float32)

    mono2 = apply_windowing(image, wc=50, ww=100, photometric="MONOCHROME2")
    mono1 = apply_windowing(image, wc=50, ww=100, photometric="MONOCHROME1")

    np.testing.assert_allclose(mono1, 255 - mono2, atol=1)


def test_apply_windowing_inverts_lowercase_monochrome1() -> None:
    image = np.array([0, 50, 100], dtype=np.float32)

    mono2 = apply_windowing(image, wc=50, ww=100, photometric="MONOCHROME2")
    mono1 = apply_windowing(image, wc=50, ww=100, photometric="monochrome1")

    np.testing.assert_allclose(mono1, 255 - mono2, atol=1)


def test_apply_windowing_zero_width_returns_black_image() -> None:
    image = np.array([0, 50, 100], dtype=np.float32)

    result = apply_windowing(image, wc=50, ww=0, photometric="MONOCHROME2")

    np.testing.assert_array_equal(result, np.zeros_like(image, dtype=np.uint8))
