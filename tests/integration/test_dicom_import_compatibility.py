from __future__ import annotations


def test_dicom_public_import_paths_are_backward_compatible() -> None:
    from mammography.io import (
        DICOM_EXTS as io_dicom_exts,
        DicomReader as io_reader,
        MammographyImage as io_image,
        apply_windowing as io_apply_windowing,
        create_dicom_reader as io_create_dicom_reader,
        create_mammography_image_from_dicom as io_create_image,
        dicom_to_pil_rgb as io_dicom_to_pil_rgb,
        extract_window_parameters as io_extract_window_parameters,
        get_disclaimer as io_get_disclaimer,
        is_dicom_path as io_is_dicom_path,
        load_dicom as io_load_dicom,
        read_dicom_directory as io_read_dicom_directory,
        read_single_dicom as io_read_single_dicom,
        robust_window as io_robust_window,
    )
    from mammography.io.dicom import (
        DICOM_EXTS,
        DicomReader,
        MammographyImage,
        apply_windowing,
        create_dicom_reader,
        create_mammography_image_from_dicom,
        dicom_to_pil_rgb,
        extract_window_parameters,
        get_disclaimer,
        is_dicom_path,
        load_dicom,
        read_dicom_directory,
        read_single_dicom,
        robust_window,
    )
    from mammography.io.dicom.metadata import (
        MammographyImage as metadata_image,
        create_mammography_image_from_dicom as metadata_create_image,
    )
    from mammography.io.dicom.pixel_processing import (
        apply_windowing as pixel_apply_windowing,
        dicom_to_pil_rgb as pixel_dicom_to_pil_rgb,
        extract_window_parameters as pixel_extract_window_parameters,
        robust_window as pixel_robust_window,
    )
    from mammography.io.dicom.reader import (
        DicomReader as reader_class,
        create_dicom_reader as reader_create_dicom_reader,
        load_dicom as reader_load_dicom,
        read_dicom_directory as reader_read_dicom_directory,
        read_single_dicom as reader_read_single_dicom,
    )

    assert DICOM_EXTS is io_dicom_exts
    assert DicomReader is reader_class
    assert DicomReader is io_reader
    assert MammographyImage is metadata_image
    assert MammographyImage is io_image
    assert create_mammography_image_from_dicom is metadata_create_image
    assert create_mammography_image_from_dicom is io_create_image
    assert robust_window is pixel_robust_window
    assert robust_window is io_robust_window
    assert apply_windowing is pixel_apply_windowing
    assert apply_windowing is io_apply_windowing
    assert extract_window_parameters is pixel_extract_window_parameters
    assert extract_window_parameters is io_extract_window_parameters
    assert dicom_to_pil_rgb is pixel_dicom_to_pil_rgb
    assert dicom_to_pil_rgb is io_dicom_to_pil_rgb
    assert create_dicom_reader is reader_create_dicom_reader
    assert create_dicom_reader is io_create_dicom_reader
    assert read_single_dicom is reader_read_single_dicom
    assert read_single_dicom is io_read_single_dicom
    assert read_dicom_directory is reader_read_dicom_directory
    assert read_dicom_directory is io_read_dicom_directory
    assert load_dicom is reader_load_dicom
    assert load_dicom is io_load_dicom
    assert get_disclaimer is io_get_disclaimer
    assert is_dicom_path is io_is_dicom_path
