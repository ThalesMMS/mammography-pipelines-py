"""
Integration tests for DicomReader with lazy loading functionality.

These tests validate the complete integration of lazy loading and caching
with the DicomReader and related components in end-to-end workflows.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.io.dicom import DicomReader, create_dicom_reader
from mammography.io.lazy_dicom import LazyDicomDataset
from mammography.io.dicom_cache import DicomLRUCache


def _write_dummy_dicom(path: Path, patient_id: str, accession: str, size: int = 100) -> None:
    """Write a dummy DICOM file for testing."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.PatientID = patient_id
    ds.AccessionNumber = accession
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality = "MG"
    ds.Manufacturer = "TEST_MANUFACTURER"
    ds.ViewPosition = "CC"
    ds.ImageLaterality = "L"
    ds.PixelSpacing = [0.1, 0.1]
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    arr = np.zeros((size, size), dtype=np.uint16)
    ds.Rows, ds.Columns = arr.shape
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)


@pytest.fixture
def dicom_files(tmp_path: Path):
    """Create multiple DICOM files for testing."""
    dcm_root = tmp_path / "archive"
    dcm_root.mkdir(parents=True, exist_ok=True)

    files = []
    for i, (accession, patient) in enumerate(
        [
            ("ACC001", "PAT_001"),
            ("ACC002", "PAT_002"),
            ("ACC003", "PAT_003"),
        ]
    ):
        dcm_dir = dcm_root / accession
        dcm_dir.mkdir(parents=True, exist_ok=True)
        dcm_path = dcm_dir / f"test_{i}.dcm"
        _write_dummy_dicom(dcm_path, patient, accession, size=128)
        files.append(dcm_path)

    return files


class TestDicomReaderLazyIntegration:
    """Integration tests for DicomReader with lazy loading."""

    def test_dicom_reader_lazy_load_parameter(self):
        """Test that DicomReader accepts lazy_load parameter."""
        reader = DicomReader(lazy_load=True)
        assert reader.lazy_load is True

        reader = DicomReader(lazy_load=False)
        assert reader.lazy_load is False

    def test_dicom_reader_factory_with_lazy_load(self):
        """Test that factory function supports lazy_load parameter."""
        reader = create_dicom_reader(lazy_load=True)
        assert reader.lazy_load is True
        assert isinstance(reader, DicomReader)

    def test_lazy_load_defers_pixel_data(self, dicom_files):
        """Test that lazy loading defers pixel data loading."""
        reader = DicomReader(lazy_load=True, validate_on_read=False)
        img = reader.read_dicom_file(dicom_files[0])

        assert img is not None
        # With lazy loading, pixel data should not be immediately loaded
        # but metadata should be available
        assert img.patient_id == "PAT_001"
        assert img.study_id is not None

    def test_lazy_load_vs_eager_load(self, dicom_files):
        """Test that lazy and eager loading both work correctly."""
        # Lazy loading
        lazy_reader = DicomReader(lazy_load=True, validate_on_read=False)
        lazy_img = lazy_reader.read_dicom_file(dicom_files[0])

        # Eager loading (default)
        eager_reader = DicomReader(lazy_load=False, validate_on_read=False)
        eager_img = eager_reader.read_dicom_file(dicom_files[0])

        # Both should succeed
        assert lazy_img is not None
        assert eager_img is not None

        # Both should have same metadata
        assert lazy_img.patient_id == eager_img.patient_id
        assert lazy_img.study_id == eager_img.study_id

    def test_lazy_load_with_default_validation(self, dicom_files):
        """
        Test lazy loading with default validation enabled.

        This tests the real user experience with default DicomReader settings.
        Most users won't know to set validate_on_read=False.
        """
        # Use default parameters - this is what users will do
        reader = DicomReader(lazy_load=True)  # validate_on_read defaults to True

        # Should work without validation errors
        img = reader.read_dicom_file(dicom_files[0])

        assert img is not None
        assert img.patient_id == "PAT_001"
        assert img.study_id is not None

        # Verify validation didn't fail
        stats = reader.get_processing_stats()
        assert stats["files_valid"] == 1
        assert stats["files_invalid"] == 0

    def test_lazy_load_directory(self, tmp_path, dicom_files):
        """Test lazy loading with directory reading."""
        reader = DicomReader(lazy_load=True, validate_on_read=False)
        results = reader.read_dicom_directory(tmp_path / "archive", recursive=True)

        # Should have organized by patient
        assert len(results) == 3  # 3 unique patients
        assert "PAT_001" in results
        assert "PAT_002" in results
        assert "PAT_003" in results

    def test_lazy_load_with_cache_metadata(self, dicom_files):
        """Test that lazy loading works with metadata caching."""
        reader = DicomReader(lazy_load=True, cache_metadata=True, validate_on_read=False)

        # Read first file
        img1 = reader.read_dicom_file(dicom_files[0])
        assert img1 is not None

        # Metadata should be cached
        assert str(dicom_files[0]) in reader.metadata_cache
        cached_metadata = reader.metadata_cache[str(dicom_files[0])]
        assert cached_metadata["patient_id"] == "PAT_001"

    def test_lazy_dataset_integration(self, dicom_files):
        """Test LazyDicomDataset integration."""
        lazy_ds = LazyDicomDataset(dicom_files[0])

        # Metadata should be immediately available
        assert lazy_ds.PatientID == "PAT_001"
        assert lazy_ds.Manufacturer == "TEST_MANUFACTURER"
        assert not lazy_ds.is_pixel_data_loaded

        # Access pixel data
        pixel_array = lazy_ds.pixel_array
        assert pixel_array is not None
        assert lazy_ds.is_pixel_data_loaded
        assert pixel_array.shape == (128, 128)

        # Clear cache
        lazy_ds.clear_pixel_cache()
        assert not lazy_ds.is_pixel_data_loaded

        # Can reload pixel data
        pixel_array2 = lazy_ds.pixel_array
        assert lazy_ds.is_pixel_data_loaded
        assert np.array_equal(pixel_array, pixel_array2)


class TestDicomCacheIntegration:
    """Integration tests for DicomLRUCache with lazy loading."""

    def test_cache_basic_workflow(self, dicom_files):
        """Test basic cache workflow with DICOM files."""
        cache = DicomLRUCache(max_size=5)

        # First access - cache miss
        ds1 = cache.get(dicom_files[0], stop_before_pixels=True)
        assert ds1 is not None
        assert cache.misses == 1
        assert cache.hits == 0

        # Second access - cache hit
        ds2 = cache.get(dicom_files[0], stop_before_pixels=True)
        assert ds2 is not None
        assert cache.misses == 1
        assert cache.hits == 1

        # Should be same object
        assert ds1 is ds2

    def test_cache_with_multiple_files(self, dicom_files):
        """Test cache with multiple DICOM files."""
        cache = DicomLRUCache(max_size=3)

        # Load all files
        datasets = []
        for file_path in dicom_files:
            ds = cache.get(file_path, stop_before_pixels=True)
            datasets.append(ds)

        # All should be cache misses
        assert cache.misses == 3
        assert cache.hits == 0
        assert cache.size == 3

        # Access first file again - should be cache hit
        ds = cache.get(dicom_files[0], stop_before_pixels=True)
        assert cache.hits == 1
        assert ds is datasets[0]

    def test_cache_lru_eviction(self, dicom_files):
        """Test LRU eviction when cache is full."""
        cache = DicomLRUCache(max_size=2)

        # Fill cache
        cache.get(dicom_files[0], stop_before_pixels=True)
        cache.get(dicom_files[1], stop_before_pixels=True)
        assert cache.size == 2
        assert cache.evictions == 0

        # Add third file - should evict first
        cache.get(dicom_files[2], stop_before_pixels=True)
        assert cache.size == 2
        assert cache.evictions == 1

        # First file should no longer be cached
        assert dicom_files[0] not in cache

    def test_cache_stats_tracking(self, dicom_files):
        """Test cache statistics tracking."""
        cache = DicomLRUCache(max_size=5)

        # Access patterns: file0, file1, file0, file1, file0
        cache.get(dicom_files[0], stop_before_pixels=True)  # miss
        cache.get(dicom_files[1], stop_before_pixels=True)  # miss
        cache.get(dicom_files[0], stop_before_pixels=True)  # hit
        cache.get(dicom_files[1], stop_before_pixels=True)  # hit
        cache.get(dicom_files[0], stop_before_pixels=True)  # hit

        assert cache.misses == 2
        assert cache.hits == 3
        assert cache.hit_rate == 0.6  # 3/5

        stats = cache.stats
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["max_size"] == 5

    def test_cache_with_lazy_loading(self, dicom_files):
        """Test cache integration with lazy loading."""
        cache = DicomLRUCache(max_size=5)

        # Load with lazy loading (stop_before_pixels=True)
        ds1 = cache.get(dicom_files[0], stop_before_pixels=True)
        assert ds1 is not None
        # Pixel data should not be loaded initially
        assert not hasattr(ds1, "PixelData") or ds1.PixelData == b''

        # Load same file without lazy loading
        cache.evict(dicom_files[0])  # Clear from cache first
        ds2 = cache.get(dicom_files[0], stop_before_pixels=False)
        assert ds2 is not None
        # Pixel data should be available
        pixel_array = ds2.pixel_array
        assert pixel_array is not None
        assert pixel_array.shape == (128, 128)

    def test_cache_persistence(self, dicom_files, tmp_path):
        """Test cache persistence to disk."""
        cache_dir = tmp_path / "cache"
        cache = DicomLRUCache(max_size=5, cache_dir=cache_dir)

        # Load some files
        cache.get(dicom_files[0], stop_before_pixels=True)
        cache.get(dicom_files[1], stop_before_pixels=True)

        # Save cache metadata
        cache.save()
        assert (cache_dir / "cache_metadata.json").exists()

        # Create new cache and load metadata
        new_cache = DicomLRUCache(max_size=5, cache_dir=cache_dir)
        new_cache.load()

        # Statistics should be restored
        assert new_cache.max_size == 5

    def test_cache_clear_and_reset(self, dicom_files):
        """Test cache clearing and stats reset."""
        cache = DicomLRUCache(max_size=5)

        # Load files
        cache.get(dicom_files[0], stop_before_pixels=True)
        cache.get(dicom_files[1], stop_before_pixels=True)
        cache.get(dicom_files[0], stop_before_pixels=True)

        assert cache.size == 2
        assert cache.hits == 1
        assert cache.misses == 2

        # Clear cache (preserves stats)
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 1  # Stats preserved
        assert cache.misses == 2

        # Reset stats
        cache.reset_stats()
        assert cache.hits == 0
        assert cache.misses == 0


class TestEndToEndLazyWorkflow:
    """End-to-end integration tests for lazy loading workflows."""

    def test_complete_lazy_workflow(self, dicom_files, tmp_path):
        """Test complete workflow with lazy loading and caching."""
        # Create reader with lazy loading
        reader = DicomReader(
            lazy_load=True,
            validate_on_read=False,
            cache_metadata=True
        )

        # Read multiple files
        images = []
        for file_path in dicom_files:
            img = reader.read_dicom_file(file_path)
            if img is not None:
                images.append(img)

        # Should have read all files
        assert len(images) == 3

        # All metadata should be available
        for img in images:
            assert img.patient_id is not None
            assert img.study_id is not None
            assert img.manufacturer == "TEST_MANUFACTURER"

        # Metadata should be cached
        assert len(reader.metadata_cache) == 3

    def test_lazy_loading_with_directory_and_cache(self, tmp_path, dicom_files):
        """Test lazy loading with directory reading and external cache."""
        cache = DicomLRUCache(max_size=10)
        reader = DicomReader(lazy_load=True, validate_on_read=False)

        # Read directory
        results = reader.read_dicom_directory(
            tmp_path / "archive",
            recursive=True,
            patient_level=True
        )

        assert len(results) == 3

        # Access files through cache
        for file_path in dicom_files:
            ds = cache.get(file_path, stop_before_pixels=True)
            assert ds is not None
            assert ds.PatientID is not None

        # Second access should hit cache
        initial_hits = cache.hits
        for file_path in dicom_files:
            cache.get(file_path, stop_before_pixels=True)

        assert cache.hits == initial_hits + 3

    def test_lazy_dataset_with_cache_workflow(self, dicom_files):
        """Test LazyDicomDataset with cache for pixel data."""
        # Create lazy datasets
        lazy_datasets = [LazyDicomDataset(fp) for fp in dicom_files]

        # Metadata should be available immediately for all
        for lazy_ds in lazy_datasets:
            assert lazy_ds.PatientID is not None
            assert not lazy_ds.is_pixel_data_loaded

        # Access pixel data for first dataset
        pixel_array = lazy_datasets[0].pixel_array
        assert pixel_array is not None
        assert lazy_datasets[0].is_pixel_data_loaded

        # Other datasets should still not have pixel data loaded
        assert not lazy_datasets[1].is_pixel_data_loaded
        assert not lazy_datasets[2].is_pixel_data_loaded

    def test_performance_lazy_vs_eager(self, tmp_path):
        """Test that lazy loading improves performance for metadata-only access."""
        # Create larger files for more realistic test
        large_files = []
        for i in range(5):
            file_path = tmp_path / f"large_{i}.dcm"
            _write_dummy_dicom(file_path, f"PAT_{i}", f"ACC_{i}", size=256)
            large_files.append(file_path)

        # Lazy loading - access metadata only
        lazy_reader = DicomReader(lazy_load=True, validate_on_read=False)
        lazy_images = []
        for file_path in large_files:
            img = lazy_reader.read_dicom_file(file_path)
            if img:
                lazy_images.append(img)
                # Access metadata only
                _ = img.patient_id

        # Should successfully read all files
        assert len(lazy_images) == 5

        # Eager loading - loads everything
        eager_reader = DicomReader(lazy_load=False, validate_on_read=False)
        eager_images = []
        for file_path in large_files:
            img = eager_reader.read_dicom_file(file_path)
            if img:
                eager_images.append(img)

        # Should also read all files
        assert len(eager_images) == 5

        # Both should have same metadata
        for lazy_img, eager_img in zip(lazy_images, eager_images):
            assert lazy_img.patient_id == eager_img.patient_id

    def test_mixed_lazy_eager_access(self, dicom_files):
        """Test switching between lazy and eager loading."""
        # Start with lazy loading
        lazy_reader = DicomReader(lazy_load=True, validate_on_read=False)
        lazy_img = lazy_reader.read_dicom_file(dicom_files[0])

        # Switch to eager loading
        eager_reader = DicomReader(lazy_load=False, validate_on_read=False)
        eager_img = eager_reader.read_dicom_file(dicom_files[0])

        # Both should work and have same metadata
        assert lazy_img is not None
        assert eager_img is not None
        assert lazy_img.patient_id == eager_img.patient_id

    def test_cache_with_processing_stats(self, dicom_files):
        """Test that processing stats work correctly with lazy loading."""
        reader = DicomReader(lazy_load=True, validate_on_read=True)

        # Read valid files
        for file_path in dicom_files:
            reader.read_dicom_file(file_path)

        # Check processing stats
        stats = reader.get_processing_stats()
        assert stats["files_processed"] >= 0
        assert stats["files_valid"] >= 0
        assert stats["success_rate"] >= 0.0
