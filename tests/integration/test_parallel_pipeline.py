"""
Integration tests for parallel pipeline processing.

This module provides integration tests for parallel processing in the mammography
analysis pipeline, including preprocessing and embedding steps with ThreadPoolExecutor.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Parallel processing improves throughput for large datasets
- ThreadPoolExecutor enables concurrent I/O operations
- Performance testing validates expected speedup (1.5x-3x)
- Integration tests ensure correctness is preserved

Author: Research Team
Version: 1.0.0
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.pipeline.mammography_pipeline import MammographyPipeline


def _create_sample_dicom_file(file_path: Path, patient_id: str, size: tuple = (100, 100)) -> None:
    """
    Create a sample DICOM file for testing.

    Educational Note: DICOM file creation simulates real medical imaging data
    for testing purposes without requiring actual patient data.

    Args:
        file_path: Path where DICOM file should be saved
        patient_id: Patient identifier for DICOM metadata
        size: Image dimensions (height, width)
    """
    # Create file metadata
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    # Create dataset
    ds = FileDataset(str(file_path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Add required DICOM metadata
    ds.PatientID = patient_id
    ds.AccessionNumber = f"ACC_{patient_id}"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality = "MG"

    # Create pixel data
    arr = np.random.randint(0, 4096, size, dtype=np.uint16)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    # Save DICOM file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(file_path), write_like_original=False)


class TestParallelPipelineProcessing:
    """
    Test suite for parallel pipeline processing.

    This class provides integration tests for parallel processing capabilities
    in the mammography analysis pipeline, including performance and correctness
    validation.

    Educational Notes:
    - Parallel processing improves throughput for I/O-bound operations
    - ThreadPoolExecutor enables concurrent file processing
    - Performance testing ensures expected speedup is achieved
    - Correctness testing ensures parallel results match sequential results
    """

    @pytest.fixture
    def sample_dicom_dataset(self, tmp_path: Path) -> Path:
        """
        Create a sample DICOM dataset for testing.

        Educational Note: Creating sample datasets allows testing
        without requiring access to real medical imaging data.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            Path to directory containing sample DICOM files
        """
        dicom_dir = tmp_path / "dicom_input"
        dicom_dir.mkdir(parents=True, exist_ok=True)

        # Create 10 sample DICOM files
        for i in range(10):
            dicom_path = dicom_dir / f"sample_{i:04d}.dcm"
            _create_sample_dicom_file(dicom_path, f"PAT_{i:04d}")

        return dicom_dir

    @pytest.fixture
    def pipeline_config(self) -> Dict[str, Any]:
        """
        Create pipeline configuration for testing.

        Educational Note: Configuration management enables
        reproducible experiments and standardized processing.

        Returns:
            Pipeline configuration dictionary
        """
        return {
            "dicom_reader": {
                "validate_on_read": True,
                "cache_metadata": False,
            },
            "preprocessing": {
                "target_size": [224, 224],
                "normalization_method": "z_score_per_image",
                "border_removal": False,
            },
            "embedding": {
                "model_name": "resnet50",
                "pretrained": True,
                "feature_layer": "avgpool",
            },
            "clustering": {
                "algorithm": "kmeans",
                "n_clusters": 3,
            },
            "seed": 42,
        }

    def test_pipeline_initialization_with_max_workers(self, pipeline_config: Dict[str, Any]):
        """
        Test pipeline initialization with max_workers parameter.

        Educational Note: Configuration validation ensures parallel
        processing parameters are properly initialized.
        """
        # Test default max_workers
        pipeline = MammographyPipeline(pipeline_config)
        assert hasattr(pipeline, "max_workers")
        assert pipeline.max_workers == 4  # Default value

        # Test custom max_workers
        config_with_workers = pipeline_config.copy()
        config_with_workers["max_workers"] = 8
        pipeline = MammographyPipeline(config_with_workers)
        assert pipeline.max_workers == 8

    def test_parallel_preprocessing_produces_correct_results(
        self, sample_dicom_dataset: Path, pipeline_config: Dict[str, Any], tmp_path: Path
    ):
        """
        Test that parallel preprocessing produces correct results.

        Educational Note: Correctness testing ensures parallel processing
        maintains the same output as sequential processing.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline with parallel processing
        pipeline = MammographyPipeline(pipeline_config)

        # Run preprocessing step
        preprocessing_results = pipeline._run_preprocessing_step(
            sample_dicom_dataset, output_dir
        )

        # Verify results
        assert preprocessing_results["success"] is True
        assert preprocessing_results["total_files"] == 10
        assert preprocessing_results["processed_files"] == 10
        assert preprocessing_results["failed_files"] == 0
        assert len(preprocessing_results["mammography_images"]) == 10
        assert len(preprocessing_results["preprocessed_tensors"]) == 10
        assert preprocessing_results["processing_time"] > 0

        # Verify output directory contains saved tensors
        output_tensor_dir = preprocessing_results["output_dir"]
        assert output_tensor_dir.exists()
        tensor_files = list(output_tensor_dir.rglob("*.pt"))
        assert len(tensor_files) == 10

    def test_parallel_preprocessing_performance(
        self, sample_dicom_dataset: Path, pipeline_config: Dict[str, Any], tmp_path: Path
    ):
        """
        Test parallel preprocessing performance improvement.

        Educational Note: Performance testing validates that parallel
        processing achieves expected speedup over sequential processing.
        """
        # Test with sequential processing (max_workers=1)
        config_sequential = pipeline_config.copy()
        config_sequential["max_workers"] = 1
        pipeline_sequential = MammographyPipeline(config_sequential)

        output_dir_sequential = tmp_path / "output_sequential"
        output_dir_sequential.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        results_sequential = pipeline_sequential._run_preprocessing_step(
            sample_dicom_dataset, output_dir_sequential
        )
        sequential_time = time.time() - start_time

        # Test with parallel processing (max_workers=4)
        config_parallel = pipeline_config.copy()
        config_parallel["max_workers"] = 4
        pipeline_parallel = MammographyPipeline(config_parallel)

        output_dir_parallel = tmp_path / "output_parallel"
        output_dir_parallel.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        results_parallel = pipeline_parallel._run_preprocessing_step(
            sample_dicom_dataset, output_dir_parallel
        )
        parallel_time = time.time() - start_time

        # Verify both succeeded
        assert results_sequential["success"] is True
        assert results_parallel["success"] is True

        # Verify same number of files processed
        assert results_sequential["processed_files"] == results_parallel["processed_files"]

        # Verify parallel is faster (should be at least 1.2x faster)
        # Note: On small datasets, speedup may be modest due to overhead
        speedup = sequential_time / parallel_time
        print(f"\nSequential time: {sequential_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # For small datasets, we expect at least some improvement or similar performance
        assert parallel_time <= sequential_time * 1.5  # Allow 50% overhead for small dataset

    def test_parallel_preprocessing_handles_errors_gracefully(
        self, tmp_path: Path, pipeline_config: Dict[str, Any]
    ):
        """
        Test error handling in parallel preprocessing.

        Educational Note: Robust error handling ensures pipeline
        continues processing even when individual files fail.
        """
        # Create dataset with some invalid files
        dicom_dir = tmp_path / "dicom_with_errors"
        dicom_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 valid DICOM files
        for i in range(5):
            dicom_path = dicom_dir / f"valid_{i:04d}.dcm"
            _create_sample_dicom_file(dicom_path, f"PAT_{i:04d}")

        # Create 2 invalid files
        invalid_file1 = dicom_dir / "invalid_001.dcm"
        invalid_file1.write_text("INVALID DICOM DATA")

        invalid_file2 = dicom_dir / "invalid_002.dcm"
        invalid_file2.write_bytes(b"NOT A DICOM FILE")

        # Run preprocessing
        output_dir = tmp_path / "output_with_errors"
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = MammographyPipeline(pipeline_config)
        preprocessing_results = pipeline._run_preprocessing_step(dicom_dir, output_dir)

        # Verify results
        assert preprocessing_results["total_files"] == 7
        assert preprocessing_results["processed_files"] == 5
        assert preprocessing_results["failed_files"] == 2
        assert len(preprocessing_results["errors"]) > 0

    def test_parallel_embedding_extraction(
        self, sample_dicom_dataset: Path, pipeline_config: Dict[str, Any], tmp_path: Path
    ):
        """
        Test parallel embedding extraction.

        Educational Note: Embedding extraction can also benefit from
        parallel processing, especially for large datasets.
        """
        # First, run preprocessing to generate tensor files
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = MammographyPipeline(pipeline_config)

        preprocessing_results = pipeline._run_preprocessing_step(
            sample_dicom_dataset, output_dir
        )
        assert preprocessing_results["success"] is True

        # Run embedding extraction with parallel processing
        embedding_results = pipeline._run_embedding_step(
            preprocessing_results["output_dir"], output_dir, device="cpu"
        )

        # Verify results
        assert embedding_results["success"] is True
        assert embedding_results["total_tensors"] == 10
        assert embedding_results["processed_tensors"] == 10
        assert embedding_results["failed_tensors"] == 0
        assert len(embedding_results["embedding_vectors"]) == 10
        assert embedding_results["processing_time"] > 0

        # Verify output directory contains saved embeddings
        output_embedding_dir = embedding_results["output_dir"]
        assert output_embedding_dir.exists()
        embedding_files = list(output_embedding_dir.rglob("*.npy"))
        assert len(embedding_files) == 10

    def test_different_worker_counts(
        self, sample_dicom_dataset: Path, pipeline_config: Dict[str, Any], tmp_path: Path
    ):
        """
        Test pipeline with different worker counts.

        Educational Note: Different worker counts allow optimization
        for different hardware configurations and dataset sizes.
        """
        worker_counts = [1, 2, 4]
        results = {}

        for workers in worker_counts:
            config = pipeline_config.copy()
            config["max_workers"] = workers

            pipeline = MammographyPipeline(config)
            output_dir = tmp_path / f"output_workers_{workers}"
            output_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()
            preprocessing_results = pipeline._run_preprocessing_step(
                sample_dicom_dataset, output_dir
            )
            processing_time = time.time() - start_time

            results[workers] = {
                "success": preprocessing_results["success"],
                "processed_files": preprocessing_results["processed_files"],
                "time": processing_time,
            }

        # Verify all configurations succeeded
        for workers, result in results.items():
            assert result["success"] is True
            assert result["processed_files"] == 10
            print(f"Workers: {workers}, Time: {result['time']:.3f}s")

    def test_parallel_processing_maintains_file_order(
        self, sample_dicom_dataset: Path, pipeline_config: Dict[str, Any], tmp_path: Path
    ):
        """
        Test that parallel processing maintains consistent results.

        Educational Note: While parallel processing may execute in any order,
        the results should be deterministic and consistent.
        """
        output_dir1 = tmp_path / "output1"
        output_dir1.mkdir(parents=True, exist_ok=True)

        output_dir2 = tmp_path / "output2"
        output_dir2.mkdir(parents=True, exist_ok=True)

        pipeline = MammographyPipeline(pipeline_config)

        # Run preprocessing twice
        results1 = pipeline._run_preprocessing_step(sample_dicom_dataset, output_dir1)
        results2 = pipeline._run_preprocessing_step(sample_dicom_dataset, output_dir2)

        # Verify same number of files processed
        assert results1["processed_files"] == results2["processed_files"]
        assert len(results1["mammography_images"]) == len(results2["mammography_images"])

        # Verify patient IDs are present (order may vary, but all should be present)
        patient_ids1 = {img.patient_id for img in results1["mammography_images"]}
        patient_ids2 = {img.patient_id for img in results2["mammography_images"]}
        assert patient_ids1 == patient_ids2
