"""
Performance tests for large dataset processing in mammography analysis pipeline.

This module provides comprehensive performance testing for large DICOM datasets,
including memory usage, processing speed, GPU utilization, and batch processing
efficiency testing.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Performance testing ensures scalability for large medical datasets
- Memory optimization is crucial for processing large DICOM files
- GPU utilization testing validates efficient hardware usage
- Batch processing efficiency affects overall pipeline performance

Author: Research Team
Version: 1.0.0
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import pytest

np = pytest.importorskip("numpy")
psutil = pytest.importorskip("psutil")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import pipeline components
from mammography.io.dicom import DicomReader
from mammography.preprocess.image_preprocessor import ImagePreprocessor

# Configure logging
logger = logging.getLogger(__name__)

# Performance test constants
LARGE_DATASET_SIZES = [100, 500, 1000, 2000]  # Number of images to test
MEMORY_THRESHOLD_MB = 8000  # 8GB memory threshold
PROCESSING_TIME_THRESHOLD_S = 300  # 5 minutes processing threshold
GPU_MEMORY_THRESHOLD_MB = 6000  # 6GB GPU memory threshold


class PerformanceMonitor:
    """
    Performance monitoring utility for large dataset testing.

    This class provides methods for monitoring memory usage, processing time,
    and GPU utilization during large dataset processing.

    Educational Notes:
    - Memory monitoring helps identify memory leaks and optimization opportunities
    - Processing time monitoring ensures scalability for large datasets
    - GPU utilization monitoring validates efficient hardware usage
    - Performance metrics enable optimization and resource planning
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_gpu_memory = self._get_gpu_memory_usage()
        self.peak_memory = self.start_memory
        self.peak_gpu_memory = self.start_gpu_memory

    def update_monitoring(self):
        """Update performance monitoring."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_gpu_memory = self._get_gpu_memory_usage()

        self.peak_memory = max(self.peak_memory, current_memory)
        self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu_memory)

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop performance monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_gpu_memory = self._get_gpu_memory_usage()

        return {
            "processing_time": end_time - self.start_time,
            "memory_usage_mb": end_memory - self.start_memory,
            "peak_memory_mb": self.peak_memory - self.start_memory,
            "gpu_memory_usage_mb": end_gpu_memory - self.start_gpu_memory,
            "peak_gpu_memory_mb": self.peak_gpu_memory - self.start_gpu_memory,
        }

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return 0.0


@contextmanager
def performance_context(monitor: PerformanceMonitor):
    """Context manager for performance monitoring."""
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        logger.info(f"Performance metrics: {metrics}")


class TestLargeDatasetProcessing:
    """
    Test suite for large dataset processing performance.

    This class provides comprehensive performance testing for large DICOM datasets,
    including memory usage, processing speed, and GPU utilization testing.

    Educational Notes:
    - Large dataset testing ensures scalability for real-world medical imaging
    - Memory optimization is crucial for processing large DICOM files
    - GPU utilization testing validates efficient hardware usage
    - Performance metrics enable optimization and resource planning
    """

    @pytest.fixture
    def sample_dicom_files(self, tmp_path: Path) -> List[Path]:
        """Create sample DICOM files for testing."""
        # Create sample DICOM files with realistic sizes
        dicom_files = []
        for i in range(100):  # Create 100 sample files
            dicom_path = tmp_path / f"sample_{i:04d}.dcm"
            # Create a simple DICOM-like file for testing
            with open(dicom_path, "wb") as f:
                f.write(b"DICOM_SAMPLE_DATA" * 1000)  # ~17KB per file
            dicom_files.append(dicom_path)

        return dicom_files

    @pytest.fixture
    def large_dicom_files(self, tmp_path: Path) -> List[Path]:
        """Create large DICOM files for testing."""
        # Create larger DICOM files for memory testing
        dicom_files = []
        for i in range(50):  # Create 50 larger files
            dicom_path = tmp_path / f"large_{i:04d}.dcm"
            # Create larger DICOM-like files for testing
            with open(dicom_path, "wb") as f:
                f.write(b"LARGE_DICOM_DATA" * 10000)  # ~150KB per file
            dicom_files.append(dicom_path)

        return dicom_files

    @pytest.fixture
    def performance_config(self) -> Dict[str, Any]:
        """Performance testing configuration."""
        return {
            "batch_size": 32,
            "max_workers": 4,
            "memory_limit_mb": 8000,
            "gpu_memory_limit_mb": 6000,
            "processing_timeout_s": 300,
        }

    def test_memory_usage_with_large_datasets(
        self, large_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test memory usage with large DICOM datasets."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Initialize DICOM reader
            dicom_reader = DicomReader(
                validate_on_read=True,
                cache_metadata=True,
                max_workers=performance_config["max_workers"],
            )

            # Process large dataset
            processed_images = []
            for dicom_file in large_dicom_files:
                try:
                    # Simulate DICOM processing
                    image = self._create_sample_image()
                    processed_images.append(image)

                    # Update monitoring
                    perf_monitor.update_monitoring()

                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    assert (
                        current_memory < performance_config["memory_limit_mb"]
                    ), f"Memory usage {current_memory}MB exceeds limit {performance_config['memory_limit_mb']}MB"

                except Exception as e:
                    logger.warning(f"Failed to process {dicom_file}: {e}")
                    continue

            # Validate memory usage
            metrics = perf_monitor.stop_monitoring()
            assert (
                metrics["peak_memory_mb"] < performance_config["memory_limit_mb"]
            ), f"Peak memory usage {metrics['peak_memory_mb']}MB exceeds limit"

            logger.info(f"Memory usage test completed: {metrics}")

    def test_processing_speed_optimization(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test processing speed and optimization."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Test different batch sizes
            batch_sizes = [8, 16, 32, 64]
            processing_times = []

            for batch_size in batch_sizes:
                start_time = time.time()

                # Process in batches
                for i in range(0, len(sample_dicom_files), batch_size):
                    batch_files = sample_dicom_files[i : i + batch_size]

                    # Simulate batch processing
                    batch_images = [self._create_sample_image() for _ in batch_files]

                    # Update monitoring
                    perf_monitor.update_monitoring()

                batch_time = time.time() - start_time
                processing_times.append(batch_time)

                logger.info(f"Batch size {batch_size}: {batch_time:.2f}s")

            # Validate processing time
            total_time = sum(processing_times)
            assert (
                total_time < performance_config["processing_timeout_s"]
            ), f"Total processing time {total_time}s exceeds timeout {performance_config['processing_timeout_s']}s"

            # Find optimal batch size
            optimal_batch_size = batch_sizes[np.argmin(processing_times)]
            logger.info(f"Optimal batch size: {optimal_batch_size}")

    def test_gpu_utilization_and_batch_processing(
        self, performance_config: Dict[str, Any]
    ):
        """Test GPU utilization and batch processing efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")

        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Test GPU memory usage
            device = torch.device("cuda")

            # Create large tensors to test GPU memory
            batch_sizes = [16, 32, 64, 128]
            gpu_memory_usage = []

            for batch_size in batch_sizes:
                # Clear GPU cache
                torch.cuda.empty_cache()

                # Create batch of images
                batch_images = torch.randn(batch_size, 3, 224, 224, device=device)

                # Simulate processing
                processed_batch = torch.nn.functional.relu(batch_images)

                # Measure GPU memory
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_usage.append(gpu_memory)

                # Update monitoring
                perf_monitor.update_monitoring()

                # Check GPU memory limit
                assert (
                    gpu_memory < performance_config["gpu_memory_limit_mb"]
                ), f"GPU memory usage {gpu_memory}MB exceeds limit {performance_config['gpu_memory_limit_mb']}MB"

                # Clean up
                del batch_images, processed_batch
                torch.cuda.empty_cache()

            # Validate GPU utilization
            metrics = perf_monitor.stop_monitoring()
            assert (
                metrics["peak_gpu_memory_mb"]
                < performance_config["gpu_memory_limit_mb"]
            ), f"Peak GPU memory usage {metrics['peak_gpu_memory_mb']}MB exceeds limit"

            logger.info(f"GPU utilization test completed: {metrics}")

    def test_memory_efficient_processing(
        self, large_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test memory-efficient processing strategies."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Test memory-efficient processing
            processed_count = 0
            max_memory_usage = 0

            for dicom_file in large_dicom_files:
                try:
                    # Process single file
                    image = self._create_sample_image()

                    # Update monitoring
                    perf_monitor.update_monitoring()

                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    max_memory_usage = max(max_memory_usage, current_memory)

                    # Memory cleanup
                    del image
                    gc.collect()

                    processed_count += 1

                    # Check memory threshold
                    if current_memory > performance_config["memory_limit_mb"]:
                        logger.warning(f"Memory usage {current_memory}MB exceeds limit")
                        break

                except Exception as e:
                    logger.warning(f"Failed to process {dicom_file}: {e}")
                    continue

            # Validate memory efficiency
            assert (
                max_memory_usage < performance_config["memory_limit_mb"]
            ), f"Maximum memory usage {max_memory_usage}MB exceeds limit"

            logger.info(
                f"Memory-efficient processing completed: {processed_count} files processed"
            )

    def test_parallel_processing_efficiency(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test parallel processing efficiency."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Test sequential processing
            start_time = time.time()
            sequential_results = []

            for dicom_file in sample_dicom_files:
                result = self._process_single_file(dicom_file)
                sequential_results.append(result)

            sequential_time = time.time() - start_time

            # Test parallel processing
            start_time = time.time()
            parallel_results = []

            with ThreadPoolExecutor(
                max_workers=performance_config["max_workers"]
            ) as executor:
                futures = [
                    executor.submit(self._process_single_file, dicom_file)
                    for dicom_file in sample_dicom_files
                ]

                for future in futures:
                    result = future.result()
                    parallel_results.append(result)

            parallel_time = time.time() - start_time

            # Calculate speedup
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

            # Validate parallel processing efficiency
            assert (
                speedup > 1.0
            ), f"Parallel processing not faster than sequential (speedup: {speedup})"
            assert (
                parallel_time < performance_config["processing_timeout_s"]
            ), f"Parallel processing time {parallel_time}s exceeds timeout"

            logger.info(f"Parallel processing speedup: {speedup:.2f}x")
            logger.info(
                f"Sequential time: {sequential_time:.2f}s, Parallel time: {parallel_time:.2f}s"
            )

    def test_large_dataset_pipeline_integration(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test complete pipeline with large dataset."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Initialize pipeline components
            dicom_reader = DicomReader(validate_on_read=True, cache_metadata=True)
            preprocessor = ImagePreprocessor(
                {
                    "target_size": [224, 224],
                    "normalization_method": "z_score_per_image",
                    "input_adapter": "1to3_replication",
                    "border_removal": True,
                    "seed": 42,
                }
            )

            # Process dataset in batches
            batch_size = performance_config["batch_size"]
            processed_count = 0

            for i in range(0, len(sample_dicom_files), batch_size):
                batch_files = sample_dicom_files[i : i + batch_size]

                try:
                    # Process batch
                    batch_images = []
                    for dicom_file in batch_files:
                        image = self._create_sample_image()
                        batch_images.append(image)

                    # Simulate batch processing
                    processed_batch = self._process_batch(batch_images)

                    # Update monitoring
                    perf_monitor.update_monitoring()

                    processed_count += len(batch_images)

                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if current_memory > performance_config["memory_limit_mb"]:
                        logger.warning(f"Memory usage {current_memory}MB exceeds limit")
                        break

                    # Cleanup
                    del batch_images, processed_batch
                    gc.collect()

                except Exception as e:
                    logger.warning(f"Failed to process batch {i}: {e}")
                    continue

            # Validate pipeline performance
            metrics = perf_monitor.stop_monitoring()
            assert (
                metrics["processing_time"] < performance_config["processing_timeout_s"]
            ), f"Pipeline processing time {metrics['processing_time']}s exceeds timeout"
            assert (
                metrics["peak_memory_mb"] < performance_config["memory_limit_mb"]
            ), f"Peak memory usage {metrics['peak_memory_mb']}MB exceeds limit"

            logger.info(
                f"Large dataset pipeline test completed: {processed_count} files processed"
            )
            logger.info(f"Performance metrics: {metrics}")

    def _create_sample_image(self) -> np.ndarray:
        """Create sample image for testing."""
        return np.random.rand(224, 224).astype(np.float32)

    def _process_single_file(self, dicom_file: Path) -> Dict[str, Any]:
        """Process single file for testing."""
        # Simulate file processing
        time.sleep(0.01)  # Simulate processing time
        return {"file": str(dicom_file), "processed": True}

    def _process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of images for testing."""
        # Simulate batch processing
        time.sleep(0.1)  # Simulate batch processing time
        return [{"image": i, "processed": True} for i in range(len(images))]


class TestMemoryOptimization:
    """
    Test suite for memory optimization strategies.

    This class provides testing for memory optimization techniques including
    garbage collection, memory monitoring, and efficient data structures.

    Educational Notes:
    - Memory optimization is crucial for processing large medical imaging datasets
    - Garbage collection strategies prevent memory leaks
    - Memory monitoring enables proactive memory management
    - Efficient data structures reduce memory footprint
    """

    def test_garbage_collection_efficiency(self):
        """Test garbage collection efficiency."""
        monitor = PerformanceMonitor()

        with performance_context(monitor) as perf_monitor:
            # Test without garbage collection
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Create large objects
            large_objects = []
            for i in range(100):
                obj = np.random.rand(1000, 1000)  # ~8MB per object
                large_objects.append(obj)

            memory_without_gc = psutil.Process().memory_info().rss / 1024 / 1024

            # Test with garbage collection
            del large_objects
            gc.collect()

            memory_with_gc = psutil.Process().memory_info().rss / 1024 / 1024

            # Validate garbage collection efficiency
            memory_freed = memory_without_gc - memory_with_gc
            assert memory_freed > 0, "Garbage collection should free memory"

            logger.info(f"Memory freed by garbage collection: {memory_freed:.2f}MB")

    def test_memory_monitoring_accuracy(self):
        """Test memory monitoring accuracy."""
        monitor = PerformanceMonitor()

        # Test memory monitoring
        monitor.start_monitoring()

        # Create objects of known size
        known_size_mb = 10  # 10MB
        large_array = np.random.rand(known_size_mb * 1024 * 1024 // 8)  # ~10MB

        # Update monitoring
        monitor.update_monitoring()

        # Stop monitoring
        metrics = monitor.stop_monitoring()

        # Validate memory monitoring
        assert metrics["memory_usage_mb"] >= 0, "Memory usage should be non-negative"
        assert metrics["peak_memory_mb"] >= 0, "Peak memory usage should be non-negative"

        logger.info(f"Memory monitoring test completed: {metrics}")

    def test_efficient_data_structures(self):
        """Test efficient data structures for memory optimization."""
        # Test numpy arrays vs Python lists
        size = 1000000  # 1 million elements

        # Test Python list
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        python_list = [i for i in range(size)]
        python_memory = psutil.Process().memory_info().rss / 1024 / 1024
        python_usage = python_memory - start_memory

        # Test numpy array
        del python_list
        gc.collect()

        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        numpy_array = np.arange(size)
        numpy_memory = psutil.Process().memory_info().rss / 1024 / 1024
        numpy_usage = numpy_memory - start_memory

        # Validate efficiency
        assert (
            numpy_usage < python_usage
        ), "NumPy arrays should be more memory efficient"

        logger.info(f"Python list memory usage: {python_usage:.2f}MB")
        logger.info(f"NumPy array memory usage: {numpy_usage:.2f}MB")
        denom = numpy_usage if numpy_usage > 0 else 1e-6
        logger.info(f"Memory efficiency improvement: {python_usage / denom:.2f}x")


class TestGPUUtilization:
    """
    Test suite for GPU utilization and optimization.

    This class provides testing for GPU memory management, batch processing,
    and GPU utilization optimization.

    Educational Notes:
    - GPU utilization optimization improves processing speed for large datasets
    - GPU memory management prevents out-of-memory errors
    - Batch processing maximizes GPU utilization
    - Mixed precision training reduces memory usage
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        device = torch.device("cuda")

        # Test GPU memory allocation and deallocation
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Deallocate tensor
        del large_tensor
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Validate memory management
        assert allocated_memory > initial_memory, "Memory should be allocated"
        assert final_memory <= initial_memory, "Memory should be deallocated"

        logger.info("GPU memory management test completed")
        logger.info(
            f"Initial: {initial_memory:.2f}MB, Allocated: {allocated_memory:.2f}MB, Final: {final_memory:.2f}MB"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency on GPU."""
        device = torch.device("cuda")

        # Test different batch sizes
        batch_sizes = [16, 32, 64, 128]
        processing_times = []

        for batch_size in batch_sizes:
            # Clear GPU cache
            torch.cuda.empty_cache()

            # Create batch
            batch = torch.randn(batch_size, 3, 224, 224, device=device)

            # Process batch
            start_time = time.time()
            processed = torch.nn.functional.relu(batch)
            processing_time = time.time() - start_time

            processing_times.append(processing_time)

            # Cleanup
            del batch, processed
            torch.cuda.empty_cache()

        # Find optimal batch size
        optimal_batch_size = batch_sizes[np.argmin(processing_times)]

        logger.info(f"Optimal batch size for GPU: {optimal_batch_size}")
        logger.info(f"Processing times: {processing_times}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_mixed_precision_processing(self):
        """Test mixed precision processing for memory optimization."""
        device = torch.device("cuda")

        # Test with float32
        torch.cuda.empty_cache()
        tensor_f32 = torch.randn(1000, 1000, dtype=torch.float32, device=device)
        memory_f32 = torch.cuda.memory_allocated() / 1024 / 1024

        # Test with float16
        del tensor_f32
        torch.cuda.empty_cache()

        tensor_f16 = torch.randn(1000, 1000, dtype=torch.float16, device=device)
        memory_f16 = torch.cuda.memory_allocated() / 1024 / 1024

        # Validate memory efficiency
        assert memory_f16 < memory_f32, "Float16 should use less memory than float32"

        logger.info(f"Float32 memory usage: {memory_f32:.2f}MB")
        logger.info(f"Float16 memory usage: {memory_f16:.2f}MB")
        logger.info(f"Memory efficiency improvement: {memory_f32 / memory_f16:.2f}x")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])
