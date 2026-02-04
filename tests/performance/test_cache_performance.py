"""
Performance benchmarks for DICOM caching and lazy loading.

This module provides comprehensive performance testing for DICOM caching
and lazy loading features, measuring memory usage reduction and loading
time improvements.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Lazy loading defers pixel data loading to reduce memory footprint
- LRU caching improves performance for repeated file access
- Performance benchmarks validate optimization effectiveness
- Memory optimization is crucial for large medical imaging datasets

Author: Research Team
Version: 1.0.0
"""

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
pydicom = pytest.importorskip("pydicom")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import cache and lazy loading components
from mammography.io.lazy_dicom import LazyDicomDataset
from mammography.io.dicom_cache import DicomLRUCache
from mammography.io.dicom import DicomReader

# Configure logging
logger = logging.getLogger(__name__)

# Performance benchmark thresholds (from acceptance criteria)
MEMORY_REDUCTION_TARGET = 0.50  # 50% memory reduction
CACHE_TIME_REDUCTION_TARGET = 0.80  # 80% time reduction for cached access
CACHE_MISS_OVERHEAD_MAX = 0.05  # 5% overhead for cache miss


class PerformanceMonitor:
    """
    Performance monitoring utility for cache and lazy loading benchmarks.

    This class provides methods for monitoring memory usage and processing time
    during DICOM loading operations with and without caching/lazy loading.

    Educational Notes:
    - Memory monitoring identifies optimization effectiveness
    - Time monitoring validates performance improvements
    - Baseline comparison ensures measurable benefits
    - Performance metrics enable cache tuning
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring."""
        # Force garbage collection for accurate baseline
        gc.collect()

        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update_monitoring(self):
        """Update performance monitoring."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop performance monitoring and return metrics."""
        # Force garbage collection before final measurement
        gc.collect()

        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "processing_time": end_time - self.start_time,
            "memory_usage_mb": end_memory - self.start_memory,
            "peak_memory_mb": self.peak_memory - self.start_memory,
        }


@contextmanager
def performance_context(monitor: PerformanceMonitor):
    """Context manager for performance monitoring."""
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        logger.info(f"Performance metrics: {metrics}")


class TestCachePerformance:
    """
    Test suite for DICOM cache performance benchmarks.

    This class provides comprehensive performance testing for DICOM caching,
    measuring cache hit rates, loading times, and cache overhead.

    Educational Notes:
    - Cache performance directly impacts user experience
    - Hit rate optimization reduces disk I/O bottlenecks
    - Cache overhead should be minimal for misses
    - Performance validation ensures production readiness
    """

    @pytest.fixture
    def sample_dicom_files(self, tmp_path: Path) -> List[Path]:
        """Create sample DICOM files for testing."""
        dicom_files = []

        # Create realistic DICOM files with pixel data
        for i in range(50):
            dicom_path = tmp_path / f"sample_{i:04d}.dcm"

            # Create a DICOM file with realistic metadata and pixel data
            ds = pydicom.dataset.FileDataset(
                str(dicom_path),
                {},
                file_meta=pydicom.dataset.FileMetaDataset(),
                preamble=b"\x00" * 128,
            )

            # Add required DICOM metadata
            ds.PatientName = f"Test^Patient{i}"
            ds.PatientID = f"PID{i:04d}"
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography
            ds.SOPInstanceUID = f"1.2.3.{i}"
            ds.StudyInstanceUID = "1.2.3.4"
            ds.SeriesInstanceUID = "1.2.3.4.5"
            ds.Modality = "MG"

            # Add pixel data (simulating mammography image)
            # Using small size for faster tests: 512x512 pixels
            ds.Rows = 512
            ds.Columns = 512
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # Create pixel array (2MB per file with 16-bit pixels)
            pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
            ds.PixelData = pixel_array.tobytes()

            # Save DICOM file
            ds.save_as(str(dicom_path), write_like_original=False)
            dicom_files.append(dicom_path)

        logger.info(f"Created {len(dicom_files)} sample DICOM files")
        return dicom_files

    @pytest.fixture
    def performance_config(self) -> Dict[str, Any]:
        """Performance testing configuration."""
        return {
            "cache_size": 100,
            "test_iterations": 10,
            "warmup_iterations": 2,
        }

    def test_cache_hit_rate(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """Test cache hit rate over repeated accesses."""
        logger.info("Testing cache hit rate...")

        cache = DicomLRUCache(max_size=performance_config["cache_size"])

        # Access files multiple times
        for iteration in range(performance_config["test_iterations"]):
            for dicom_file in sample_dicom_files[:20]:  # Use subset for faster test
                _ = cache.get(dicom_file)

        # Verify hit rate
        stats = cache.stats
        logger.info(f"Cache statistics: {stats}")

        # After first iteration, all subsequent accesses should be hits
        # Expected: 20 misses (first iteration) + 180 hits (9 more iterations)
        expected_hits = (performance_config["test_iterations"] - 1) * 20
        assert stats["hits"] >= expected_hits * 0.9, (
            f"Cache hit count too low: {stats['hits']} < {expected_hits * 0.9}"
        )

        # Hit rate should be very high (>90%)
        assert stats["hit_rate"] >= 0.90, (
            f"Cache hit rate too low: {stats['hit_rate']:.2%} < 90%"
        )

    def test_cache_loading_time_improvement(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """
        Test loading time improvement with cache.

        Verifies that cached access is >= 80% faster than uncached access
        (acceptance criteria).
        """
        logger.info("Testing cache loading time improvement...")

        test_files = sample_dicom_files[:10]  # Use 10 files for benchmark

        # Benchmark 1: Loading without cache (baseline)
        monitor_nocache = PerformanceMonitor()
        with performance_context(monitor_nocache) as perf:
            for _ in range(performance_config["test_iterations"]):
                for dicom_file in test_files:
                    # Load directly with pydicom (no cache)
                    _ = pydicom.dcmread(str(dicom_file))
                    perf.update_monitoring()

        nocache_metrics = perf.stop_monitoring()
        nocache_time = nocache_metrics["processing_time"]
        logger.info(f"No-cache loading time: {nocache_time:.3f}s")

        # Benchmark 2: Loading with cache
        cache = DicomLRUCache(max_size=performance_config["cache_size"])

        # Warmup: Load all files into cache
        for dicom_file in test_files:
            _ = cache.get(dicom_file)

        monitor_cache = PerformanceMonitor()
        with performance_context(monitor_cache) as perf:
            for _ in range(performance_config["test_iterations"]):
                for dicom_file in test_files:
                    # Load from cache
                    _ = cache.get(dicom_file)
                    perf.update_monitoring()

        cache_metrics = perf.stop_monitoring()
        cache_time = cache_metrics["processing_time"]
        logger.info(f"Cached loading time: {cache_time:.3f}s")

        # Calculate improvement
        time_reduction = (nocache_time - cache_time) / nocache_time
        logger.info(f"Cache time reduction: {time_reduction:.2%}")

        # Verify acceptance criteria: >= 80% time reduction
        assert time_reduction >= CACHE_TIME_REDUCTION_TARGET, (
            f"Cache time reduction {time_reduction:.2%} below target "
            f"{CACHE_TIME_REDUCTION_TARGET:.2%}"
        )

    def test_cache_miss_overhead(
        self, sample_dicom_files: List[Path], performance_config: Dict[str, Any]
    ):
        """
        Test that cache miss overhead is minimal (<5%).

        Cache misses should not add significant overhead compared to direct loading.
        """
        logger.info("Testing cache miss overhead...")

        test_files = sample_dicom_files[:10]

        # Benchmark 1: Direct loading (baseline)
        monitor_direct = PerformanceMonitor()
        with performance_context(monitor_direct) as perf:
            for dicom_file in test_files:
                _ = pydicom.dcmread(str(dicom_file))
                perf.update_monitoring()

        direct_metrics = perf.stop_monitoring()
        direct_time = direct_metrics["processing_time"]
        logger.info(f"Direct loading time: {direct_time:.3f}s")

        # Benchmark 2: Cache miss loading (cache cleared each time)
        monitor_cachemiss = PerformanceMonitor()
        with performance_context(monitor_cachemiss) as perf:
            cache = DicomLRUCache(max_size=performance_config["cache_size"])
            for dicom_file in test_files:
                cache.clear()  # Force cache miss
                _ = cache.get(dicom_file)
                perf.update_monitoring()

        cachemiss_metrics = perf.stop_monitoring()
        cachemiss_time = cachemiss_metrics["processing_time"]
        logger.info(f"Cache miss loading time: {cachemiss_time:.3f}s")

        # Calculate overhead
        overhead = (cachemiss_time - direct_time) / direct_time
        logger.info(f"Cache miss overhead: {overhead:.2%}")

        # Verify overhead is minimal (< 5%)
        assert overhead <= CACHE_MISS_OVERHEAD_MAX, (
            f"Cache miss overhead {overhead:.2%} exceeds maximum "
            f"{CACHE_MISS_OVERHEAD_MAX:.2%}"
        )


class TestLazyLoadingPerformance:
    """
    Test suite for lazy loading performance benchmarks.

    This class provides comprehensive performance testing for lazy DICOM loading,
    measuring memory usage reduction compared to eager loading.

    Educational Notes:
    - Lazy loading defers expensive pixel data loading
    - Memory reduction is critical for large datasets
    - Metadata access should remain fast without pixel data
    - Trade-offs between memory and computation time
    """

    @pytest.fixture
    def sample_dicom_files(self, tmp_path: Path) -> List[Path]:
        """Create sample DICOM files for testing."""
        dicom_files = []

        # Create DICOM files with larger pixel data for memory testing
        for i in range(20):
            dicom_path = tmp_path / f"large_{i:04d}.dcm"

            ds = pydicom.dataset.FileDataset(
                str(dicom_path),
                {},
                file_meta=pydicom.dataset.FileMetaDataset(),
                preamble=b"\x00" * 128,
            )

            # Add required metadata
            ds.PatientName = f"Test^LargePatient{i}"
            ds.PatientID = f"LPID{i:04d}"
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
            ds.SOPInstanceUID = f"1.2.3.100.{i}"
            ds.StudyInstanceUID = "1.2.3.4.100"
            ds.SeriesInstanceUID = "1.2.3.4.5.100"
            ds.Modality = "MG"

            # Add larger pixel data (1024x1024 = 2MB per file with 16-bit pixels)
            ds.Rows = 1024
            ds.Columns = 1024
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # Create 2MB pixel array per file
            pixel_array = np.random.randint(0, 4096, (1024, 1024), dtype=np.uint16)
            ds.PixelData = pixel_array.tobytes()

            ds.save_as(str(dicom_path), write_like_original=False)
            dicom_files.append(dicom_path)

        logger.info(f"Created {len(dicom_files)} large DICOM files for memory testing")
        return dicom_files

    def test_lazy_loading_memory_reduction(self, sample_dicom_files: List[Path]):
        """
        Test memory usage reduction with lazy loading.

        Verifies that lazy loading reduces memory usage by >= 50%
        (acceptance criteria).
        """
        logger.info("Testing lazy loading memory reduction...")

        # Benchmark 1: Eager loading (load all pixel data immediately)
        monitor_eager = PerformanceMonitor()
        with performance_context(monitor_eager) as perf:
            eager_datasets = []
            for dicom_file in sample_dicom_files:
                # Load with pixel data
                ds = pydicom.dcmread(str(dicom_file))
                _ = ds.pixel_array  # Force pixel array access
                eager_datasets.append(ds)
                perf.update_monitoring()

        eager_metrics = perf.stop_monitoring()
        eager_memory = eager_metrics["peak_memory_mb"]
        logger.info(f"Eager loading peak memory: {eager_memory:.2f} MB")

        # Clear memory
        del eager_datasets
        gc.collect()

        # Benchmark 2: Lazy loading (load metadata only, no pixel data)
        monitor_lazy = PerformanceMonitor()
        with performance_context(monitor_lazy) as perf:
            lazy_datasets = []
            for dicom_file in sample_dicom_files:
                # Load with lazy loading (metadata only)
                ds = LazyDicomDataset(str(dicom_file), stop_before_pixels=True)
                # Access metadata only (no pixel data)
                _ = ds.PatientID
                lazy_datasets.append(ds)
                perf.update_monitoring()

        lazy_metrics = perf.stop_monitoring()
        lazy_memory = lazy_metrics["peak_memory_mb"]
        logger.info(f"Lazy loading peak memory: {lazy_memory:.2f} MB")

        # Calculate memory reduction
        memory_reduction = (eager_memory - lazy_memory) / eager_memory
        logger.info(f"Memory reduction: {memory_reduction:.2%}")

        # Verify acceptance criteria: >= 50% memory reduction
        assert memory_reduction >= MEMORY_REDUCTION_TARGET, (
            f"Memory reduction {memory_reduction:.2%} below target "
            f"{MEMORY_REDUCTION_TARGET:.2%}"
        )

    def test_lazy_loading_metadata_access_time(self, sample_dicom_files: List[Path]):
        """
        Test that metadata access is fast with lazy loading.

        Lazy loading should not significantly impact metadata access time.
        """
        logger.info("Testing lazy loading metadata access time...")

        test_files = sample_dicom_files[:10]

        # Benchmark 1: Eager loading metadata access
        monitor_eager = PerformanceMonitor()
        with performance_context(monitor_eager):
            for dicom_file in test_files:
                ds = pydicom.dcmread(str(dicom_file))
                # Access common metadata fields
                _ = ds.PatientID
                _ = ds.PatientName
                _ = ds.Modality

        eager_metrics = monitor_eager.stop_monitoring()
        eager_time = eager_metrics["processing_time"]
        logger.info(f"Eager loading metadata access time: {eager_time:.3f}s")

        # Benchmark 2: Lazy loading metadata access
        monitor_lazy = PerformanceMonitor()
        with performance_context(monitor_lazy):
            for dicom_file in test_files:
                ds = LazyDicomDataset(str(dicom_file), stop_before_pixels=True)
                # Access same metadata fields
                _ = ds.PatientID
                _ = ds.PatientName
                _ = ds.Modality

        lazy_metrics = monitor_lazy.stop_monitoring()
        lazy_time = lazy_metrics["processing_time"]
        logger.info(f"Lazy loading metadata access time: {lazy_time:.3f}s")

        # Lazy loading should be faster or similar (not loading pixel data)
        # Allow up to 20% slower due to wrapper overhead
        time_ratio = lazy_time / eager_time
        logger.info(f"Lazy/Eager time ratio: {time_ratio:.2f}")

        assert time_ratio <= 1.2, (
            f"Lazy loading metadata access too slow: {time_ratio:.2f}x vs eager"
        )

    def test_lazy_loading_selective_pixel_access(self, sample_dicom_files: List[Path]):
        """
        Test memory efficiency when accessing pixel data selectively.

        With lazy loading, only accessed pixel data should consume memory.
        """
        logger.info("Testing lazy loading selective pixel access...")

        # Load all files with lazy loading
        monitor_selective = PerformanceMonitor()
        with performance_context(monitor_selective) as perf:
            lazy_datasets = []
            for dicom_file in sample_dicom_files:
                ds = LazyDicomDataset(str(dicom_file), stop_before_pixels=True)
                lazy_datasets.append(ds)
                perf.update_monitoring()

            # Access pixel data for only 25% of files
            num_to_access = len(lazy_datasets) // 4
            for i in range(num_to_access):
                _ = lazy_datasets[i].pixel_array
                perf.update_monitoring()

        selective_metrics = perf.stop_monitoring()
        selective_memory = selective_metrics["peak_memory_mb"]
        logger.info(f"Selective pixel access memory: {selective_memory:.2f} MB")

        # Compare to loading all pixel data
        monitor_all = PerformanceMonitor()
        with performance_context(monitor_all) as perf:
            all_datasets = []
            for dicom_file in sample_dicom_files:
                ds = pydicom.dcmread(str(dicom_file))
                _ = ds.pixel_array
                all_datasets.append(ds)
                perf.update_monitoring()

        all_metrics = perf.stop_monitoring()
        all_memory = all_metrics["peak_memory_mb"]
        logger.info(f"All pixel data loaded memory: {all_memory:.2f} MB")

        # Selective access should use significantly less memory
        memory_ratio = selective_memory / all_memory
        logger.info(f"Selective/All memory ratio: {memory_ratio:.2%}")

        # Should use roughly 25-50% of memory (allowing some overhead)
        assert memory_ratio <= 0.60, (
            f"Selective pixel access memory too high: {memory_ratio:.2%} of full load"
        )


class TestCombinedPerformance:
    """
    Test suite for combined cache and lazy loading performance.

    This class tests the synergistic effects of using both caching and
    lazy loading together for optimal performance.

    Educational Notes:
    - Combining optimizations can provide compounding benefits
    - Cache + lazy loading minimizes both memory and I/O
    - Real-world workflows benefit from integrated optimizations
    - Comprehensive benchmarks validate production performance
    """

    @pytest.fixture
    def sample_dicom_files(self, tmp_path: Path) -> List[Path]:
        """Create sample DICOM files for testing."""
        dicom_files = []

        for i in range(30):
            dicom_path = tmp_path / f"combined_{i:04d}.dcm"

            ds = pydicom.dataset.FileDataset(
                str(dicom_path),
                {},
                file_meta=pydicom.dataset.FileMetaDataset(),
                preamble=b"\x00" * 128,
            )

            ds.PatientName = f"Test^Combined{i}"
            ds.PatientID = f"CPID{i:04d}"
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
            ds.SOPInstanceUID = f"1.2.3.200.{i}"
            ds.StudyInstanceUID = "1.2.3.4.200"
            ds.SeriesInstanceUID = "1.2.3.4.5.200"
            ds.Modality = "MG"

            ds.Rows = 512
            ds.Columns = 512
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
            ds.PixelData = pixel_array.tobytes()

            ds.save_as(str(dicom_path), write_like_original=False)
            dicom_files.append(dicom_path)

        logger.info(f"Created {len(dicom_files)} DICOM files for combined testing")
        return dicom_files

    def test_combined_cache_and_lazy_loading(self, sample_dicom_files: List[Path]):
        """
        Test performance with both caching and lazy loading enabled.

        Verifies that combining both optimizations provides maximum benefit.
        """
        logger.info("Testing combined cache and lazy loading performance...")

        test_files = sample_dicom_files[:15]
        iterations = 5

        # Benchmark 1: No optimization (baseline)
        monitor_baseline = PerformanceMonitor()
        with performance_context(monitor_baseline) as perf:
            for _ in range(iterations):
                for dicom_file in test_files:
                    ds = pydicom.dcmread(str(dicom_file))
                    _ = ds.pixel_array
                    perf.update_monitoring()

        baseline_metrics = perf.stop_monitoring()
        logger.info(
            f"Baseline: time={baseline_metrics['processing_time']:.3f}s, "
            f"memory={baseline_metrics['peak_memory_mb']:.2f}MB"
        )

        # Benchmark 2: Cache + Lazy Loading (metadata only, with cache)
        monitor_optimized = PerformanceMonitor()
        with performance_context(monitor_optimized) as perf:
            cache = DicomLRUCache(max_size=100)
            for _ in range(iterations):
                for dicom_file in test_files:
                    # Load metadata only (stop_before_pixels=True)
                    ds = cache.get(dicom_file, stop_before_pixels=True)
                    # Access metadata (no pixel data)
                    _ = ds.PatientID
                    perf.update_monitoring()

        optimized_metrics = perf.stop_monitoring()
        logger.info(
            f"Optimized: time={optimized_metrics['processing_time']:.3f}s, "
            f"memory={optimized_metrics['peak_memory_mb']:.2f}MB"
        )

        # Calculate improvements
        time_improvement = (
            baseline_metrics["processing_time"] - optimized_metrics["processing_time"]
        ) / baseline_metrics["processing_time"]

        memory_improvement = (
            baseline_metrics["peak_memory_mb"] - optimized_metrics["peak_memory_mb"]
        ) / baseline_metrics["peak_memory_mb"]

        logger.info(f"Time improvement: {time_improvement:.2%}")
        logger.info(f"Memory improvement: {memory_improvement:.2%}")

        # Cache statistics
        cache_stats = cache.stats
        logger.info(f"Cache stats: {cache_stats}")

        # Verify significant improvements
        assert time_improvement > 0, "Time should improve with optimizations"
        assert memory_improvement > 0, "Memory should improve with optimizations"
        assert cache_stats["hit_rate"] >= 0.70, (
            f"Cache hit rate too low: {cache_stats['hit_rate']:.2%}"
        )
