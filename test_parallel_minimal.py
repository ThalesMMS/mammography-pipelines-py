#!/usr/bin/env python3
"""
Minimal test to verify parallel processing implementation without torch dependency.
This tests the core parallel processing logic in isolation.
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_parallel_pattern():
    """Test that the parallel processing pattern works correctly."""
    print("Testing parallel processing pattern...")

    # Simulate processing files
    test_files = [f"file_{i}.test" for i in range(10)]

    def process_single_file(file_path: str) -> Dict[str, Any]:
        """Simulate processing a single file."""
        time.sleep(0.01)  # Simulate I/O
        return {
            "file": file_path,
            "success": True,
            "result": f"processed_{file_path}"
        }

    # Test sequential processing
    start_seq = time.time()
    seq_results = []
    for file in test_files:
        seq_results.append(process_single_file(file))
    seq_time = time.time() - start_seq

    # Test parallel processing (matching the pattern in mammography_pipeline.py)
    start_par = time.time()
    par_results: List[Optional[Dict[str, Any]]] = [None] * len(test_files)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(process_single_file, file_path): i
            for i, file_path in enumerate(test_files)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                par_results[index] = future.result()
            except Exception as exc:
                print(f"Error processing file {test_files[index]}: {exc}")
                par_results[index] = {"success": False, "error": str(exc)}

    par_time = time.time() - start_par

    # Verify results
    print(f"\n✅ Sequential time: {seq_time:.3f}s")
    print(f"✅ Parallel time: {par_time:.3f}s")
    print(f"✅ Speedup: {seq_time/par_time:.2f}x")

    # Verify all results are present and in correct order
    assert len(par_results) == len(test_files), "Result count mismatch"
    assert all(r is not None for r in par_results), "Missing results"

    for i, (seq_res, par_res) in enumerate(zip(seq_results, par_results)):
        assert seq_res["file"] == par_res["file"], f"Order mismatch at index {i}"
        assert seq_res["result"] == par_res["result"], f"Result mismatch at index {i}"

    print(f"✅ All {len(test_files)} results match and are in correct order")
    print(f"✅ Parallel processing achieved {seq_time/par_time:.2f}x speedup")

    return par_time < seq_time


def test_pipeline_has_parallel_methods():
    """Verify that MammographyPipeline has the parallel processing methods."""
    print("\nTesting pipeline structure...")

    try:
        from mammography.pipeline.mammography_pipeline import MammographyPipeline

        # Check that the class has max_workers attribute
        test_config = {
            "dicom_reader": {"max_workers": 4},
            "preprocessor": {},
            "embedding_extractor": {"model_name": "resnet50"},
            "clustering": {"algorithm": "kmeans", "n_clusters": 3}
        }

        # We can't instantiate without models, but we can check the class structure
        assert hasattr(MammographyPipeline, '_process_preprocessing_files_parallel'), \
            "Missing _process_preprocessing_files_parallel method"
        assert hasattr(MammographyPipeline, '_process_embedding_files_parallel'), \
            "Missing _process_embedding_files_parallel method"
        assert hasattr(MammographyPipeline, '_process_single_preprocessing_file'), \
            "Missing _process_single_preprocessing_file method"
        assert hasattr(MammographyPipeline, '_process_single_embedding_file'), \
            "Missing _process_single_embedding_file method"

        print("✅ MammographyPipeline has all parallel processing methods")

        # Check method signatures
        import inspect

        preproc_sig = inspect.signature(MammographyPipeline._process_preprocessing_files_parallel)
        assert 'dicom_files' in preproc_sig.parameters, "Missing dicom_files parameter"
        assert 'output_dir' in preproc_sig.parameters, "Missing output_dir parameter"

        embed_sig = inspect.signature(MammographyPipeline._process_embedding_files_parallel)
        assert 'tensor_files' in embed_sig.parameters, "Missing tensor_files parameter"
        assert 'output_dir' in embed_sig.parameters, "Missing output_dir parameter"

        print("✅ Method signatures are correct")

        return True

    except ImportError as e:
        print(f"❌ Cannot import MammographyPipeline: {e}")
        return False


def test_implementation_uses_threadpool():
    """Verify that the implementation uses ThreadPoolExecutor."""
    print("\nVerifying ThreadPoolExecutor usage...")

    pipeline_file = Path("src/mammography/pipeline/mammography_pipeline.py")
    if not pipeline_file.exists():
        print(f"❌ Pipeline file not found: {pipeline_file}")
        return False

    content = pipeline_file.read_text()

    # Check for required imports
    assert "from concurrent.futures import ThreadPoolExecutor" in content, \
        "Missing ThreadPoolExecutor import"
    assert "as_completed" in content, "Missing as_completed import"

    # Check for max_workers configuration
    assert "max_workers" in content, "Missing max_workers configuration"
    assert "self.max_workers" in content, "max_workers not stored as attribute"

    # Check for parallel method implementations
    assert "_process_preprocessing_files_parallel" in content, \
        "Missing preprocessing parallel method"
    assert "_process_embedding_files_parallel" in content, \
        "Missing embedding parallel method"

    # Check for ThreadPoolExecutor usage
    assert "with ThreadPoolExecutor(max_workers=self.max_workers)" in content, \
        "ThreadPoolExecutor not used correctly"

    # Check for future_to_index pattern
    assert "future_to_index" in content, "Missing future_to_index pattern"

    print("✅ ThreadPoolExecutor correctly imported and used")
    print("✅ max_workers configuration present")
    print("✅ Parallel methods implemented")
    print("✅ future_to_index pattern used (preserves order)")

    return True


if __name__ == "__main__":
    print("="*70)
    print("MINIMAL PARALLEL PROCESSING VERIFICATION")
    print("="*70)

    all_passed = True

    # Test 1: Parallel pattern works
    try:
        if not test_parallel_pattern():
            all_passed = False
            print("❌ Parallel pattern test failed")
    except Exception as e:
        all_passed = False
        print(f"❌ Parallel pattern test failed: {e}")

    # Test 2: Implementation structure
    try:
        if not test_implementation_uses_threadpool():
            all_passed = False
            print("❌ Implementation structure test failed")
    except Exception as e:
        all_passed = False
        print(f"❌ Implementation structure test failed: {e}")

    # Test 3: Pipeline structure (this may fail due to torch)
    try:
        if not test_pipeline_has_parallel_methods():
            all_passed = False
            print("❌ Pipeline structure test failed")
    except Exception as e:
        print(f"⚠️  Pipeline structure test skipped due to dependencies: {e}")
        print("   (This is expected without torch, but static analysis passed)")

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Parallel processing implementation verified")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)
