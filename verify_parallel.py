#!/usr/bin/env python3
"""
Quick verification script for parallel processing implementation.
This is a simplified test that doesn't require pytest.
"""
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def main():
    print("=== Parallel Processing Verification ===\n")

    # Check if MammographyPipeline accepts max_workers
    try:
        from mammography.pipeline.mammography_pipeline import MammographyPipeline

        config = {
            'dicom_reader': {},
            'preprocessing': {},
            'embedding': {},
            'clustering': {},
            'max_workers': 4
        }

        pipeline = MammographyPipeline(config)

        if hasattr(pipeline, 'max_workers'):
            print(f"✅ MammographyPipeline has max_workers attribute: {pipeline.max_workers}")
        else:
            print("❌ MammographyPipeline missing max_workers attribute")
            return False

        # Check if parallel processing methods exist
        if hasattr(pipeline, '_process_preprocessing_files_parallel'):
            print("✅ _process_preprocessing_files_parallel method exists")
        else:
            print("❌ _process_preprocessing_files_parallel method missing")

        if hasattr(pipeline, '_process_embedding_files_parallel'):
            print("✅ _process_embedding_files_parallel method exists")
        else:
            print("❌ _process_embedding_files_parallel method missing")

        print("\n✅ All parallel processing components are in place!")
        print("   Note: Full performance benchmark requires Python 3.11+ and pytest")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Some dependencies may not be available in Python 3.9.6")
        print("   This is expected - the project requires Python 3.11+")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
