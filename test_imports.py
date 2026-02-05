#!/usr/bin/env python3
"""
Quick import test to identify critical errors without running full test suite.
"""
import sys
from pathlib import Path

# Add src to path
src_root = Path(__file__).parent / "src"
sys.path.insert(0, str(src_root))

print("Testing critical imports...")

try:
    print("1. Testing CLI import...")
    from mammography import cli
    print("   ✓ CLI imported successfully")
except Exception as e:
    print(f"   ✗ CLI import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("2. Testing config import...")
    from mammography.config import TrainConfig, ExtractConfig
    print("   ✓ Config imported successfully")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("3. Testing commands import...")
    from mammography.commands import train, extract_features, inference
    print("   ✓ Commands imported successfully")
except Exception as e:
    print(f"   ✗ Commands import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("4. Testing new modules (gradcam, reproducibility, z_score_normalize)...")
    from mammography.vis.gradcam import apply_gradcam
    from mammography.utils.reproducibility import fix_seeds
    from mammography.utils.normalization import z_score_normalize
    print("   ✓ New modules imported successfully")
except Exception as e:
    print(f"   ✗ New modules import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("5. Testing model imports...")
    from mammography.models.nets import build_model
    print("   ✓ Models imported successfully")
except Exception as e:
    print(f"   ✗ Models import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("6. Testing data imports...")
    from mammography.data.dataset import MammoDensityDataset, EmbeddingStore
    print("   ✓ Data modules imported successfully")
except Exception as e:
    print(f"   ✗ Data import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("7. Testing training imports...")
    from mammography.training.engine import train_one_epoch
    print("   ✓ Training modules imported successfully")
except Exception as e:
    print(f"   ✗ Training import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("8. Testing visualization imports...")
    from mammography.vis.advanced import plot_umap_2d
    print("   ✓ Visualization modules imported successfully")
except Exception as e:
    print(f"   ✗ Visualization import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("9. Testing IO modules...")
    from mammography.io.dicom import apply_windowing, extract_window_parameters
    print("   ✓ IO modules imported successfully")
except Exception as e:
    print(f"   ✗ IO import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All critical imports passed!")
print("\nTesting CLI help command...")

try:
    exit_code = cli.main(["--help"])
    print(f"✓ CLI help command executed (exit code: {exit_code})")
except SystemExit as e:
    if e.code == 0:
        print(f"✓ CLI help command executed successfully")
    else:
        print(f"✗ CLI help command failed with exit code: {e.code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ CLI help command failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ All tests passed! ✓✓✓")
