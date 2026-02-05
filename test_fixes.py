#!/usr/bin/env python
"""
Quick test script to verify model instantiation and device detection fixes.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def test_device_detection():
    """Test device detection returns string."""
    print("Testing device detection...")
    from mammography.utils.device_detection import get_optimal_device

    device = get_optimal_device()
    print(f"  ✓ get_optimal_device() returns: {device!r} (type: {type(device).__name__})")

    # Verify it's a string
    assert isinstance(device, str), f"Expected str, got {type(device)}"
    assert device in ["cpu", "cuda", "mps"], f"Invalid device: {device}"
    print(f"  ✓ Device is valid string: {device}")
    return True

def test_model_instantiation():
    """Test model instantiation works."""
    print("\nTesting model instantiation...")
    from mammography.models.nets import build_model
    import torch

    # Test EfficientNet
    print("  Testing EfficientNetB0...")
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
    assert model is not None, "EfficientNet model is None"
    print("  ✓ EfficientNetB0 created successfully")

    # Test ResNet50
    print("  Testing ResNet50...")
    model = build_model("resnet50", num_classes=4, pretrained=False)
    assert model is not None, "ResNet50 model is None"
    print("  ✓ ResNet50 created successfully")

    return True

def test_model_forward_pass():
    """Test model forward pass works on CPU."""
    print("\nTesting model forward pass...")
    from mammography.models.nets import build_model
    import torch

    model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
    model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Verify output shape
    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
    print(f"  ✓ Forward pass successful: input {tuple(x.shape)} → output {tuple(output.shape)}")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Model Instantiation and Device Detection Fixes")
    print("=" * 60)

    try:
        test_device_detection()
        test_model_instantiation()
        test_model_forward_pass()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
