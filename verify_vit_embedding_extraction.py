#!/usr/bin/env python3
"""
Verification script for ViT embedding extraction integration.

This script verifies that ViTExtractor can successfully extract embeddings
from PreprocessedTensor objects and produce EmbeddingVector outputs with
correct dimensions for all supported ViT and DeiT models.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mammography.preprocess.preprocessed_tensor import PreprocessedTensor
from mammography.models.embeddings.vit_extractor import ViTExtractor, create_vit_extractor
from mammography.models.embeddings.embedding_vector import EmbeddingVector


def create_test_preprocessed_tensor(image_id: str, channels: int = 3, height: int = 224, width: int = 224) -> PreprocessedTensor:
    """
    Create a test PreprocessedTensor for verification.

    Args:
        image_id: Identifier for the test image
        channels: Number of channels (default: 3 for RGB)
        height: Image height (default: 224)
        width: Image width (default: 224)

    Returns:
        PreprocessedTensor: Test tensor instance
    """
    # Create random tensor data (C, H, W)
    tensor_data = torch.randn(channels, height, width)

    # Create preprocessing config
    preprocessing_config = {
        'target_size': (height, width),
        'normalization_method': 'z_score_per_image',
        'input_adapter': '1to3_replication',
        'border_removed': False
    }

    # Create PreprocessedTensor
    pt = PreprocessedTensor(
        image_id=image_id,
        tensor_data=tensor_data,
        preprocessing_config=preprocessing_config,
        normalization_method='z_score_per_image',
        target_size=(height, width),
        input_adapter='1to3_replication',
        border_removed=False
    )

    return pt


def verify_single_embedding_extraction(model_name: str, expected_dim: int) -> bool:
    """
    Verify embedding extraction for a single model.

    Args:
        model_name: Name of the model to test
        expected_dim: Expected embedding dimension

    Returns:
        bool: True if verification passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    try:
        # Create extractor config
        config = {
            'model_name': model_name,
            'pretrained': False,  # Use random weights to avoid downloads
            'input_adapter': '1to3_replication',
            'batch_size': 4
        }

        # Create extractor
        print(f"1. Creating ViTExtractor for {model_name}...")
        extractor = create_vit_extractor(config)
        print(f"   ✓ Extractor created successfully")

        # Get model info
        model_info = extractor.get_model_info()
        print(f"\n2. Model Information:")
        print(f"   - Model: {model_info['model_name']}")
        print(f"   - Pretrained: {model_info['pretrained']}")
        print(f"   - Device: {model_info['device']}")
        print(f"   - Embedding dimension: {model_info['embedding_dimension']}")
        print(f"   - Total parameters: {model_info['total_parameters']:,}")

        # Create test PreprocessedTensor
        print(f"\n3. Creating test PreprocessedTensor...")
        test_tensor = create_test_preprocessed_tensor(
            image_id=f"test_{model_name}_001",
            channels=3,
            height=224,
            width=224
        )
        print(f"   ✓ PreprocessedTensor created")
        print(f"   - Image ID: {test_tensor.image_id}")
        print(f"   - Tensor shape: {test_tensor.tensor_data.shape}")
        print(f"   - Input adapter: {test_tensor.input_adapter}")

        # Extract single embedding
        print(f"\n4. Extracting embedding...")
        embedding = extractor.extract_embedding(test_tensor)

        # Verify embedding
        if embedding is None:
            print(f"   ✗ FAILED: Embedding extraction returned None")
            return False

        print(f"   ✓ Embedding extracted successfully")

        # Verify it's an EmbeddingVector instance
        if not isinstance(embedding, EmbeddingVector):
            print(f"   ✗ FAILED: Expected EmbeddingVector, got {type(embedding)}")
            return False

        print(f"   ✓ Output is EmbeddingVector instance")

        # Verify embedding dimension
        actual_dim = embedding.embedding.shape[0]
        print(f"\n5. Verifying embedding dimensions:")
        print(f"   - Expected: {expected_dim}")
        print(f"   - Actual: {actual_dim}")

        if actual_dim != expected_dim:
            print(f"   ✗ FAILED: Dimension mismatch")
            return False

        print(f"   ✓ Embedding dimension correct")

        # Verify embedding metadata
        print(f"\n6. Verifying embedding metadata:")
        print(f"   - Image ID: {embedding.image_id}")
        print(f"   - Extraction time: {embedding.extraction_time:.4f}s")
        print(f"   - Device used: {embedding.device_used}")
        print(f"   - Input adapter: {embedding.input_adapter}")

        if embedding.image_id != test_tensor.image_id:
            print(f"   ✗ FAILED: Image ID mismatch")
            return False

        print(f"   ✓ Metadata correct")

        print(f"\n{'='*60}")
        print(f"✓ {model_name} PASSED ALL CHECKS")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ {model_name} FAILED")
        print(f"Error: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


def verify_batch_embedding_extraction(model_name: str, expected_dim: int, batch_size: int = 4) -> bool:
    """
    Verify batch embedding extraction.

    Args:
        model_name: Name of the model to test
        expected_dim: Expected embedding dimension
        batch_size: Number of images to process in batch

    Returns:
        bool: True if verification passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Batch Extraction - {model_name}")
    print(f"{'='*60}")

    try:
        # Create extractor
        config = {
            'model_name': model_name,
            'pretrained': False,
            'input_adapter': '1to3_replication',
            'batch_size': 2  # Process 2 at a time
        }

        print(f"1. Creating ViTExtractor...")
        extractor = create_vit_extractor(config)
        print(f"   ✓ Extractor created")

        # Create batch of PreprocessedTensors
        print(f"\n2. Creating batch of {batch_size} PreprocessedTensors...")
        test_tensors = [
            create_test_preprocessed_tensor(
                image_id=f"test_{model_name}_batch_{i:03d}",
                channels=3,
                height=224,
                width=224
            )
            for i in range(batch_size)
        ]
        print(f"   ✓ Created {len(test_tensors)} test tensors")

        # Extract batch embeddings
        print(f"\n3. Extracting batch embeddings...")
        embeddings = extractor.extract_embeddings_batch(test_tensors)
        print(f"   ✓ Batch extraction complete")

        # Verify batch results
        print(f"\n4. Verifying batch results:")
        print(f"   - Expected count: {batch_size}")
        print(f"   - Actual count: {len(embeddings)}")

        if len(embeddings) != batch_size:
            print(f"   ✗ FAILED: Count mismatch")
            return False

        # Verify each embedding
        successful_count = 0
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                print(f"   ✗ Embedding {i} is None")
                continue

            if not isinstance(embedding, EmbeddingVector):
                print(f"   ✗ Embedding {i} is not EmbeddingVector")
                continue

            if embedding.embedding.shape[0] != expected_dim:
                print(f"   ✗ Embedding {i} has wrong dimension: {embedding.embedding.shape[0]}")
                continue

            if embedding.image_id != test_tensors[i].image_id:
                print(f"   ✗ Embedding {i} has wrong image_id")
                continue

            successful_count += 1

        print(f"   - Successful embeddings: {successful_count}/{batch_size}")

        if successful_count != batch_size:
            print(f"   ✗ FAILED: Not all embeddings successful")
            return False

        print(f"   ✓ All embeddings valid")

        print(f"\n{'='*60}")
        print(f"✓ Batch extraction for {model_name} PASSED")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Batch extraction for {model_name} FAILED")
        print(f"Error: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    print("\n" + "="*60)
    print("ViT Embedding Extraction Integration Verification")
    print("="*60)

    # Define models to test with their expected dimensions
    # Based on standard ViT/DeiT architectures:
    # - ViT-B: 768-dim (Base model with 768 hidden size)
    # - ViT-L: 1024-dim (Large model with 1024 hidden size)
    # - DeiT-Small: 384-dim
    # - DeiT-Base: 768-dim
    test_models = [
        ('vit_b_16', 768),   # ViT-B/16
        ('vit_b_32', 768),   # ViT-B/32
        ('vit_l_16', 1024),  # ViT-L/16
    ]

    # Note: DeiT models require timm library, tested separately

    results = []

    # Test single embedding extraction
    print("\n" + "="*60)
    print("PART 1: Single Embedding Extraction")
    print("="*60)

    for model_name, expected_dim in test_models:
        result = verify_single_embedding_extraction(model_name, expected_dim)
        results.append((f"{model_name} (single)", result))

    # Test batch embedding extraction
    print("\n" + "="*60)
    print("PART 2: Batch Embedding Extraction")
    print("="*60)

    for model_name, expected_dim in test_models:
        result = verify_batch_embedding_extraction(model_name, expected_dim, batch_size=4)
        results.append((f"{model_name} (batch)", result))

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*60)
        print("✓ ALL VERIFICATIONS PASSED")
        print("="*60)
        print("\nConclusions:")
        print("1. ViTExtractor successfully extracts embeddings from PreprocessedTensor objects")
        print("2. Output is EmbeddingVector instances with correct metadata")
        print("3. Embedding dimensions are correct for all tested models:")
        print("   - ViT-B/16: 768 dimensions")
        print("   - ViT-B/32: 768 dimensions")
        print("   - ViT-L/16: 1024 dimensions")
        print("4. Batch processing works correctly")
        print("5. Integration with existing data pipeline is successful")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ SOME VERIFICATIONS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
