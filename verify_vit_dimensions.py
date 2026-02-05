#!/usr/bin/env python3
"""
Script to verify actual embedding dimensions for all ViT and DeiT models.

This script inspects the model architectures to confirm the embedding
dimensions that should be expected from each model.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_vit_dimensions():
    """Check dimensions for torchvision ViT models."""
    print("\n" + "="*60)
    print("Checking torchvision ViT Model Dimensions")
    print("="*60)

    from torchvision.models import vit_b_16, vit_b_32, vit_l_16

    models = [
        ('vit_b_16', vit_b_16),
        ('vit_b_32', vit_b_32),
        ('vit_l_16', vit_l_16),
    ]

    results = []

    for name, model_fn in models:
        print(f"\n{name}:")
        try:
            # Create model without pretrained weights
            model = model_fn(pretrained=False)

            # Get the hidden dimension from the heads layer
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                hidden_dim = model.heads.head.in_features
                print(f"  ✓ Hidden dimension: {hidden_dim}")
                results.append((name, hidden_dim))

                # Test forward pass
                test_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    # Remove classification head to get features
                    model.heads = torch.nn.Identity()
                    output = model(test_input)
                    print(f"  ✓ Feature output shape: {output.shape}")
                    print(f"  ✓ Confirmed embedding dimension: {output.shape[1]}")
            else:
                print(f"  ✗ Could not find heads.head.in_features")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    return results


def check_deit_dimensions():
    """Check dimensions for timm DeiT models."""
    print("\n" + "="*60)
    print("Checking timm DeiT Model Dimensions")
    print("="*60)

    try:
        import timm
        print("✓ timm library available")
    except ImportError:
        print("✗ timm library not available - skipping DeiT tests")
        return []

    models = [
        ('deit_small_patch16_224', 'DeiT-Small'),
        ('deit_base_patch16_224', 'DeiT-Base'),
    ]

    results = []

    for model_name, display_name in models:
        print(f"\n{display_name} ({model_name}):")
        try:
            # Create model without pretrained weights
            model = timm.create_model(model_name, pretrained=False)

            # Get the hidden dimension
            if hasattr(model, 'head'):
                hidden_dim = model.head.in_features
                print(f"  ✓ Hidden dimension: {hidden_dim}")
                results.append((display_name, hidden_dim))

                # Test forward pass
                test_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    # Use forward_features to get features
                    features = model.forward_features(test_input)
                    print(f"  ✓ Feature output shape: {features.shape}")

                    # Extract class token
                    if features.dim() == 3:
                        cls_token = features[:, 0]
                        print(f"  ✓ Class token shape: {cls_token.shape}")
                        print(f"  ✓ Confirmed embedding dimension: {cls_token.shape[1]}")
            elif hasattr(model, 'num_features'):
                hidden_dim = model.num_features
                print(f"  ✓ Hidden dimension (num_features): {hidden_dim}")
                results.append((display_name, hidden_dim))
            else:
                print(f"  ✗ Could not find head.in_features or num_features")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    return results


def verify_extractor_dimensions():
    """Verify ViTExtractor reports correct dimensions."""
    print("\n" + "="*60)
    print("Verifying ViTExtractor Dimension Reporting")
    print("="*60)

    from mammography.models.embeddings.vit_extractor import ViTExtractor

    print(f"\nViTExtractor.EXPECTED_EMBEDDING_DIM: {ViTExtractor.EXPECTED_EMBEDDING_DIM}")
    print(f"Note: This is the default for ViT-B models (768)")

    # Test with actual models
    models_to_test = [
        ('vit_b_16', 768),
        ('vit_b_32', 768),
        ('vit_l_16', 1024),
    ]

    for model_name, expected_dim in models_to_test:
        print(f"\n{model_name}:")
        try:
            config = {
                'model_name': model_name,
                'pretrained': False,
                'input_adapter': '1to3_replication'
            }
            extractor = ViTExtractor(config)

            # Test extraction
            test_input = torch.randn(3, 224, 224)
            from mammography.preprocess.preprocessed_tensor import PreprocessedTensor

            pt = PreprocessedTensor(
                image_id="test",
                tensor_data=test_input,
                preprocessing_config={},
                normalization_method='z_score_per_image',
                target_size=(224, 224),
                input_adapter='1to3_replication'
            )

            embedding = extractor.extract_embedding(pt)
            if embedding:
                actual_dim = embedding.embedding.shape[0]
                print(f"  ✓ Extracted embedding dimension: {actual_dim}")
                if actual_dim == expected_dim:
                    print(f"  ✓ Matches expected dimension: {expected_dim}")
                else:
                    print(f"  ✗ Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            else:
                print(f"  ✗ Extraction failed")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Vision Transformer Embedding Dimension Verification")
    print("="*70)
    print("\nThis script verifies the actual embedding dimensions for all")
    print("supported ViT and DeiT models by inspecting their architectures.")

    # Check ViT dimensions
    vit_results = check_vit_dimensions()

    # Check DeiT dimensions
    deit_results = check_deit_dimensions()

    # Verify extractor
    verify_extractor_dimensions()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Expected Embedding Dimensions")
    print("="*70)

    print("\nTorchvision ViT Models:")
    for name, dim in vit_results:
        print(f"  - {name}: {dim} dimensions")

    if deit_results:
        print("\nTimm DeiT Models:")
        for name, dim in deit_results:
            print(f"  - {name}: {dim} dimensions")

    print("\n" + "="*70)
    print("Expected dimensions for verification:")
    print("  - vit_b_16: 768 (ViT-Base/16)")
    print("  - vit_b_32: 768 (ViT-Base/32)")
    print("  - vit_l_16: 1024 (ViT-Large/16)")
    print("  - deit_small: 384 (DeiT-Small)")
    print("  - deit_base: 768 (DeiT-Base)")
    print("="*70)


if __name__ == "__main__":
    main()
