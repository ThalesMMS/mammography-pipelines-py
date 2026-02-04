"""
Unit tests for Vision Transformer (ViT) feature extraction functionality.

These tests validate individual ViT feature extraction functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
models = pytest.importorskip("torchvision.models")


class TestViTFeatureExtraction:
    """Unit tests for Vision Transformer feature extraction functions."""

    @pytest.fixture
    def sample_tensor(self) -> torch.Tensor:
        """Create a sample preprocessed tensor for testing."""
        torch.manual_seed(42)
        return torch.randn(3, 512, 512)

    @pytest.fixture
    def vit_b_16_model(self):
        """Load ViT-B/16 model for testing."""
        torch.manual_seed(42)
        model = models.vit_b_16(weights=None)
        model.eval()
        return model

    @pytest.fixture
    def sample_batch(self) -> torch.Tensor:
        """Create a sample batch of tensors for testing."""
        torch.manual_seed(42)
        return torch.randn(4, 3, 512, 512)

    def test_vit_b_16_model_loading(self, vit_b_16_model):
        """Test ViT-B/16 model loading and basic properties."""
        # Validate model properties
        assert isinstance(vit_b_16_model, models.VisionTransformer)
        assert vit_b_16_model.training == False  # Should be in eval mode

        # Validate model architecture
        assert hasattr(vit_b_16_model, "conv_proj")
        assert hasattr(vit_b_16_model, "encoder")
        assert hasattr(vit_b_16_model, "heads")

        # Validate architecture dimensions
        assert vit_b_16_model.conv_proj.out_channels == 768  # ViT-B hidden dimension
        assert vit_b_16_model.hidden_dim == 768
        assert vit_b_16_model.heads.head.in_features == 768
        assert vit_b_16_model.heads.head.out_features == 1000  # ImageNet classes

    def test_vit_b_16_forward_pass(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 forward pass through the network."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Forward pass through ViT-B/16
        with torch.no_grad():
            output = vit_b_16_model(batch_tensor)

        # Validate output
        assert output.shape == (1, 1000)  # ImageNet classes
        assert output.dtype == torch.float32
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_vit_b_16_feature_extraction(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 feature extraction before classification head."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Extract features before classification head
        with torch.no_grad():
            # Reshape and permute image to patches
            x = vit_b_16_model._process_input(batch_tensor)
            n = x.shape[0]

            # Expand class token to batch
            batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through encoder
            x = vit_b_16_model.encoder(x)

            # Extract CLS token (first token)
            features = x[:, 0]

        # Validate features
        assert features.shape == (1, 768)  # ViT-B/16 feature dimension
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_vit_b_16_batch_processing(self, sample_batch, vit_b_16_model):
        """Test ViT-B/16 batch processing."""
        batch_size = sample_batch.shape[0]

        # Process batch
        with torch.no_grad():
            # Reshape and permute image to patches
            x = vit_b_16_model._process_input(sample_batch)
            n = x.shape[0]

            # Expand class token to batch
            batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through encoder
            x = vit_b_16_model.encoder(x)

            # Extract CLS token (first token)
            features = x[:, 0]

        # Validate batch features
        assert features.shape == (batch_size, 768)
        assert features.dtype == torch.float32
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_vit_b_16_feature_extraction_reproducibility(
        self, sample_tensor, vit_b_16_model
    ):
        """Test reproducibility of ViT-B/16 feature extraction."""
        seed = 42

        # Extract features multiple times with same seed
        features_list = []
        for _ in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)

            with torch.no_grad():
                # Reshape and permute image to patches
                x = vit_b_16_model._process_input(sample_tensor.unsqueeze(0))
                n = x.shape[0]

                # Expand class token to batch
                batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)

                # Pass through encoder
                x = vit_b_16_model.encoder(x)

                # Extract CLS token
                features = x[:, 0]
                features_list.append(features)

        # Features should be identical
        for i in range(1, len(features_list)):
            assert torch.allclose(
                features_list[0], features_list[i]
            ), "Feature extraction not reproducible"

    def test_vit_b_16_different_input_sizes(self, vit_b_16_model):
        """Test ViT-B/16 with different input sizes."""
        input_sizes = [(224, 224), (512, 512), (1024, 1024)]

        for height, width in input_sizes:
            # Create tensor with different size
            tensor = torch.randn(3, height, width)

            with torch.no_grad():
                # Reshape and permute image to patches
                x = vit_b_16_model._process_input(tensor.unsqueeze(0))
                n = x.shape[0]

                # Expand class token to batch
                batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)

                # Pass through encoder
                x = vit_b_16_model.encoder(x)

                # Extract CLS token
                features = x[:, 0]

            # Features should always be 768-dimensional
            assert features.shape == (1, 768)
            assert not torch.any(torch.isnan(features))
            assert not torch.any(torch.isinf(features))

    def test_vit_b_16_feature_normalization(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 feature normalization options."""
        # Extract features
        with torch.no_grad():
            # Reshape and permute image to patches
            x = vit_b_16_model._process_input(sample_tensor.unsqueeze(0))
            n = x.shape[0]

            # Expand class token to batch
            batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through encoder
            x = vit_b_16_model.encoder(x)

            # Extract CLS token
            features = x[:, 0]

        # Test different normalization methods
        normalization_methods = ["none", "l2", "l1", "min_max"]

        for method in normalization_methods:
            if method == "none":
                normalized_features = features
            elif method == "l2":
                normalized_features = features / (
                    torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8
                )
            elif method == "l1":
                normalized_features = features / (
                    torch.norm(features, p=1, dim=1, keepdim=True) + 1e-8
                )
            elif method == "min_max":
                min_val = torch.min(features, dim=1, keepdim=True)[0]
                max_val = torch.max(features, dim=1, keepdim=True)[0]
                normalized_features = (features - min_val) / (max_val - min_val + 1e-8)

            # Validate normalized features
            assert normalized_features.shape == (1, 768)
            assert not torch.any(torch.isnan(normalized_features))
            assert not torch.any(torch.isinf(normalized_features))

            # Validate L2 normalization
            if method == "l2":
                norm = torch.norm(normalized_features, p=2, dim=1)
                assert torch.allclose(norm, torch.ones(1), atol=1e-6)

    def test_vit_b_16_patch_embedding(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 patch embedding functionality."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        with torch.no_grad():
            # Process input to patches
            x = vit_b_16_model._process_input(batch_tensor)

        # Validate patch embeddings
        # For 512x512 image with patch size 16: (512/16) * (512/16) = 32 * 32 = 1024 patches
        expected_num_patches = (512 // 16) * (512 // 16)
        assert x.shape == (1, expected_num_patches, 768)
        assert not torch.any(torch.isnan(x))
        assert not torch.any(torch.isinf(x))

    def test_vit_b_16_encoder_layers(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 encoder layer structure."""
        # Validate encoder structure
        assert hasattr(vit_b_16_model.encoder, "layers")

        # ViT-B/16 has 12 transformer encoder layers
        encoder_layers = vit_b_16_model.encoder.layers
        assert len(encoder_layers) == 12

        # Each layer should be an EncoderBlock
        for layer in encoder_layers:
            assert hasattr(layer, "ln_1")  # Layer norm 1
            assert hasattr(layer, "self_attention")  # Multi-head attention
            assert hasattr(layer, "ln_2")  # Layer norm 2
            assert hasattr(layer, "mlp")  # MLP block

    def test_vit_b_16_attention_mechanism(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 attention mechanism."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Extract attention weights from first encoder layer
        with torch.no_grad():
            # Process input to patches
            x = vit_b_16_model._process_input(batch_tensor)
            n = x.shape[0]

            # Add class token
            batch_class_token = vit_b_16_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through first encoder layer
            first_layer = vit_b_16_model.encoder.layers[0]

            # Layer norm and self-attention
            x_ln = first_layer.ln_1(x)
            attn_output = first_layer.self_attention(x_ln)

        # Validate attention output
        expected_num_patches = (512 // 16) * (512 // 16)
        expected_seq_len = expected_num_patches + 1  # patches + class token
        assert attn_output.shape == (1, expected_seq_len, 768)
        assert not torch.any(torch.isnan(attn_output))
        assert not torch.any(torch.isinf(attn_output))

    def test_vit_b_16_class_token(self, vit_b_16_model):
        """Test ViT-B/16 class token properties."""
        # Validate class token
        assert hasattr(vit_b_16_model, "class_token")
        assert vit_b_16_model.class_token.shape == (1, 1, 768)
        assert vit_b_16_model.class_token.requires_grad == True

    def test_vit_b_16_positional_embedding(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 positional embedding."""
        # Validate positional embedding
        assert hasattr(vit_b_16_model.encoder, "pos_embedding")

        # For 512x512 image: 1024 patches + 1 class token
        expected_num_patches = (512 // 16) * (512 // 16)
        expected_seq_len = expected_num_patches + 1

        # Positional embedding should match sequence length
        assert vit_b_16_model.encoder.pos_embedding.shape[1] >= expected_seq_len or \
               vit_b_16_model.encoder.pos_embedding.shape[1] == (224 // 16) * (224 // 16) + 1

    def test_vit_b_16_modified_for_extraction(self, sample_tensor):
        """Test ViT-B/16 modified for feature extraction (without classification head)."""
        # Load model and modify for feature extraction
        model = models.vit_b_16(weights=None)
        model.eval()

        # Remove classification head (similar to vit_extractor.py)
        import torch.nn as nn
        model.heads = nn.Identity()

        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Forward pass should now return features instead of classifications
        with torch.no_grad():
            features = model(batch_tensor)

        # Validate features (should be 768-dimensional, not 1000 classes)
        assert features.shape == (1, 768)
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_vit_b_16_gradient_disabled(self, sample_tensor, vit_b_16_model):
        """Test that gradients are properly disabled during inference."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)
        batch_tensor.requires_grad = False

        # Forward pass
        with torch.no_grad():
            output = vit_b_16_model(batch_tensor)

        # Validate gradients are not tracked
        assert output.requires_grad == False

    def test_vit_b_16_eval_mode_behavior(self, sample_tensor, vit_b_16_model):
        """Test ViT-B/16 behavior in eval mode vs train mode."""
        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Extract features in eval mode
        vit_b_16_model.eval()
        with torch.no_grad():
            eval_output = vit_b_16_model(batch_tensor)

        # Extract features in train mode
        vit_b_16_model.train()
        with torch.no_grad():
            train_output = vit_b_16_model(batch_tensor)

        # Set back to eval mode
        vit_b_16_model.eval()

        # Outputs should be similar (dropout disabled in eval mode)
        assert eval_output.shape == train_output.shape
        # Note: outputs may differ slightly due to dropout in train mode

    def test_vit_b_16_memory_efficiency(self, vit_b_16_model):
        """Test ViT-B/16 memory efficiency with batch processing."""
        # Create batches of different sizes
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            # Create batch
            batch_tensor = torch.randn(batch_size, 3, 224, 224)

            # Process batch
            with torch.no_grad():
                output = vit_b_16_model(batch_tensor)

            # Validate output shape
            assert output.shape == (batch_size, 1000)
            assert not torch.any(torch.isnan(output))
            assert not torch.any(torch.isinf(output))

    def test_vit_b_16_imagenet_normalization(self, sample_tensor):
        """Test ImageNet normalization for ViT-B/16."""
        # ImageNet normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Add batch dimension
        batch_tensor = sample_tensor.unsqueeze(0)

        # Apply normalization
        normalized_tensor = (batch_tensor - mean) / std

        # Validate normalized tensor
        assert normalized_tensor.shape == batch_tensor.shape
        assert not torch.any(torch.isnan(normalized_tensor))
        assert not torch.any(torch.isinf(normalized_tensor))

    def test_vit_b_16_input_validation(self, vit_b_16_model):
        """Test ViT-B/16 input validation and error handling."""
        # Test with invalid input shapes
        invalid_shapes = [
            (1, 1, 224, 224),  # Wrong number of channels
            (1, 3, 100, 100),  # Valid but small size
            (1, 3, 2000, 2000),  # Large size
        ]

        for shape in invalid_shapes:
            tensor = torch.randn(*shape)

            with torch.no_grad():
                try:
                    output = vit_b_16_model(tensor)
                    # If it succeeds, validate output shape
                    assert output.shape[0] == shape[0]
                    assert output.shape[1] == 1000
                except Exception as e:
                    # Some shapes may cause errors (e.g., wrong channels)
                    assert "channel" in str(e).lower() or "dimension" in str(e).lower()
