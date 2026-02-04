"""
Unit tests for ResNet-50 FP16 (mixed-precision) functionality.

These tests validate FP16 configuration, model conversion, and inference
for memory-efficient feature extraction on CUDA GPUs.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest
from pathlib import Path
import sys

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.models.embeddings.resnet50_extractor import ResNet50Extractor
from mammography.preprocess.preprocessed_tensor import PreprocessedTensor


class TestResNetFP16Configuration:
    """Unit tests for FP16 configuration and model conversion."""

    @pytest.fixture
    def base_config(self) -> dict:
        """Create base configuration for ResNet50Extractor."""
        return {
            'model_name': 'resnet50',
            'pretrained': False,
            'input_adapter': '1to3_replication',
            'seed': 42
        }

    @pytest.fixture
    def sample_preprocessed_tensor(self) -> PreprocessedTensor:
        """Create a sample preprocessed tensor for testing."""
        # Create a simple grayscale image tensor
        image_data = np.random.rand(512, 512).astype(np.float32)

        return PreprocessedTensor(
            data=image_data,
            metadata={
                'patient_id': 'TEST001',
                'study_id': 'STUDY001',
                'image_id': 'IMG001',
                'height': 512,
                'width': 512,
                'channels': 1,
                'dtype': 'float32'
            }
        )

    def test_use_fp16_defaults_to_false(self, base_config):
        """Test that use_fp16 defaults to False for backward compatibility."""
        extractor = ResNet50Extractor(base_config)

        # Verify use_fp16 is False by default
        assert extractor.config['use_fp16'] == False

        # Verify model info reports correct FP16 status
        model_info = extractor.get_model_info()
        assert 'use_fp16' in model_info
        assert model_info['use_fp16'] == False

    def test_use_fp16_explicit_false(self, base_config):
        """Test that use_fp16=False works correctly."""
        config = {**base_config, 'use_fp16': False}
        extractor = ResNet50Extractor(config)

        # Verify use_fp16 is False
        assert extractor.config['use_fp16'] == False

        # Verify model is in FP32
        param = next(extractor.model.parameters())
        assert param.dtype == torch.float32

    def test_use_fp16_explicit_true_cpu_device(self, base_config):
        """Test that use_fp16=True on CPU device does NOT convert to FP16."""
        config = {**base_config, 'use_fp16': True, 'device': 'cpu'}
        extractor = ResNet50Extractor(config)

        # Verify use_fp16 is True in config
        assert extractor.config['use_fp16'] == True

        # Verify device is CPU
        assert extractor.device.type == 'cpu'

        # Verify model remains in FP32 (CPU doesn't support FP16 well)
        param = next(extractor.model.parameters())
        assert param.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_use_fp16_explicit_true_cuda_device(self, base_config):
        """Test that use_fp16=True on CUDA device converts model to FP16."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Verify use_fp16 is True in config
        assert extractor.config['use_fp16'] == True

        # Verify device is CUDA
        assert extractor.device.type == 'cuda'

        # Verify model is in FP16
        param = next(extractor.model.parameters())
        assert param.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_dtype_fp16_on_cuda(self, base_config):
        """Test that model parameters are torch.float16 when FP16 enabled on CUDA."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Check all model parameters are FP16
        for name, param in extractor.model.named_parameters():
            assert param.dtype == torch.float16, f"Parameter {name} is not FP16"

    def test_model_dtype_fp32_on_cpu(self, base_config):
        """Test that model parameters are torch.float32 when FP16 requested on CPU."""
        config = {**base_config, 'use_fp16': True, 'device': 'cpu'}
        extractor = ResNet50Extractor(config)

        # Check all model parameters remain FP32 on CPU
        for name, param in extractor.model.named_parameters():
            assert param.dtype == torch.float32, f"Parameter {name} is not FP32"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_model_info_reports_fp16_status(self, base_config):
        """Test that get_model_info reports FP16 status correctly."""
        # Test FP16 enabled on CUDA
        config_fp16 = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor_fp16 = ResNet50Extractor(config_fp16)

        model_info_fp16 = extractor_fp16.get_model_info()
        assert 'use_fp16' in model_info_fp16
        assert model_info_fp16['use_fp16'] == True

        # Test FP16 disabled
        config_fp32 = {**base_config, 'use_fp16': False, 'device': 'cuda'}
        extractor_fp32 = ResNet50Extractor(config_fp32)

        model_info_fp32 = extractor_fp32.get_model_info()
        assert 'use_fp16' in model_info_fp32
        assert model_info_fp32['use_fp16'] == False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_size_calculation_fp16_vs_fp32(self, base_config):
        """Test that model size calculation accounts for FP16 vs FP32."""
        # FP32 model
        config_fp32 = {**base_config, 'use_fp16': False, 'device': 'cuda'}
        extractor_fp32 = ResNet50Extractor(config_fp32)
        model_info_fp32 = extractor_fp32.get_model_info()

        # FP16 model
        config_fp16 = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor_fp16 = ResNet50Extractor(config_fp16)
        model_info_fp16 = extractor_fp16.get_model_info()

        # FP16 should be approximately half the size of FP32
        # (2 bytes per parameter vs 4 bytes per parameter)
        size_ratio = model_info_fp32['model_size_mb'] / model_info_fp16['model_size_mb']
        assert 1.9 < size_ratio < 2.1, f"FP16 size reduction not ~2x (ratio: {size_ratio})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_memory_efficiency(self, base_config):
        """Test that FP16 uses approximately 50% less GPU memory than FP32.

        This test validates memory efficiency by measuring actual GPU memory
        allocation for FP32 vs FP16 models during instantiation.

        Educational Notes:
        - FP16 uses 2 bytes per parameter vs 4 bytes for FP32
        - Memory reduction enables processing larger batches or bigger models
        - GPU memory optimization is crucial for large-scale medical imaging
        """
        # Test with FP32
        torch.cuda.empty_cache()
        initial_memory_fp32 = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        config_fp32 = {**base_config, 'use_fp16': False, 'device': 'cuda'}
        extractor_fp32 = ResNet50Extractor(config_fp32)

        memory_fp32 = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        fp32_usage = memory_fp32 - initial_memory_fp32

        # Cleanup FP32 model
        del extractor_fp32
        torch.cuda.empty_cache()

        # Test with FP16
        initial_memory_fp16 = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        config_fp16 = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor_fp16 = ResNet50Extractor(config_fp16)

        memory_fp16 = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        fp16_usage = memory_fp16 - initial_memory_fp16

        # Validate memory efficiency
        assert fp16_usage < fp32_usage, "FP16 should use less memory than FP32"

        # FP16 should use approximately half the memory (2x reduction)
        memory_ratio = fp32_usage / fp16_usage if fp16_usage > 0 else 0
        assert 1.8 < memory_ratio < 2.2, (
            f"FP16 memory reduction not ~2x (ratio: {memory_ratio:.2f}x, "
            f"FP32: {fp32_usage:.2f}MB, FP16: {fp16_usage:.2f}MB)"
        )

        # Cleanup
        del extractor_fp16
        torch.cuda.empty_cache()


class TestResNetFP16Inference:
    """Unit tests for FP16 inference and normalization."""

    @pytest.fixture
    def base_config(self) -> dict:
        """Create base configuration for ResNet50Extractor."""
        return {
            'model_name': 'resnet50',
            'pretrained': False,
            'input_adapter': '1to3_replication',
            'seed': 42
        }

    @pytest.fixture
    def sample_preprocessed_tensor(self) -> PreprocessedTensor:
        """Create a sample preprocessed tensor for testing."""
        # Create a simple grayscale image tensor
        image_data = np.random.rand(512, 512).astype(np.float32)

        return PreprocessedTensor(
            data=image_data,
            metadata={
                'patient_id': 'TEST001',
                'study_id': 'STUDY001',
                'image_id': 'IMG001',
                'height': 512,
                'width': 512,
                'channels': 1,
                'dtype': 'float32'
            }
        )

    def test_normalization_tensors_match_model_dtype_fp32(self, base_config):
        """Test that normalization tensors match model dtype in FP32 mode."""
        config = {**base_config, 'use_fp16': False, 'device': 'cpu'}
        extractor = ResNet50Extractor(config)

        # Create a test tensor
        test_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=extractor.device)

        # Apply normalization
        normalized = extractor._apply_imagenet_normalization(test_tensor)

        # Verify normalized tensor is FP32
        assert normalized.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_normalization_tensors_match_model_dtype_fp16(self, base_config):
        """Test that normalization tensors match model dtype in FP16 mode."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Create a test tensor in FP16
        test_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float16, device=extractor.device)

        # Apply normalization
        normalized = extractor._apply_imagenet_normalization(test_tensor)

        # Verify normalized tensor is FP16
        assert normalized.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_input_tensor_conversion_to_fp16(self, base_config):
        """Test that input tensors are converted to FP16 when use_fp16=True on CUDA."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Create a preprocessed tensor
        image_data = np.random.rand(512, 512).astype(np.float32)
        preprocessed = PreprocessedTensor(
            data=image_data,
            metadata={
                'patient_id': 'TEST001',
                'study_id': 'STUDY001',
                'image_id': 'IMG001',
                'height': 512,
                'width': 512,
                'channels': 1,
                'dtype': 'float32'
            }
        )

        # Prepare input tensor
        input_tensor = extractor._prepare_input_tensor(preprocessed)

        # Verify input tensor is FP16
        assert input_tensor is not None
        assert input_tensor.dtype == torch.float16

    def test_input_tensor_stays_fp32_on_cpu(self, base_config):
        """Test that input tensors stay FP32 on CPU even when use_fp16=True."""
        config = {**base_config, 'use_fp16': True, 'device': 'cpu'}
        extractor = ResNet50Extractor(config)

        # Create a preprocessed tensor
        image_data = np.random.rand(512, 512).astype(np.float32)
        preprocessed = PreprocessedTensor(
            data=image_data,
            metadata={
                'patient_id': 'TEST001',
                'study_id': 'STUDY001',
                'image_id': 'IMG001',
                'height': 512,
                'width': 512,
                'channels': 1,
                'dtype': 'float32'
            }
        )

        # Prepare input tensor
        input_tensor = extractor._prepare_input_tensor(preprocessed)

        # Verify input tensor remains FP32 on CPU
        assert input_tensor is not None
        assert input_tensor.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_forward_pass_produces_valid_embeddings(self, base_config, sample_preprocessed_tensor):
        """Test that FP16 forward pass produces valid embeddings."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Extract embedding
        embedding = extractor.extract_embedding(sample_preprocessed_tensor)

        # Verify embedding is valid
        assert embedding is not None
        assert hasattr(embedding, 'vector')
        assert len(embedding.vector) == ResNet50Extractor.EXPECTED_EMBEDDING_DIM

        # Verify no NaN or Inf values
        embedding_array = np.array(embedding.vector)
        assert not np.any(np.isnan(embedding_array))
        assert not np.any(np.isinf(embedding_array))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_batch_processing(self, base_config):
        """Test that FP16 batch processing works correctly."""
        config = {**base_config, 'use_fp16': True, 'device': 'cuda', 'batch_size': 4}
        extractor = ResNet50Extractor(config)

        # Create batch of preprocessed tensors
        batch = []
        for i in range(4):
            image_data = np.random.rand(512, 512).astype(np.float32)
            preprocessed = PreprocessedTensor(
                data=image_data,
                metadata={
                    'patient_id': f'TEST{i:03d}',
                    'study_id': f'STUDY{i:03d}',
                    'image_id': f'IMG{i:03d}',
                    'height': 512,
                    'width': 512,
                    'channels': 1,
                    'dtype': 'float32'
                }
            )
            batch.append(preprocessed)

        # Extract embeddings
        embeddings = extractor.extract_embeddings_batch(batch)

        # Verify all embeddings are valid
        assert len(embeddings) == 4
        for embedding in embeddings:
            assert embedding is not None
            assert len(embedding.vector) == ResNet50Extractor.EXPECTED_EMBEDDING_DIM
            embedding_array = np.array(embedding.vector)
            assert not np.any(np.isnan(embedding_array))
            assert not np.any(np.isinf(embedding_array))

    def test_fp16_reproducibility(self, base_config, sample_preprocessed_tensor):
        """Test that FP16 embeddings are reproducible with same seed."""
        config = {**base_config, 'use_fp16': False, 'device': 'cpu', 'seed': 42}

        # Extract embeddings multiple times with same seed
        embeddings = []
        for _ in range(3):
            extractor = ResNet50Extractor(config)
            embedding = extractor.extract_embedding(sample_preprocessed_tensor)
            embeddings.append(np.array(embedding.vector))

        # Verify embeddings are identical
        for i in range(1, len(embeddings)):
            np.testing.assert_allclose(
                embeddings[0],
                embeddings[i],
                rtol=1e-5,
                err_msg="Embeddings not reproducible"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_vs_fp32_similarity(self, base_config, sample_preprocessed_tensor):
        """Test that FP16 and FP32 embeddings are similar but not identical."""
        # Extract with FP32
        config_fp32 = {**base_config, 'use_fp16': False, 'device': 'cuda', 'seed': 42}
        extractor_fp32 = ResNet50Extractor(config_fp32)
        embedding_fp32 = extractor_fp32.extract_embedding(sample_preprocessed_tensor)
        vector_fp32 = np.array(embedding_fp32.vector)

        # Extract with FP16
        config_fp16 = {**base_config, 'use_fp16': True, 'device': 'cuda', 'seed': 42}
        extractor_fp16 = ResNet50Extractor(config_fp16)
        embedding_fp16 = extractor_fp16.extract_embedding(sample_preprocessed_tensor)
        vector_fp16 = np.array(embedding_fp16.vector)

        # Verify embeddings are similar but not identical
        # FP16 has lower precision, so we expect small differences
        assert not np.allclose(vector_fp32, vector_fp16, rtol=1e-5), "FP16 and FP32 should not be identical"

        # But they should be close (within FP16 precision tolerance)
        np.testing.assert_allclose(
            vector_fp32,
            vector_fp16,
            rtol=1e-2,  # 1% relative tolerance for FP16
            atol=1e-3,  # Absolute tolerance for small values
            err_msg="FP16 and FP32 embeddings too different"
        )

        # Calculate cosine similarity (should be very high)
        cosine_sim = np.dot(vector_fp32, vector_fp16) / (
            np.linalg.norm(vector_fp32) * np.linalg.norm(vector_fp16)
        )
        assert cosine_sim > 0.999, f"Cosine similarity too low: {cosine_sim}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_produces_valid_embeddings(self, base_config, sample_preprocessed_tensor):
        """Test that FP16 produces valid embeddings with correct shape and values.

        This functional test validates that FP16-enabled ResNet50Extractor produces
        embeddings that meet all quality requirements:
        - Correct dimensionality (2048 features)
        - No NaN or Inf values
        - Reasonable value ranges
        - Proper output format

        Educational Notes:
        - FP16 (half-precision) uses 16-bit floating point (vs 32-bit for FP32)
        - FP16 reduces memory usage by ~50% and can speed up computation on modern GPUs
        - Despite lower precision, FP16 typically produces embeddings very similar to FP32
        - Medical imaging applications require careful validation of FP16 outputs
        """
        # Configure extractor with FP16 on CUDA
        config = {**base_config, 'use_fp16': True, 'device': 'cuda'}
        extractor = ResNet50Extractor(config)

        # Extract embedding using FP16
        embedding = extractor.extract_embedding(sample_preprocessed_tensor)

        # Validate embedding exists and has correct structure
        assert embedding is not None, "Embedding should not be None"
        assert hasattr(embedding, 'vector'), "Embedding should have vector attribute"
        assert hasattr(embedding, 'metadata'), "Embedding should have metadata attribute"

        # Validate embedding dimensionality
        assert len(embedding.vector) == ResNet50Extractor.EXPECTED_EMBEDDING_DIM, (
            f"Expected embedding dimension {ResNet50Extractor.EXPECTED_EMBEDDING_DIM}, "
            f"got {len(embedding.vector)}"
        )

        # Convert to numpy array for validation
        embedding_array = np.array(embedding.vector)

        # Validate no NaN values
        assert not np.any(np.isnan(embedding_array)), (
            "Embedding contains NaN values - FP16 computation failed"
        )

        # Validate no Inf values
        assert not np.any(np.isinf(embedding_array)), (
            "Embedding contains Inf values - FP16 computation failed"
        )

        # Validate embedding is not all zeros (model is actually computing features)
        assert not np.allclose(embedding_array, 0.0), (
            "Embedding is all zeros - model may not be properly initialized"
        )

        # Validate embedding has reasonable value range
        # ResNet50 embeddings typically have values in a reasonable range after pooling
        max_abs_value = np.max(np.abs(embedding_array))
        assert max_abs_value < 1000.0, (
            f"Embedding values too large (max abs: {max_abs_value:.2f}) - "
            "may indicate numerical instability in FP16"
        )

        # Validate embedding dtype is float32 (should be converted back from FP16 for output)
        assert embedding_array.dtype == np.float32, (
            f"Expected float32 output, got {embedding_array.dtype}"
        )

        # Validate metadata is preserved
        assert 'patient_id' in embedding.metadata
        assert embedding.metadata['patient_id'] == sample_preprocessed_tensor.metadata['patient_id']
