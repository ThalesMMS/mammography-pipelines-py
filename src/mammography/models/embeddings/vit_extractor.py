"""
Vision Transformer (ViT) embedding extractor for mammography feature extraction.

This module provides Vision Transformer-based feature extraction capabilities for
mammography images, including input adapters for grayscale to RGB conversion
and batch processing for efficient embedding extraction.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Vision Transformers provide attention-based visual feature representations
- Feature extraction from the final hidden state captures high-level semantic features
- Input adapters handle grayscale to RGB conversion for pretrained models
- Batch processing improves efficiency for large datasets

Author: Research Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import logging
import numpy as np
from pathlib import Path
import time

from ...preprocess.preprocessed_tensor import PreprocessedTensor
from .embedding_vector import EmbeddingVector, create_embedding_vector_from_extraction, batch_create_embedding_vectors

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ViTExtractor:
    """
    Vision Transformer (ViT) feature extractor for mammography images.

    This class provides methods for extracting 768-dimensional feature vectors
    from mammography images using a pre-trained Vision Transformer model. It includes
    input adapters for handling grayscale images and batch processing capabilities.

    Educational Notes:
    - ViT-B/16 provides 768-dimensional features from the final hidden state
    - Pre-trained ImageNet weights provide strong visual representations
    - Input adapters handle grayscale to RGB conversion
    - Batch processing improves GPU utilization and efficiency
    - FP16 mixed-precision reduces GPU memory usage by ~50% on CUDA devices
    - FP16 mode provides faster inference with negligible quality impact

    Configuration Example:
        Basic configuration (FP32):
            config = {
                'model_name': 'vit_b_16',
                'pretrained': True,
                'input_adapter': '1to3_replication',
                'batch_size': 8
            }

        Memory-efficient configuration (FP16 on CUDA):
            config = {
                'model_name': 'vit_b_16',
                'pretrained': True,
                'input_adapter': '1to3_replication',
                'batch_size': 16,  # Can use larger batches with FP16
                'use_fp16': True,  # Enables FP16 on CUDA devices only
                'device': 'cuda'
            }

        Note: FP16 mode only activates on CUDA GPUs. On CPU or MPS devices,
        the model remains in FP32 precision even if use_fp16=True is set.

    Attributes:
        model: Vision Transformer model for feature extraction
        config: Configuration dictionary for extraction parameters
        device: Computing device (CPU/GPU)
        input_adapter: Method for grayscale to RGB conversion
    """

    # Expected embedding dimension from ViT-B/16 final hidden state
    EXPECTED_EMBEDDING_DIM = 768

    # Supported input adapters
    SUPPORTED_INPUT_ADAPTERS = [
        "1to3_replication",
        "conv1_adapted"
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Vision Transformer extractor with configuration.

        Args:
            config: Configuration dictionary for feature extraction

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for feature extraction.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if 'seed' in self.config:
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])

        # Initialize device
        self.device = self._setup_device()

        # Initialize model
        self.model = self._initialize_model()

        # Set model to evaluation mode
        self.model.eval()

        logger.info(f"Initialized ViTExtractor with device: {self.device}")

    def extract_embedding(self, preprocessed_tensor: PreprocessedTensor) -> Optional[EmbeddingVector]:
        """
        Extract embedding from a single preprocessed tensor.

        Educational Note: This method demonstrates the complete feature
        extraction process from preprocessed tensor to embedding vector.

        Args:
            preprocessed_tensor: PreprocessedTensor instance

        Returns:
            EmbeddingVector: Extracted embedding if successful, None otherwise

        Raises:
            ValueError: If preprocessed tensor is invalid
        """
        start_time = time.time()

        try:
            # Validate input tensor
            if not self._validate_tensor(preprocessed_tensor):
                return None

            # Prepare tensor for model input
            input_tensor = self._prepare_input_tensor(preprocessed_tensor)
            if input_tensor is None:
                return None

            # FP16 Inference: Extract features using automatic mixed precision
            # - torch.no_grad() disables gradient computation for inference
            # - torch.autocast enables automatic mixed precision when use_fp16=True
            # - Autocast automatically casts operations to FP16 where beneficial
            # - Maintains FP32 for operations that need higher precision (e.g., loss)
            # - Only activates when enabled=True and device supports mixed precision
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.config.get('use_fp16', False)):
                    features = self.model(input_tensor)

            # Calculate extraction time
            extraction_time = time.time() - start_time

            # Create EmbeddingVector instance
            embedding_vector = create_embedding_vector_from_extraction(
                image_id=preprocessed_tensor.image_id,
                embedding=features.squeeze(0),  # Remove batch dimension
                model_config=self.config,
                input_adapter=self.config['input_adapter'],
                extraction_time=extraction_time,
                device_used=str(self.device)
            )

            logger.info(f"Successfully extracted embedding for {preprocessed_tensor.image_id} in {extraction_time:.3f}s")
            return embedding_vector

        except Exception as e:
            logger.error(f"Error extracting embedding for {preprocessed_tensor.image_id}: {str(e)}")
            return None

    def extract_embeddings_batch(self, preprocessed_tensors: List[PreprocessedTensor]) -> List[Optional[EmbeddingVector]]:
        """
        Extract embeddings from a batch of preprocessed tensors.

        Educational Note: Batch processing significantly improves efficiency
        when extracting embeddings from large numbers of images.

        Args:
            preprocessed_tensors: List of PreprocessedTensor instances

        Returns:
            List[Optional[EmbeddingVector]]: List of extracted embeddings
        """
        if not preprocessed_tensors:
            return []

        # Filter valid tensors
        valid_tensors = [tensor for tensor in preprocessed_tensors if self._validate_tensor(tensor)]

        if not valid_tensors:
            logger.warning("No valid tensors found in batch")
            return [None] * len(preprocessed_tensors)

        # Process in batches
        batch_size = self.config.get('batch_size', 8)
        all_embeddings = []

        for i in range(0, len(valid_tensors), batch_size):
            batch_tensors = valid_tensors[i:i + batch_size]
            batch_embeddings = self._extract_batch_embeddings(batch_tensors)
            all_embeddings.extend(batch_embeddings)

        # Pad with None values for invalid tensors
        result_embeddings = []
        valid_index = 0

        for tensor in preprocessed_tensors:
            if self._validate_tensor(tensor):
                result_embeddings.append(all_embeddings[valid_index])
                valid_index += 1
            else:
                result_embeddings.append(None)

        successful_count = sum(1 for emb in result_embeddings if emb is not None)
        logger.info(f"Successfully extracted {successful_count}/{len(preprocessed_tensors)} embeddings")

        return result_embeddings

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extraction configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['model_name', 'pretrained', 'input_adapter']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required configuration parameter: {param}")

        # Validate model name
        model_name = config['model_name']
        if model_name not in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Validate pretrained flag
        if not isinstance(config['pretrained'], bool):
            raise ValueError("pretrained must be a boolean")

        # Validate input adapter
        input_adapter = config['input_adapter']
        if input_adapter not in self.SUPPORTED_INPUT_ADAPTERS:
            raise ValueError(f"Unsupported input adapter: {input_adapter}")

        # Set default values for optional parameters
        config.setdefault('feature_layer', 'final_hidden')
        config.setdefault('batch_size', 8)
        config.setdefault('normalize_embeddings', False)
        config.setdefault('normalization_method', 'l2')
        config.setdefault('seed', 42)
        # FP16 Configuration: Mixed-precision uses 16-bit floats instead of 32-bit
        # Benefits: ~50% memory reduction, faster inference on modern GPUs
        # Only activated on CUDA devices; defaults to FP32 on CPU/MPS
        config.setdefault('use_fp16', False)

        return config

    def _setup_device(self) -> torch.device:
        """
        Setup computing device for feature extraction.

        Educational Note: Device selection affects processing speed and
        memory usage. GPU acceleration significantly improves performance.

        Returns:
            torch.device: Selected computing device
        """
        device_config = self.config.get('device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using MPS device (Apple Silicon)")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
            logger.info(f"Using specified device: {device}")

        return device

    def _initialize_model(self) -> nn.Module:
        """
        Initialize Vision Transformer model for feature extraction.

        Educational Note: This method loads a pre-trained ViT model
        and modifies it to extract features from the final hidden state.

        Returns:
            nn.Module: Modified Vision Transformer model
        """
        model_name = self.config['model_name']
        pretrained = self.config['pretrained']

        # Load pre-trained model
        if model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            model = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_l_32':
            model = models.vit_l_32(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify model for feature extraction
        model = self._modify_model_for_extraction(model)

        # Move model to device
        model = model.to(self.device)

        # FP16 Model Conversion: Convert model weights from FP32 to FP16
        # - model.half() converts all model parameters to float16 dtype
        # - Only enabled on CUDA devices (not supported efficiently on CPU/MPS)
        # - Reduces model memory footprint by 50% (2 bytes vs 4 bytes per param)
        # - Works with torch.autocast for automatic mixed-precision inference
        if self.config.get('use_fp16', False) and self.device.type == 'cuda':
            model = model.half()
            logger.info(f"Converted model to FP16 precision for CUDA device")

        logger.info(f"Initialized {model_name} model with pretrained={pretrained}")
        return model

    def _modify_model_for_extraction(self, model: nn.Module) -> nn.Module:
        """
        Modify Vision Transformer model for feature extraction.

        Educational Note: This method removes the final classification head
        and returns features from the CLS token for embedding extraction.

        Args:
            model: Original Vision Transformer model

        Returns:
            nn.Module: Modified model for feature extraction
        """
        # Remove the final classification head (heads.head)
        # ViT uses a classification head on top of the encoder output
        # We want the encoder output (hidden state from CLS token)
        model.heads = nn.Identity()

        return model

    def _validate_tensor(self, preprocessed_tensor: PreprocessedTensor) -> bool:
        """
        Validate preprocessed tensor for feature extraction.

        Educational Note: Tensor validation ensures the input is in the
        correct format for Vision Transformer feature extraction.

        Args:
            preprocessed_tensor: PreprocessedTensor to validate

        Returns:
            bool: True if tensor is valid, False otherwise
        """
        if preprocessed_tensor is None:
            return False

        # Check tensor shape (should be 3, H, W)
        if preprocessed_tensor.tensor_data.ndim != 3:
            logger.error(f"Invalid tensor dimensions: {preprocessed_tensor.tensor_data.shape}")
            return False

        if preprocessed_tensor.tensor_data.shape[0] != 3:
            logger.error(f"Invalid number of channels: {preprocessed_tensor.tensor_data.shape[0]}")
            return False

        # Check for NaN or infinite values
        if torch.any(torch.isnan(preprocessed_tensor.tensor_data)):
            logger.error("Tensor contains NaN values")
            return False

        if torch.any(torch.isinf(preprocessed_tensor.tensor_data)):
            logger.error("Tensor contains infinite values")
            return False

        return True

    def _prepare_input_tensor(self, preprocessed_tensor: PreprocessedTensor) -> Optional[torch.Tensor]:
        """
        Prepare tensor for model input.

        Educational Note: This method prepares the tensor for Vision Transformer
        input including normalization, device transfer, and dtype conversion.

        Args:
            preprocessed_tensor: PreprocessedTensor instance

        Returns:
            torch.Tensor: Prepared input tensor, None if failed
        """
        try:
            # Get tensor data
            tensor_data = preprocessed_tensor.tensor_data

            # Add batch dimension
            input_tensor = tensor_data.unsqueeze(0)  # Shape: (1, 3, H, W)

            # Apply ImageNet normalization if using pretrained weights
            if self.config.get('pretrained', True):
                input_tensor = self._apply_imagenet_normalization(input_tensor)

            # Move to device
            input_tensor = input_tensor.to(self.device)

            # FP16 Input Conversion: Convert input tensor to match model precision
            # - When model is in FP16, inputs must also be FP16 for efficient computation
            # - tensor.half() converts from float32 to float16 dtype
            # - Ensures entire forward pass runs in FP16, maximizing GPU throughput
            # - Prevents automatic dtype promotion which would negate FP16 benefits
            # - Combined with model.half() and torch.autocast for full mixed-precision
            if self.config.get('use_fp16', False) and self.device.type == 'cuda':
                input_tensor = input_tensor.half()

            return input_tensor

        except Exception as e:
            logger.error(f"Error preparing input tensor: {str(e)}")
            return None

    def _apply_imagenet_normalization(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to input tensor.

        Educational Note: ImageNet normalization is required when using
        pre-trained weights to match the training distribution.

        Args:
            tensor: Input tensor

        Returns:
            torch.Tensor: Normalized tensor
        """
        # ImageNet normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # FP16 Dtype Matching: Convert normalization params to match input tensor
        # - Critical for FP16: parameters must match tensor dtype to avoid conversion
        # - If tensor is FP16, mean/std must also be FP16 for efficient computation
        # - Mismatched dtypes would trigger automatic promotion to FP32, losing FP16 benefits
        # - .to() ensures both device (CUDA/CPU) and dtype (FP16/FP32) match
        mean = mean.to(device=tensor.device, dtype=tensor.dtype)
        std = std.to(device=tensor.device, dtype=tensor.dtype)

        # Apply normalization
        normalized_tensor = (tensor - mean) / std

        return normalized_tensor

    def _extract_batch_embeddings(self, preprocessed_tensors: List[PreprocessedTensor]) -> List[Optional[EmbeddingVector]]:
        """
        Extract embeddings from a batch of tensors.

        Educational Note: Batch processing improves GPU utilization
        and overall processing efficiency.

        Args:
            preprocessed_tensors: List of PreprocessedTensor instances

        Returns:
            List[Optional[EmbeddingVector]]: List of extracted embeddings
        """
        start_time = time.time()

        try:
            # Prepare batch input
            batch_tensors = []
            valid_indices = []

            for i, tensor in enumerate(preprocessed_tensors):
                if self._validate_tensor(tensor):
                    input_tensor = self._prepare_input_tensor(tensor)
                    if input_tensor is not None:
                        batch_tensors.append(input_tensor)
                        valid_indices.append(i)

            if not batch_tensors:
                return [None] * len(preprocessed_tensors)

            # Stack tensors into batch
            batch_input = torch.cat(batch_tensors, dim=0)

            # FP16 Batch Inference: Process entire batch with mixed precision
            # - Batch processing with FP16 provides maximum GPU efficiency
            # - FP16 allows ~2x larger batch sizes due to 50% memory reduction
            # - torch.autocast handles automatic precision casting for all operations
            # - Larger batches improve GPU utilization and reduce per-image inference time
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.config.get('use_fp16', False)):
                    batch_features = self.model(batch_input)

            # Calculate extraction time per image
            total_time = time.time() - start_time
            extraction_time_per_image = total_time / len(batch_tensors)

            # Create embedding vectors
            embedding_vectors = []
            for i, tensor in enumerate(preprocessed_tensors):
                if i in valid_indices:
                    # Find corresponding features
                    batch_idx = valid_indices.index(i)
                    features = batch_features[batch_idx]

                    # Create EmbeddingVector
                    embedding_vector = create_embedding_vector_from_extraction(
                        image_id=tensor.image_id,
                        embedding=features,
                        model_config=self.config,
                        input_adapter=self.config['input_adapter'],
                        extraction_time=extraction_time_per_image,
                        device_used=str(self.device)
                    )
                    embedding_vectors.append(embedding_vector)
                else:
                    embedding_vectors.append(None)

            return embedding_vectors

        except Exception as e:
            logger.error(f"Error in batch extraction: {str(e)}")
            return [None] * len(preprocessed_tensors)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Educational Note: This information helps understand the model
        configuration and capabilities.

        Returns:
            Dict[str, Any]: Model information dictionary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # FP16 Memory Calculation: Compute actual model memory usage
        # - FP32 (float32): 4 bytes per parameter (standard precision)
        # - FP16 (float16): 2 bytes per parameter (half precision)
        # - FP16 reduces model size by 50% compared to FP32
        # - Example: ViT-B/16 (~86M params): 344MB (FP32) vs 172MB (FP16)
        # - Only reflects actual FP16 usage when use_fp16=True AND device=cuda
        use_fp16 = self.config.get('use_fp16', False)
        bytes_per_param = 2 if (use_fp16 and self.device.type == 'cuda') else 4
        model_size_mb = total_params * bytes_per_param / (1024 * 1024)

        return {
            'model_name': self.config['model_name'],
            'pretrained': self.config['pretrained'],
            'feature_layer': self.config['feature_layer'],
            'embedding_dimension': self.EXPECTED_EMBEDDING_DIM,
            'input_adapter': self.config['input_adapter'],
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_fp16': use_fp16,
            'model_size_mb': model_size_mb
        }


def create_vit_extractor(config: Dict[str, Any]) -> ViTExtractor:
    """
    Factory function to create a ViTExtractor instance.

    Educational Note: This factory function provides a convenient way
    to create ViTExtractor instances with validated configurations.

    Args:
        config: Configuration dictionary for feature extraction

    Returns:
        ViTExtractor: Configured ViTExtractor instance
    """
    return ViTExtractor(config)


def extract_single_embedding(preprocessed_tensor: PreprocessedTensor,
                           config: Dict[str, Any]) -> Optional[EmbeddingVector]:
    """
    Convenience function to extract embedding from a single tensor.

    Educational Note: This function provides a simple interface for
    extracting individual embeddings without creating an ViTExtractor instance.

    Args:
        preprocessed_tensor: PreprocessedTensor instance
        config: Configuration dictionary for feature extraction

    Returns:
        EmbeddingVector: Extracted embedding if successful, None otherwise
    """
    extractor = create_vit_extractor(config)
    return extractor.extract_embedding(preprocessed_tensor)


def extract_batch_embeddings(preprocessed_tensors: List[PreprocessedTensor],
                           config: Dict[str, Any]) -> List[Optional[EmbeddingVector]]:
    """
    Convenience function to extract embeddings from a batch of tensors.

    Educational Note: This function provides a simple interface for
    batch embedding extraction without creating an ViTExtractor instance.

    Args:
        preprocessed_tensors: List of PreprocessedTensor instances
        config: Configuration dictionary for feature extraction

    Returns:
        List[Optional[EmbeddingVector]]: List of extracted embeddings
    """
    extractor = create_vit_extractor(config)
    return extractor.extract_embeddings_batch(preprocessed_tensors)
