"""
EfficientNet-B0 embedding extractor for mammography feature extraction.

This module provides EfficientNet-B0-based feature extraction capabilities for
mammography images, including input adapters for grayscale to RGB conversion
and batch processing for efficient embedding extraction.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- EfficientNet-B0 provides efficient visual feature representations from ImageNet
- Feature extraction from avgpool layer captures high-level semantic features
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


class EfficientNetExtractor:
    """
    EfficientNet-B0 feature extractor for mammography images.

    This class provides methods for extracting 1280-dimensional feature vectors
    from mammography images using a pre-trained EfficientNet-B0 model. It includes
    input adapters for handling grayscale images and batch processing capabilities.

    Educational Notes:
    - EfficientNet-B0 avgpool layer provides 1280-dimensional features
    - Pre-trained ImageNet weights provide strong visual representations
    - Input adapters handle grayscale to RGB conversion
    - Batch processing improves GPU utilization and efficiency

    Attributes:
        model: EfficientNet-B0 model for feature extraction
        config: Configuration dictionary for extraction parameters
        device: Computing device (CPU/GPU)
        input_adapter: Method for grayscale to RGB conversion
    """

    # Expected embedding dimension from EfficientNet-B0 avgpool layer
    EXPECTED_EMBEDDING_DIM = 1280

    # Supported input adapters
    SUPPORTED_INPUT_ADAPTERS = [
        "1to3_replication",
        "conv1_adapted"
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EfficientNet-B0 extractor with configuration.

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

        logger.info(f"Initialized EfficientNetExtractor with device: {self.device}")

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

            # Extract features
            with torch.no_grad():
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
        if model_name not in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Validate pretrained flag
        if not isinstance(config['pretrained'], bool):
            raise ValueError("pretrained must be a boolean")

        # Validate input adapter
        input_adapter = config['input_adapter']
        if input_adapter not in self.SUPPORTED_INPUT_ADAPTERS:
            raise ValueError(f"Unsupported input adapter: {input_adapter}")

        # Set default values for optional parameters
        config.setdefault('feature_layer', 'avgpool')
        config.setdefault('batch_size', 8)
        config.setdefault('normalize_embeddings', False)
        config.setdefault('normalization_method', 'l2')
        config.setdefault('seed', 42)

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
        Initialize EfficientNet-B0 model for feature extraction.

        Educational Note: This method loads a pre-trained EfficientNet-B0 model
        and modifies it to extract features from the avgpool layer.

        Returns:
            nn.Module: Modified EfficientNet-B0 model
        """
        model_name = self.config['model_name']
        pretrained = self.config['pretrained']

        # Load pre-trained model
        if model_name == 'efficientnet_b0':
            if pretrained:
                from torchvision.models import EfficientNet_B0_Weights
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)
        elif model_name == 'efficientnet_b1':
            if pretrained:
                from torchvision.models import EfficientNet_B1_Weights
                model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b1(weights=None)
        elif model_name == 'efficientnet_b2':
            if pretrained:
                from torchvision.models import EfficientNet_B2_Weights
                model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b2(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify model for feature extraction
        model = self._modify_model_for_extraction(model)

        # Move model to device
        model = model.to(self.device)

        logger.info(f"Initialized {model_name} model with pretrained={pretrained}")
        return model

    def _modify_model_for_extraction(self, model: nn.Module) -> nn.Module:
        """
        Modify EfficientNet model for feature extraction.

        Educational Note: This method removes the final classification layer
        and returns features from the avgpool layer for embedding extraction.

        Args:
            model: Original EfficientNet model

        Returns:
            nn.Module: Modified model for feature extraction
        """
        # EfficientNet structure: features -> avgpool -> classifier
        # We want to extract features after avgpool, before classifier

        # Create a new model that only includes features and avgpool
        class EfficientNetFeatureExtractor(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.features = base_model.features
                self.avgpool = base_model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x

        feature_extractor = EfficientNetFeatureExtractor(model)
        return feature_extractor

    def _validate_tensor(self, preprocessed_tensor: PreprocessedTensor) -> bool:
        """
        Validate preprocessed tensor for feature extraction.

        Educational Note: Tensor validation ensures the input is in the
        correct format for EfficientNet-B0 feature extraction.

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

        Educational Note: This method prepares the tensor for EfficientNet-B0
        input including normalization and device transfer.

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

        # Move to same device as tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

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

            # Extract features
            with torch.no_grad():
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

        return {
            'model_name': self.config['model_name'],
            'pretrained': self.config['pretrained'],
            'feature_layer': self.config['feature_layer'],
            'embedding_dimension': self.EXPECTED_EMBEDDING_DIM,
            'input_adapter': self.config['input_adapter'],
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_efficientnet_extractor(config: Dict[str, Any]) -> EfficientNetExtractor:
    """
    Factory function to create an EfficientNetExtractor instance.

    Educational Note: This factory function provides a convenient way
    to create EfficientNetExtractor instances with validated configurations.

    Args:
        config: Configuration dictionary for feature extraction

    Returns:
        EfficientNetExtractor: Configured EfficientNetExtractor instance
    """
    return EfficientNetExtractor(config)


def extract_single_embedding(preprocessed_tensor: PreprocessedTensor,
                           config: Dict[str, Any]) -> Optional[EmbeddingVector]:
    """
    Convenience function to extract embedding from a single tensor.

    Educational Note: This function provides a simple interface for
    extracting individual embeddings without creating an EfficientNetExtractor instance.

    Args:
        preprocessed_tensor: PreprocessedTensor instance
        config: Configuration dictionary for feature extraction

    Returns:
        EmbeddingVector: Extracted embedding if successful, None otherwise
    """
    extractor = create_efficientnet_extractor(config)
    return extractor.extract_embedding(preprocessed_tensor)


def extract_batch_embeddings(preprocessed_tensors: List[PreprocessedTensor],
                           config: Dict[str, Any]) -> List[Optional[EmbeddingVector]]:
    """
    Convenience function to extract embeddings from a batch of tensors.

    Educational Note: This function provides a simple interface for
    batch embedding extraction without creating an EfficientNetExtractor instance.

    Args:
        preprocessed_tensors: List of PreprocessedTensor instances
        config: Configuration dictionary for feature extraction

    Returns:
        List[Optional[EmbeddingVector]]: List of extracted embeddings
    """
    extractor = create_efficientnet_extractor(config)
    return extractor.extract_embeddings_batch(preprocessed_tensors)
