"""
EmbeddingVector model for ResNet-50 feature representation.

This module defines the data structure for representing 2048-dimensional
feature vectors extracted from ResNet-50, including extraction metadata
and validation.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- This model represents the third step in our unsupervised learning pipeline
- It captures high-level features extracted from ResNet-50's avgpool layer
- 2048-dimensional embeddings enable dimensionality reduction and clustering
- Extraction metadata enables reproducibility and performance analysis

Author: Research Team
Version: 1.0.0
"""

import torch
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class EmbeddingVector:
    """
    Represents 2048-dimensional feature vector from ResNet-50.
    
    This class encapsulates feature vectors extracted from ResNet-50's
    average pooling layer, which serves as a high-level representation
    of mammography images for clustering and analysis.
    
    Educational Notes:
    - Embedding dimension: 2048 (from ResNet-50 avgpool layer)
    - Feature extraction: Using pre-trained ResNet-50 on ImageNet
    - Input adapters: Handle grayscale to RGB conversion
    - Metadata tracking: Enables reproducibility and performance analysis
    
    Attributes:
        image_id (str): References MammographyImage.instance_id
        embedding (torch.Tensor): 2048-dimensional feature vector
        model_config (dict): ResNet-50 configuration used
        input_adapter (str): Channel handling method used
        extraction_time (float): Time taken for feature extraction (seconds)
        created_at (datetime): Timestamp of extraction
    """
    
    # Define expected embedding dimension for ResNet-50
    EXPECTED_EMBEDDING_DIM = 2048
    
    # Define valid input adapters
    VALID_INPUT_ADAPTERS = ["1to3_replication", "conv1_adapted"]
    
    # Define valid model configurations
    VALID_MODEL_CONFIGS = ["resnet50", "resnet50_pretrained"]
    
    def __init__(
        self,
        image_id: str,
        embedding: torch.Tensor,
        model_config: Dict[str, Any],
        input_adapter: str,
        extraction_time: float,
        device_used: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize an EmbeddingVector instance.
        
        Args:
            image_id: References MammographyImage.instance_id
            embedding: 2048-dimensional feature vector
            model_config: ResNet-50 configuration used
            input_adapter: Channel handling method used
            extraction_time: Time taken for feature extraction (seconds)
            device_used: Device used for extraction (CPU/CUDA/MPS)
            created_at: Timestamp of extraction (default: now)
            
        Raises:
            ValueError: If validation rules are violated
            TypeError: If data types are incorrect
        """
        # Initialize core attributes with validation
        self.image_id = self._validate_image_id(image_id)
        self.embedding = self._validate_embedding(embedding)
        self.model_config = self._validate_model_config(model_config)
        self.input_adapter = self._validate_input_adapter(input_adapter)
        self.extraction_time = self._validate_extraction_time(extraction_time)
        self.device_used = device_used or "cpu"
        self.created_at = created_at or datetime.now()
        
        # Initialize tracking attributes
        self.validation_errors: List[str] = []
        self.updated_at = datetime.now()
        
        # Validate embedding dimension
        self._validate_embedding_dimension()
        
        # Log creation for educational purposes
        logger.info(f"Created EmbeddingVector: {self.image_id} with dimension {self.embedding.shape[0]}")
    
    def _validate_image_id(self, image_id: str) -> str:
        """
        Validate image ID.
        
        Educational Note: Image ID links this embedding back to the original
        MammographyImage and PreprocessedTensor for complete traceability.
        
        Args:
            image_id: Image identifier to validate
            
        Returns:
            str: Validated image ID
            
        Raises:
            ValueError: If image ID is invalid
            TypeError: If image ID is not a string
        """
        if not isinstance(image_id, str):
            raise TypeError(f"image_id must be a string, got {type(image_id)}")
        
        if not image_id.strip():
            raise ValueError("image_id cannot be empty or whitespace")
        
        return image_id.strip()
    
    def _validate_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Validate embedding tensor.
        
        Educational Note: Embedding validation ensures proper format for
        clustering algorithms and dimensionality reduction techniques.
        
        Args:
            embedding: Embedding tensor to validate
            
        Returns:
            torch.Tensor: Validated embedding tensor
            
        Raises:
            ValueError: If embedding is invalid
            TypeError: If embedding is not a PyTorch tensor
        """
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"embedding must be a torch.Tensor, got {type(embedding)}")
        
        # Check tensor dimensions (should be 1D)
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1D, got {embedding.ndim}D")
        
        # Check embedding dimension (should be 2048 for ResNet-50)
        if embedding.shape[0] != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"embedding must have {self.EXPECTED_EMBEDDING_DIM} dimensions, got {embedding.shape[0]}")
        
        # Check data type (should be float32)
        if embedding.dtype != torch.float32:
            logger.warning(f"Converting embedding from {embedding.dtype} to float32")
            embedding = embedding.float()
        
        # Check for NaN or infinite values
        if torch.any(torch.isnan(embedding)):
            raise ValueError("embedding contains NaN values")
        
        if torch.any(torch.isinf(embedding)):
            raise ValueError("embedding contains infinite values")
        
        return embedding
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration.
        
        Educational Note: Model configuration validation ensures all necessary
        parameters for ResNet-50 feature extraction are present and valid.
        
        Args:
            model_config: Model configuration dictionary to validate
            
        Returns:
            Dict[str, Any]: Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If configuration is not a dictionary
        """
        if not isinstance(model_config, dict):
            raise TypeError(f"model_config must be a dictionary, got {type(model_config)}")
        
        # Check for required configuration keys
        required_keys = ["model_name", "pretrained", "feature_layer"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"Missing required model configuration key: {key}")
        
        # Validate model name
        model_name = model_config.get("model_name")
        if model_name not in self.VALID_MODEL_CONFIGS:
            logger.warning(f"Unknown model configuration: {model_name}")
        
        # Validate feature layer
        feature_layer = model_config.get("feature_layer")
        if feature_layer != "avgpool":
            logger.warning(f"Expected feature_layer 'avgpool', got '{feature_layer}'")
        
        return model_config
    
    def _validate_input_adapter(self, input_adapter: str) -> str:
        """
        Validate input adapter.
        
        Educational Note: Input adapter validation ensures the correct
        grayscale to RGB conversion method was used.
        
        Args:
            input_adapter: Input adapter to validate
            
        Returns:
            str: Validated input adapter
            
        Raises:
            ValueError: If input adapter is invalid
            TypeError: If input adapter is not a string
        """
        if not isinstance(input_adapter, str):
            raise TypeError(f"input_adapter must be a string, got {type(input_adapter)}")
        
        if input_adapter not in self.VALID_INPUT_ADAPTERS:
            raise ValueError(f"input_adapter must be one of {self.VALID_INPUT_ADAPTERS}, got {input_adapter}")
        
        return input_adapter
    
    def _validate_extraction_time(self, extraction_time: float) -> float:
        """
        Validate extraction time.
        
        Educational Note: Extraction time validation ensures reasonable
        performance metrics for feature extraction.
        
        Args:
            extraction_time: Extraction time to validate (seconds)
            
        Returns:
            float: Validated extraction time
            
        Raises:
            ValueError: If extraction time is invalid
            TypeError: If extraction time is not a number
        """
        if not isinstance(extraction_time, (int, float)):
            raise TypeError(f"extraction_time must be a number, got {type(extraction_time)}")
        
        if extraction_time < 0:
            raise ValueError(f"extraction_time must be non-negative, got {extraction_time}")
        
        if extraction_time > 3600:  # 1 hour
            logger.warning(f"extraction_time seems unusually long: {extraction_time}s")
        
        return float(extraction_time)
    
    def _validate_embedding_dimension(self) -> None:
        """
        Validate embedding dimension matches expected value.
        
        Educational Note: This validation ensures consistency with
        ResNet-50 architecture and downstream processing requirements.
        
        Raises:
            ValueError: If embedding dimension is incorrect
        """
        if self.embedding.shape[0] != self.EXPECTED_EMBEDDING_DIM:
            error_msg = f"Embedding dimension {self.embedding.shape[0]} doesn't match expected {self.EXPECTED_EMBEDDING_DIM}"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
    
    def get_embedding_stats(self) -> Dict[str, float]:
        """
        Get statistical information about the embedding.
        
        Educational Note: Embedding statistics are useful for understanding
        the feature distribution and ensuring proper normalization.
        
        Returns:
            Dict[str, float]: Dictionary containing embedding statistics
        """
        return {
            "mean": float(torch.mean(self.embedding).item()),
            "std": float(torch.std(self.embedding).item()),
            "min": float(torch.min(self.embedding).item()),
            "max": float(torch.max(self.embedding).item()),
            "norm": float(torch.norm(self.embedding).item()),
            "dimension": int(self.embedding.shape[0]),
            "dtype": str(self.embedding.dtype)
        }
    
    def normalize_embedding(self, method: str = "l2") -> torch.Tensor:
        """
        Normalize the embedding vector.
        
        Educational Note: Normalization can improve clustering performance
        and ensure consistent feature scales across different images.
        
        Args:
            method: Normalization method ("l2", "l1", "min_max")
            
        Returns:
            torch.Tensor: Normalized embedding vector
            
        Raises:
            ValueError: If normalization method is invalid
        """
        if method == "l2":
            # L2 normalization (unit norm)
            norm = torch.norm(self.embedding, p=2)
            if norm == 0:
                return self.embedding
            return self.embedding / norm
        
        elif method == "l1":
            # L1 normalization (sum to 1)
            norm = torch.norm(self.embedding, p=1)
            if norm == 0:
                return self.embedding
            return self.embedding / norm
        
        elif method == "min_max":
            # Min-max normalization (0 to 1)
            min_val = torch.min(self.embedding)
            max_val = torch.max(self.embedding)
            if max_val == min_val:
                return self.embedding
            return (self.embedding - min_val) / (max_val - min_val)
        
        else:
            raise ValueError(f"Invalid normalization method: {method}. Must be one of ['l2', 'l1', 'min_max']")
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the embedding extraction process.
        
        Educational Note: This summary provides a complete record of
        the feature extraction process for reproducibility and analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing extraction summary
        """
        return {
            "image_id": self.image_id,
            "embedding_dimension": int(self.embedding.shape[0]),
            "model_config": self.model_config,
            "input_adapter": self.input_adapter,
            "extraction_time": self.extraction_time,
            "device_used": self.device_used,
            "embedding_stats": self.get_embedding_stats(),
            "created_at": self.created_at.isoformat(),
            "validation_errors": self.validation_errors
        }
    
    def save_embedding(self, file_path: Union[str, Path]) -> bool:
        """
        Save embedding data to file.
        
        Educational Note: Embedding saving enables caching of extracted
        features to avoid reprocessing during experiments.
        
        Args:
            file_path: Path where to save the embedding
            
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save embedding data
            torch.save({
                "embedding": self.embedding,
                "image_id": self.image_id,
                "model_config": self.model_config,
                "input_adapter": self.input_adapter,
                "extraction_time": self.extraction_time,
                "device_used": self.device_used,
                "created_at": self.created_at.isoformat(),
                "validation_errors": self.validation_errors
            }, file_path)
            
            logger.info(f"Saved EmbeddingVector to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving EmbeddingVector to {file_path}: {str(e)}")
            return False
    
    @classmethod
    def load_embedding(cls, file_path: Union[str, Path]) -> "EmbeddingVector":
        """
        Load embedding data from file.
        
        Educational Note: This class method enables loading of previously
        saved embeddings for analysis and experimentation.
        
        Args:
            file_path: Path to the saved embedding file
            
        Returns:
            EmbeddingVector: Loaded instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Embedding file not found: {file_path}")
            
            # Load embedding data
            data = torch.load(file_path, map_location="cpu")
            
            # Parse creation timestamp
            created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            
            # Create EmbeddingVector instance
            embedding_vector = cls(
                image_id=data["image_id"],
                embedding=data["embedding"],
                model_config=data["model_config"],
                input_adapter=data["input_adapter"],
                extraction_time=data["extraction_time"],
                device_used=data.get("device_used", "cpu"),
                created_at=created_at
            )
            
            # Restore validation errors if any
            if data.get("validation_errors"):
                embedding_vector.validation_errors = data["validation_errors"]
            
            logger.info(f"Loaded EmbeddingVector from {file_path}")
            return embedding_vector
            
        except Exception as e:
            raise ValueError(f"Error loading EmbeddingVector from {file_path}: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"EmbeddingVector("
            f"image_id='{self.image_id}', "
            f"dimension={self.embedding.shape[0]}, "
            f"adapter='{self.input_adapter}', "
            f"time={self.extraction_time:.3f}s)"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Embedding Vector: {self.image_id}\n"
            f"Dimension: {self.embedding.shape[0]}\n"
            f"Input Adapter: {self.input_adapter}\n"
            f"Extraction Time: {self.extraction_time:.3f}s\n"
            f"Device Used: {self.device_used}\n"
            f"Model Config: {self.model_config.get('model_name', 'unknown')}"
        )


def create_embedding_vector_from_extraction(
    image_id: str,
    embedding: torch.Tensor,
    model_config: Dict[str, Any],
    input_adapter: str,
    extraction_time: float,
    device_used: str = "cpu"
) -> EmbeddingVector:
    """
    Create an EmbeddingVector instance from feature extraction.
    
    Educational Note: This factory function demonstrates how to create
    EmbeddingVector instances from ResNet-50 feature extraction,
    enabling standardized embedding creation.
    
    Args:
        image_id: Image identifier
        embedding: Extracted feature vector
        model_config: ResNet-50 configuration used
        input_adapter: Channel handling method used
        extraction_time: Time taken for extraction
        device_used: Device used for extraction
        
    Returns:
        EmbeddingVector: Created instance
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Create EmbeddingVector instance
    embedding_vector = EmbeddingVector(
        image_id=image_id,
        embedding=embedding,
        model_config=model_config,
        input_adapter=input_adapter,
        extraction_time=extraction_time,
        device_used=device_used
    )
    
    return embedding_vector


def batch_create_embedding_vectors(
    image_ids: List[str],
    embeddings: torch.Tensor,
    model_config: Dict[str, Any],
    input_adapter: str,
    extraction_times: List[float],
    device_used: str = "cpu"
) -> List[EmbeddingVector]:
    """
    Create multiple EmbeddingVector instances from batch extraction.
    
    Educational Note: This function enables efficient creation of multiple
    embeddings from batch processing, which is common in deep learning
    workflows.
    
    Args:
        image_ids: List of image identifiers
        embeddings: Batch of extracted feature vectors (N, 2048)
        model_config: ResNet-50 configuration used
        input_adapter: Channel handling method used
        extraction_times: List of extraction times per image
        device_used: Device used for extraction
        
    Returns:
        List[EmbeddingVector]: List of created instances
        
    Raises:
        ValueError: If batch dimensions don't match
    """
    if len(image_ids) != embeddings.shape[0]:
        raise ValueError(f"Number of image_ids ({len(image_ids)}) doesn't match batch size ({embeddings.shape[0]})")
    
    if len(image_ids) != len(extraction_times):
        raise ValueError(f"Number of image_ids ({len(image_ids)}) doesn't match number of extraction_times ({len(extraction_times)})")
    
    embedding_vectors = []
    for i, image_id in enumerate(image_ids):
        embedding_vector = EmbeddingVector(
            image_id=image_id,
            embedding=embeddings[i],
            model_config=model_config,
            input_adapter=input_adapter,
            extraction_time=extraction_times[i],
            device_used=device_used
        )
        embedding_vectors.append(embedding_vector)
    
    return embedding_vectors
