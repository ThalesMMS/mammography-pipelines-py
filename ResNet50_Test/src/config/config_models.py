"""
Configuration models for pipeline components using Pydantic validation.

This module defines configuration classes for all pipeline components
using Pydantic for automatic validation, type checking, and serialization.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Pydantic provides automatic validation and type checking
- Configuration classes ensure consistent parameter handling
- Default values enable easy experimentation
- Validation rules prevent invalid configurations

Author: Research Team
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseModel):
    """
    Configuration for image preprocessing operations.

    This configuration class defines all parameters needed for
    mammography image preprocessing, including resizing, normalization,
    and input adapter settings.

    Educational Notes:
    - Target size determines final image dimensions
    - Normalization methods affect feature extraction quality
    - Input adapters handle grayscale to RGB conversion
    - Border removal can improve clustering quality

    Fields:
        target_size: Final image dimensions (H, W)
        normalization_method: Normalization approach
        input_adapter: Grayscale to RGB conversion method
        border_removal: Whether to remove image borders
        padding_strategy: Padding method for resizing
        seed: Random seed for reproducibility
    """

    # Image dimensions
    target_size: Tuple[int, int] = Field(
        default=(512, 512), description="Target image dimensions (height, width)"
    )

    # Normalization options
    normalization_method: Literal[
        "z_score_per_image", "fixed_window", "min_max_scaling", "percentile_scaling"
    ] = Field(
        default="z_score_per_image",
        description="Normalization method for image preprocessing",
    )

    # Input adapter options
    input_adapter: Literal["1to3_replication", "conv1_adapted"] = Field(
        default="1to3_replication", description="Method for converting grayscale to RGB"
    )

    # Border processing
    border_removal: bool = Field(
        default=True, description="Whether to remove image borders"
    )

    # Padding strategy
    padding_strategy: Literal["reflect", "constant", "edge"] = Field(
        default="reflect", description="Padding strategy for image resizing"
    )

    # Reproducibility
    seed: int = Field(
        default=42, description="Random seed for reproducible preprocessing"
    )

    # Advanced options
    keep_aspect_ratio: bool = Field(
        default=True, description="Whether to preserve aspect ratio during resizing"
    )

    border_threshold: float = Field(
        default=0.1, description="Threshold for border detection"
    )

    min_breast_area: float = Field(
        default=0.05, description="Minimum breast area ratio for border removal"
    )

    @validator("target_size")
    def validate_target_size(cls, v):
        """Validate target size dimensions."""
        if len(v) != 2:
            raise ValueError("target_size must have exactly 2 elements")

        for i, size in enumerate(v):
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"target_size[{i}] must be a positive integer, got {size}"
                )

            if size < 64 or size > 2048:
                logger.warning(
                    f"target_size[{i}] = {size} is outside typical range [64, 2048]"
                )

        return v

    @validator("seed")
    def validate_seed(cls, v):
        """Validate random seed."""
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"seed must be a non-negative integer, got {v}")
        return v

    @validator("border_threshold")
    def validate_border_threshold(cls, v):
        """Validate border threshold."""
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(f"border_threshold must be between 0 and 1, got {v}")
        return v

    @validator("min_breast_area")
    def validate_min_breast_area(cls, v):
        """Validate minimum breast area."""
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(f"min_breast_area must be between 0 and 1, got {v}")
        return v

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


class EmbeddingConfig(BaseModel):
    """
    Configuration for ResNet-50 embedding extraction.

    This configuration class defines all parameters needed for
    ResNet-50 feature extraction, including model settings,
    device configuration, and performance options.

    Educational Notes:
    - Model configuration affects feature quality
    - Device selection impacts processing speed
    - Batch size affects memory usage and speed
    - Input adapter must match preprocessing config

    Fields:
        model_name: ResNet model variant
        pretrained: Whether to use pre-trained weights
        feature_layer: Layer for feature extraction
        device: Computing device to use
        batch_size: Batch size for processing
        input_adapter: Channel handling method
    """

    # Model configuration
    model_name: Literal["resnet50", "resnet34", "resnet18"] = Field(
        default="resnet50", description="ResNet model variant for feature extraction"
    )

    pretrained: bool = Field(
        default=True, description="Whether to use pre-trained ImageNet weights"
    )

    feature_layer: Literal["avgpool", "fc"] = Field(
        default="avgpool", description="Layer for feature extraction"
    )

    # Device configuration
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto", description="Computing device for feature extraction"
    )

    # Performance options
    batch_size: int = Field(
        default=8, description="Batch size for embedding extraction"
    )

    # Input adapter (must match preprocessing)
    input_adapter: Literal["1to3_replication", "conv1_adapted"] = Field(
        default="1to3_replication", description="Method for handling grayscale input"
    )

    # Advanced options
    normalize_embeddings: bool = Field(
        default=False, description="Whether to normalize embeddings after extraction"
    )

    normalization_method: Literal["l2", "l1", "min_max"] = Field(
        default="l2", description="Normalization method for embeddings"
    )

    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducible extraction")

    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")

        if v > 64:
            logger.warning(f"batch_size = {v} is quite large, may cause memory issues")

        return v

    @validator("seed")
    def validate_seed(cls, v):
        """Validate random seed."""
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"seed must be a non-negative integer, got {v}")
        return v

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


class ClusteringConfig(BaseModel):
    """
    Configuration for clustering algorithms and evaluation.

    This configuration class defines all parameters needed for
    clustering algorithms, dimensionality reduction, and evaluation
    metrics computation.

    Educational Notes:
    - Algorithm selection affects clustering quality
    - Dimensionality reduction improves clustering performance
    - Evaluation metrics enable algorithm comparison
    - Hyperparameters control clustering behavior

    Fields:
        algorithm: Clustering algorithm to use
        n_clusters: Number of clusters (for some algorithms)
        pca_dimensions: PCA dimensionality reduction
        evaluation_metrics: Metrics to compute
        hyperparameters: Algorithm-specific parameters
    """

    # Algorithm selection
    algorithm: Literal["kmeans", "gmm", "hdbscan", "agglomerative"] = Field(
        default="kmeans", description="Clustering algorithm to use"
    )

    # Cluster configuration
    n_clusters: Optional[int] = Field(
        default=4, description="Number of clusters (for K-means, GMM, Agglomerative)"
    )

    # Dimensionality reduction
    pca_dimensions: int = Field(
        default=50, description="Number of PCA dimensions for clustering"
    )

    # Evaluation metrics
    evaluation_metrics: List[
        Literal["silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi"]
    ] = Field(
        default=["silhouette", "davies_bouldin", "calinski_harabasz"],
        description="Evaluation metrics to compute",
    )

    # Algorithm-specific hyperparameters
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific hyperparameters"
    )

    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducible clustering")

    # Advanced options
    umap_visualization: bool = Field(
        default=True, description="Whether to create UMAP visualization"
    )

    umap_dimensions: int = Field(
        default=2, description="UMAP embedding dimensions for visualization"
    )

    @validator("n_clusters")
    def validate_n_clusters(cls, v, values):
        """Validate number of clusters."""
        if v is not None:
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"n_clusters must be a positive integer, got {v}")

            if v > 20:
                logger.warning(
                    f"n_clusters = {v} is quite large, may affect clustering quality"
                )

        return v

    @validator("pca_dimensions")
    def validate_pca_dimensions(cls, v):
        """Validate PCA dimensions."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"pca_dimensions must be a positive integer, got {v}")

        if v < 2:
            raise ValueError("pca_dimensions must be at least 2")

        if v > 2048:
            logger.warning(
                f"pca_dimensions = {v} is larger than ResNet-50 embedding dimension"
            )

        return v

    @validator("seed")
    def validate_seed(cls, v):
        """Validate random seed."""
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"seed must be a non-negative integer, got {v}")
        return v

    @validator("hyperparameters")
    def validate_hyperparameters(cls, v, values):
        """Validate algorithm-specific hyperparameters."""
        algorithm = values.get("algorithm", "kmeans")

        # Set default hyperparameters based on algorithm
        if algorithm == "kmeans":
            defaults = {
                "n_clusters": values.get("n_clusters", 4),
                "random_state": values.get("seed", 42),
                "n_init": 10,
                "max_iter": 300,
            }
        elif algorithm == "gmm":
            defaults = {
                "n_components": values.get("n_clusters", 4),
                "random_state": values.get("seed", 42),
                "covariance_type": "full",
                "max_iter": 100,
            }
        elif algorithm == "hdbscan":
            defaults = {
                "min_cluster_size": 10,
                "min_samples": 5,
                "random_state": values.get("seed", 42),
            }
        elif algorithm == "agglomerative":
            defaults = {"n_clusters": values.get("n_clusters", 4), "linkage": "ward"}
        else:
            defaults = {}

        # Merge with provided hyperparameters
        merged = {**defaults, **v}

        # Validate algorithm-specific parameters
        if algorithm == "hdbscan":
            if "min_cluster_size" in merged and merged["min_cluster_size"] < 2:
                raise ValueError("hdbscan min_cluster_size must be at least 2")

        return merged

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


class PipelineConfig(BaseModel):
    """
    Complete pipeline configuration combining all component configs.

    This configuration class combines all component configurations
    into a single, validated configuration object for the entire
    breast density exploration pipeline.

    Educational Notes:
    - Unified configuration ensures consistency
    - Cross-component validation prevents conflicts
    - Default values enable easy experimentation
    - Configuration can be saved/loaded for reproducibility

    Fields:
        preprocessing: Preprocessing configuration
        embedding: Embedding extraction configuration
        clustering: Clustering configuration
        experiment: Experiment metadata
    """

    # Component configurations
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Image preprocessing configuration",
    )

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding extraction configuration",
    )

    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="Clustering configuration"
    )

    # Experiment metadata
    experiment_name: str = Field(
        default="breast_density_exploration", description="Name of the experiment"
    )

    experiment_description: str = Field(
        default="Unsupervised clustering of mammography images using ResNet-50 embeddings",
        description="Description of the experiment",
    )

    # Paths
    data_dir: str = Field(default="data", description="Directory containing input data")

    output_dir: str = Field(
        default="results", description="Directory for output results"
    )

    cache_dir: str = Field(
        default="cache", description="Directory for cached intermediate results"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # Reproducibility
    seed: int = Field(default=42, description="Global random seed for reproducibility")

    @validator("experiment_name")
    def validate_experiment_name(cls, v):
        """Validate experiment name."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("experiment_name must be a non-empty string")

        # Remove invalid characters
        import re

        v = re.sub(r"[^\w\-_]", "_", v.strip())

        return v

    @validator("seed")
    def validate_seed(cls, v):
        """Validate global random seed."""
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"seed must be a non-negative integer, got {v}")
        return v

    @validator("preprocessing")
    def validate_preprocessing_config(cls, v, values):
        """Validate preprocessing configuration consistency."""
        # Ensure preprocessing seed matches global seed if not explicitly set
        global_seed = values.get("seed", 42)
        if v.seed == 42:  # Default value
            v.seed = global_seed
        return v

    @validator("embedding")
    def validate_embedding_config(cls, v, values):
        """Validate embedding configuration consistency."""
        # Ensure embedding seed matches global seed if not explicitly set
        global_seed = values.get("seed", 42)
        if v.seed == 42:  # Default value
            v.seed = global_seed

        # Ensure input adapter matches preprocessing config
        preprocessing_config = values.get("preprocessing")
        if (
            preprocessing_config
            and preprocessing_config.input_adapter != v.input_adapter
        ):
            logger.warning(
                f"Input adapter mismatch: preprocessing={preprocessing_config.input_adapter}, embedding={v.input_adapter}"
            )

        return v

    @validator("clustering")
    def validate_clustering_config(cls, v, values):
        """Validate clustering configuration consistency."""
        # Ensure clustering seed matches global seed if not explicitly set
        global_seed = values.get("seed", 42)
        if v.seed == 42:  # Default value
            v.seed = global_seed

        return v

    def save_config(self, file_path: Union[str, Path]) -> bool:
        """
        Save configuration to file.

        Educational Note: Configuration saving enables reproducibility
        and sharing of experimental setups.

        Args:
            file_path: Path where to save the configuration

        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(self.json(indent=2))

            logger.info(f"Saved configuration to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e!s}")
            return False

    @classmethod
    def load_config(cls, file_path: Union[str, Path]) -> "PipelineConfig":
        """
        Load configuration from file.

        Educational Note: Configuration loading enables reproduction
        of previous experiments and sharing of setups.

        Args:
            file_path: Path to the configuration file

        Returns:
            PipelineConfig: Loaded configuration instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")

            with open(file_path, "r") as f:
                config_data = f.read()

            config = cls.parse_raw(config_data)
            logger.info(f"Loaded configuration from {file_path}")
            return config

        except Exception as e:
            raise ValueError(f"Error loading configuration from {file_path}: {e!s}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.

        Educational Note: This summary provides a concise overview
        of the configuration for logging and documentation.

        Returns:
            Dict[str, Any]: Configuration summary
        """
        return {
            "experiment_name": self.experiment_name,
            "experiment_description": self.experiment_description,
            "seed": self.seed,
            "preprocessing": {
                "target_size": self.preprocessing.target_size,
                "normalization_method": self.preprocessing.normalization_method,
                "input_adapter": self.preprocessing.input_adapter,
                "border_removal": self.preprocessing.border_removal,
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "pretrained": self.embedding.pretrained,
                "batch_size": self.embedding.batch_size,
                "device": self.embedding.device,
            },
            "clustering": {
                "algorithm": self.clustering.algorithm,
                "n_clusters": self.clustering.n_clusters,
                "pca_dimensions": self.clustering.pca_dimensions,
                "evaluation_metrics": self.clustering.evaluation_metrics,
            },
        }

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


# Convenience functions for creating default configurations
def create_default_preprocessing_config(**kwargs) -> PreprocessingConfig:
    """Create default preprocessing configuration with optional overrides."""
    return PreprocessingConfig(**kwargs)


def create_default_embedding_config(**kwargs) -> EmbeddingConfig:
    """Create default embedding configuration with optional overrides."""
    return EmbeddingConfig(**kwargs)


def create_default_clustering_config(**kwargs) -> ClusteringConfig:
    """Create default clustering configuration with optional overrides."""
    return ClusteringConfig(**kwargs)


def create_default_pipeline_config(**kwargs) -> PipelineConfig:
    """Create default pipeline configuration with optional overrides."""
    return PipelineConfig(**kwargs)
