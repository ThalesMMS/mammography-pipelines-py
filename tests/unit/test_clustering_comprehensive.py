"""
Comprehensive unit tests for clustering modules.

These tests validate the complete clustering pipeline including algorithms,
results, and integration with embedding vectors.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("hdbscan")

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mammography.clustering.clustering_algorithms import (
    ClusteringAlgorithms,
    create_clustering_algorithms,
    cluster_embeddings,
)
from mammography.clustering.clustering_result import (
    ClusteringResult,
    create_clustering_result_from_algorithm,
)
from mammography.models.embeddings.embedding_vector import EmbeddingVector


MATMUL_ERRSTATE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


@pytest.fixture(autouse=True)
def _suppress_numpy_matmul_warnings():
    with np.errstate(**MATMUL_ERRSTATE):
        yield


class TestClusteringAlgorithmsComprehensive:
    """Comprehensive tests for ClusteringAlgorithms class."""

    @pytest.fixture
    def sample_embedding_vectors(self) -> List[EmbeddingVector]:
        """Create sample embedding vectors for testing."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 512

        # Create embeddings with cluster structure
        cluster_centers = rng.normal(scale=0.5, size=(4, n_features))
        embeddings = []

        for i in range(4):
            cluster_samples = (
                rng.normal(scale=0.1, size=(25, n_features)) + cluster_centers[i]
            )
            embeddings.append(cluster_samples)

        embeddings_array = np.vstack(embeddings).astype(np.float32)

        # Create EmbeddingVector instances
        embedding_vectors = []
        for i in range(n_samples):
            emb_tensor = torch.tensor(embeddings_array[i], dtype=torch.float32)
            embedding_vectors.append(
                EmbeddingVector(
                    image_id=f"img_{i:03d}",
                    embedding=emb_tensor,
                    model_config={
                        "model_name": "resnet50_pretrained",
                        "pretrained": True,
                        "feature_layer": "avgpool"
                    },
                    input_adapter="1to3_replication",
                    extraction_time=0.1,
                    device_used="cpu"
                )
            )

        return embedding_vectors

    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Create base configuration for testing."""
        return {
            "algorithm": "kmeans",
            "pca_dimensions": 50,
            "n_clusters": 4,
            "seed": 42,
        }

    def test_clustering_algorithms_init(self, base_config):
        """Test ClusteringAlgorithms initialization."""
        clusterer = ClusteringAlgorithms(base_config)

        assert clusterer.config["algorithm"] == "kmeans"
        assert clusterer.config["pca_dimensions"] == 50
        assert clusterer.config["n_clusters"] == 4
        assert clusterer.config["seed"] == 42
        assert clusterer.pca_model is None
        assert clusterer.scaler is not None

    def test_clustering_algorithms_invalid_algorithm(self):
        """Test ClusteringAlgorithms with invalid algorithm."""
        config = {
            "algorithm": "invalid_algorithm",
            "pca_dimensions": 50,
        }

        with pytest.raises(ValueError, match="Unsupported clustering algorithm"):
            ClusteringAlgorithms(config)

    def test_clustering_algorithms_invalid_pca_dimensions(self):
        """Test ClusteringAlgorithms with invalid PCA dimensions."""
        config = {
            "algorithm": "kmeans",
            "pca_dimensions": 0,
        }

        with pytest.raises(ValueError, match="pca_dimensions must be a positive integer"):
            ClusteringAlgorithms(config)

    def test_clustering_algorithms_missing_config(self):
        """Test ClusteringAlgorithms with missing required config."""
        config = {
            "algorithm": "kmeans",
            # Missing pca_dimensions
        }

        with pytest.raises(ValueError, match="Missing required configuration parameter"):
            ClusteringAlgorithms(config)

    def test_kmeans_clustering(self, base_config, sample_embedding_vectors):
        """Test K-means clustering algorithm."""
        base_config["algorithm"] = "kmeans"
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate result
        assert result is not None
        assert result.algorithm == "kmeans"
        assert len(result.cluster_labels) == len(sample_embedding_vectors)
        assert result.centroids is not None
        assert result.centroids.shape[0] == base_config["n_clusters"]
        assert result.metrics is not None

    def test_gmm_clustering(self, base_config, sample_embedding_vectors):
        """Test GMM clustering algorithm."""
        base_config["algorithm"] = "gmm"
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate result
        assert result is not None
        assert result.algorithm == "gmm"
        assert len(result.cluster_labels) == len(sample_embedding_vectors)
        assert result.centroids is not None
        assert result.uncertainty_scores is not None
        assert len(result.uncertainty_scores) == len(sample_embedding_vectors)
        assert result.metrics is not None

    def test_hdbscan_clustering(self, base_config, sample_embedding_vectors):
        """Test HDBSCAN clustering algorithm."""
        base_config["algorithm"] = "hdbscan"
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate result
        assert result is not None
        assert result.algorithm == "hdbscan"
        assert len(result.cluster_labels) == len(sample_embedding_vectors)
        assert result.uncertainty_scores is not None
        assert len(result.uncertainty_scores) == len(sample_embedding_vectors)
        assert result.metrics is not None

    def test_agglomerative_clustering(self, base_config, sample_embedding_vectors):
        """Test Agglomerative clustering algorithm."""
        base_config["algorithm"] = "agglomerative"
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate result
        assert result is not None
        assert result.algorithm == "agglomerative"
        assert len(result.cluster_labels) == len(sample_embedding_vectors)
        assert result.metrics is not None

    def test_clustering_reproducibility(self, base_config, sample_embedding_vectors):
        """Test clustering reproducibility with fixed seeds."""
        base_config["algorithm"] = "kmeans"
        base_config["seed"] = 42

        results = []
        for _ in range(3):
            clusterer = ClusteringAlgorithms(base_config)
            result = clusterer.cluster_embeddings(sample_embedding_vectors)
            results.append(result.cluster_labels)

        # Results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), "Clustering not reproducible"

    def test_clustering_pca_reduction(self, base_config, sample_embedding_vectors):
        """Test PCA dimensionality reduction in clustering."""
        base_config["pca_dimensions"] = 50
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate PCA was applied
        assert result is not None
        assert clusterer.pca_model is not None
        assert clusterer.pca_model.n_components_ == 50

        # Get PCA info
        pca_info = clusterer.get_pca_info()
        assert pca_info is not None
        assert pca_info["n_components"] == 50
        assert "explained_variance_ratio" in pca_info
        assert "total_explained_variance" in pca_info

    def test_clustering_skip_pca_when_dimensions_too_high(
        self, base_config, sample_embedding_vectors
    ):
        """Test that PCA is skipped when dimensions are >= embedding dimensions."""
        base_config["pca_dimensions"] = 1000  # Higher than embedding dimension
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate result
        assert result is not None
        # PCA model may still be None since it was skipped
        # But result should still be valid

    def test_clustering_evaluation_metrics(self, base_config, sample_embedding_vectors):
        """Test clustering evaluation metrics computation."""
        clusterer = ClusteringAlgorithms(base_config)
        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate metrics
        assert result is not None
        assert "silhouette" in result.metrics
        assert "davies_bouldin" in result.metrics
        assert "calinski_harabasz" in result.metrics

        # Validate metric ranges
        assert -1 <= result.metrics["silhouette"] <= 1
        assert result.metrics["davies_bouldin"] >= 0
        assert result.metrics["calinski_harabasz"] >= 0

    def test_clustering_different_cluster_numbers(
        self, base_config, sample_embedding_vectors
    ):
        """Test clustering with different numbers of clusters."""
        n_clusters_list = [2, 3, 4, 5, 6]

        for n_clusters in n_clusters_list:
            base_config["n_clusters"] = n_clusters
            clusterer = ClusteringAlgorithms(base_config)

            result = clusterer.cluster_embeddings(sample_embedding_vectors)

            # Validate result
            assert result is not None
            assert len(torch.unique(result.cluster_labels)) <= n_clusters

    def test_clustering_empty_embeddings(self, base_config):
        """Test clustering with empty embedding list."""
        clusterer = ClusteringAlgorithms(base_config)
        result = clusterer.cluster_embeddings([])

        assert result is None

    def test_clustering_single_embedding(self, base_config, sample_embedding_vectors):
        """Test clustering with single embedding."""
        clusterer = ClusteringAlgorithms(base_config)
        result = clusterer.cluster_embeddings([sample_embedding_vectors[0]])

        # Should fail with too few samples
        assert result is None

    def test_clustering_inconsistent_dimensions(self, base_config, sample_embedding_vectors):
        """Test clustering with inconsistent embedding dimensions."""
        # Create embedding with different dimension
        bad_embedding = EmbeddingVector(
            image_id="bad_img",
            embedding=torch.randn(256),  # Different dimension
            metadata={},
        )

        embeddings = sample_embedding_vectors[:10] + [bad_embedding]
        clusterer = ClusteringAlgorithms(base_config)
        result = clusterer.cluster_embeddings(embeddings)

        # Should fail validation
        assert result is None

    def test_clustering_hyperparameters(self, base_config, sample_embedding_vectors):
        """Test clustering with custom hyperparameters."""
        base_config["hyperparameters"] = {
            "n_clusters": 5,
            "n_init": 20,
            "max_iter": 500,
            "random_state": 42,
        }
        clusterer = ClusteringAlgorithms(base_config)

        result = clusterer.cluster_embeddings(sample_embedding_vectors)

        # Validate hyperparameters were used
        assert result is not None
        assert result.hyperparameters["n_clusters"] == 5
        assert result.hyperparameters["n_init"] == 20
        assert result.hyperparameters["max_iter"] == 500

    def test_create_clustering_algorithms_factory(self, base_config):
        """Test create_clustering_algorithms factory function."""
        clusterer = create_clustering_algorithms(base_config)

        assert isinstance(clusterer, ClusteringAlgorithms)
        assert clusterer.config["algorithm"] == base_config["algorithm"]

    def test_cluster_embeddings_convenience_function(
        self, base_config, sample_embedding_vectors
    ):
        """Test cluster_embeddings convenience function."""
        result = cluster_embeddings(sample_embedding_vectors, base_config)

        assert result is not None
        assert isinstance(result, ClusteringResult)
        assert len(result.cluster_labels) == len(sample_embedding_vectors)


class TestClusteringResultComprehensive:
    """Comprehensive tests for ClusteringResult class."""

    @pytest.fixture
    def sample_cluster_labels(self) -> torch.Tensor:
        """Create sample cluster labels."""
        return torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1], dtype=torch.long)

    @pytest.fixture
    def sample_centroids(self) -> torch.Tensor:
        """Create sample centroids."""
        return torch.randn(4, 50, dtype=torch.float32)

    @pytest.fixture
    def sample_uncertainty_scores(self) -> torch.Tensor:
        """Create sample uncertainty scores."""
        return torch.rand(10, dtype=torch.float32)

    @pytest.fixture
    def sample_metrics(self) -> Dict[str, float]:
        """Create sample evaluation metrics."""
        return {
            "silhouette": 0.75,
            "davies_bouldin": 0.5,
            "calinski_harabasz": 100.0,
        }

    @pytest.fixture
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Create sample hyperparameters."""
        return {
            "n_clusters": 4,
            "random_state": 42,
            "n_init": 10,
            "max_iter": 300,
        }

    def test_clustering_result_init(
        self,
        sample_cluster_labels,
        sample_metrics,
        sample_hyperparameters,
    ):
        """Test ClusteringResult initialization."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        assert result.experiment_id == "test_experiment"
        assert result.algorithm == "kmeans"
        assert len(result.cluster_labels) == len(sample_cluster_labels)
        assert result.hyperparameters == sample_hyperparameters
        assert result.metrics == sample_metrics

    def test_clustering_result_with_centroids(
        self,
        sample_cluster_labels,
        sample_centroids,
        sample_metrics,
        sample_hyperparameters,
    ):
        """Test ClusteringResult with centroids."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            centroids=sample_centroids,
        )

        assert result.centroids is not None
        assert result.centroids.shape[0] == 4
        assert result.centroids.shape[1] == 50

    def test_clustering_result_with_uncertainty(
        self,
        sample_cluster_labels,
        sample_uncertainty_scores,
        sample_metrics,
        sample_hyperparameters,
    ):
        """Test ClusteringResult with uncertainty scores."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="gmm",
            cluster_labels=sample_cluster_labels,
            hyperparameters={"n_components": 4, "random_state": 42},
            metrics=sample_metrics,
            uncertainty_scores=sample_uncertainty_scores,
        )

        assert result.uncertainty_scores is not None
        assert len(result.uncertainty_scores) == len(sample_cluster_labels)

    def test_clustering_result_invalid_algorithm(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test ClusteringResult with invalid algorithm."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            ClusteringResult(
                experiment_id="test_experiment",
                algorithm="invalid_algorithm",
                cluster_labels=sample_cluster_labels,
                hyperparameters=sample_hyperparameters,
                metrics=sample_metrics,
            )

    def test_clustering_result_invalid_experiment_id(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test ClusteringResult with invalid experiment ID."""
        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            ClusteringResult(
                experiment_id="",
                algorithm="kmeans",
                cluster_labels=sample_cluster_labels,
                hyperparameters=sample_hyperparameters,
                metrics=sample_metrics,
            )

    def test_clustering_result_invalid_cluster_labels(
        self, sample_metrics, sample_hyperparameters
    ):
        """Test ClusteringResult with invalid cluster labels."""
        with pytest.raises(ValueError, match="cluster_labels must be 1D"):
            ClusteringResult(
                experiment_id="test_experiment",
                algorithm="kmeans",
                cluster_labels=torch.randn(10, 5),  # 2D instead of 1D
                hyperparameters=sample_hyperparameters,
                metrics=sample_metrics,
            )

    def test_clustering_result_missing_metrics(
        self, sample_cluster_labels, sample_hyperparameters
    ):
        """Test ClusteringResult with missing required metrics."""
        with pytest.raises(ValueError, match="Missing required metric"):
            ClusteringResult(
                experiment_id="test_experiment",
                algorithm="kmeans",
                cluster_labels=sample_cluster_labels,
                hyperparameters=sample_hyperparameters,
                metrics={"silhouette": 0.75},  # Missing other metrics
            )

    def test_clustering_result_invalid_silhouette_range(
        self, sample_cluster_labels, sample_hyperparameters
    ):
        """Test ClusteringResult with silhouette score out of range."""
        with pytest.raises(ValueError, match="Silhouette score must be between -1 and 1"):
            ClusteringResult(
                experiment_id="test_experiment",
                algorithm="kmeans",
                cluster_labels=sample_cluster_labels,
                hyperparameters=sample_hyperparameters,
                metrics={
                    "silhouette": 2.0,  # Out of range
                    "davies_bouldin": 0.5,
                    "calinski_harabasz": 100.0,
                },
            )

    def test_clustering_result_get_cluster_summary(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test get_cluster_summary method."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        summary = result.get_cluster_summary()

        # Validate summary
        assert isinstance(summary, dict)
        assert len(summary) == 4  # 4 clusters
        assert "cluster_0" in summary
        assert "cluster_1" in summary
        assert summary["cluster_0"]["size"] >= 1
        assert 0 <= summary["cluster_0"]["percentage"] <= 100

    def test_clustering_result_get_result_summary(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test get_result_summary method."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        summary = result.get_result_summary()

        # Validate summary
        assert summary["experiment_id"] == "test_experiment"
        assert summary["algorithm"] == "kmeans"
        assert summary["n_clusters"] == 4
        assert summary["n_samples"] == len(sample_cluster_labels)
        assert "hyperparameters" in summary
        assert "metrics" in summary
        assert "cluster_summary" in summary

    def test_clustering_result_save_and_load(
        self, tmp_path, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test save and load methods."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        # Save result
        save_path = tmp_path / "clustering_result.pt"
        success = result.save_result(save_path)
        assert success
        assert save_path.exists()

        # Load result
        loaded_result = ClusteringResult.load_result(save_path)

        # Validate loaded result
        assert loaded_result.experiment_id == result.experiment_id
        assert loaded_result.algorithm == result.algorithm
        assert torch.equal(loaded_result.cluster_labels, result.cluster_labels)
        assert loaded_result.metrics == result.metrics

    def test_clustering_result_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file."""
        save_path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            ClusteringResult.load_result(save_path)

    def test_clustering_result_repr(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test __repr__ method."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        repr_str = repr(result)
        assert "ClusteringResult" in repr_str
        assert "test_experiment" in repr_str
        assert "kmeans" in repr_str

    def test_clustering_result_str(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test __str__ method."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        str_repr = str(result)
        assert "test_experiment" in str_repr
        assert "kmeans" in str_repr
        assert "Silhouette" in str_repr

    def test_clustering_result_embedding_ids(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test ClusteringResult with embedding IDs."""
        embedding_ids = [f"img_{i:03d}" for i in range(len(sample_cluster_labels))]

        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            embedding_ids=embedding_ids,
        )

        assert len(result.embedding_ids) == len(sample_cluster_labels)
        assert result.embedding_ids[0] == "img_000"

    def test_clustering_result_processing_time(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test ClusteringResult with processing time."""
        result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            processing_time=1.234,
        )

        assert result.processing_time == 1.234

    def test_create_clustering_result_from_algorithm(
        self, sample_cluster_labels, sample_metrics, sample_hyperparameters
    ):
        """Test create_clustering_result_from_algorithm factory function."""
        result = create_clustering_result_from_algorithm(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=sample_cluster_labels,
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
        )

        assert isinstance(result, ClusteringResult)
        assert result.experiment_id == "test_experiment"
        assert result.algorithm == "kmeans"


class TestClusteringIntegration:
    """Integration tests for clustering pipeline."""

    @pytest.fixture
    def sample_embedding_vectors(self) -> List[EmbeddingVector]:
        """Create sample embedding vectors for testing."""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_features = 256

        embeddings_array = rng.normal(scale=0.5, size=(n_samples, n_features)).astype(
            np.float32
        )

        embedding_vectors = []
        for i in range(n_samples):
            emb_tensor = torch.tensor(embeddings_array[i], dtype=torch.float32)
            embedding_vectors.append(
                EmbeddingVector(
                    image_id=f"img_{i:03d}",
                    embedding=emb_tensor,
                    metadata={"view": "CC" if i % 2 == 0 else "MLO"},
                )
            )

        return embedding_vectors

    def test_end_to_end_clustering_pipeline(self, sample_embedding_vectors, tmp_path):
        """Test end-to-end clustering pipeline."""
        # Configure clustering
        config = {
            "algorithm": "kmeans",
            "pca_dimensions": 20,
            "n_clusters": 4,
            "seed": 42,
        }

        # Perform clustering
        result = cluster_embeddings(sample_embedding_vectors, config)

        # Validate result
        assert result is not None
        assert len(result.cluster_labels) == len(sample_embedding_vectors)
        assert result.metrics is not None

        # Save result
        save_path = tmp_path / "clustering_result.pt"
        success = result.save_result(save_path)
        assert success

        # Load result
        loaded_result = ClusteringResult.load_result(save_path)
        assert torch.equal(loaded_result.cluster_labels, result.cluster_labels)

    def test_clustering_pipeline_all_algorithms(self, sample_embedding_vectors):
        """Test clustering pipeline with all algorithms."""
        algorithms = ["kmeans", "gmm", "hdbscan", "agglomerative"]

        for algorithm in algorithms:
            config = {
                "algorithm": algorithm,
                "pca_dimensions": 20,
                "n_clusters": 3,
                "seed": 42,
            }

            result = cluster_embeddings(sample_embedding_vectors, config)

            # Validate result
            assert result is not None, f"Failed for algorithm: {algorithm}"
            assert result.algorithm == algorithm
            assert len(result.cluster_labels) == len(sample_embedding_vectors)

    def test_clustering_pipeline_different_pca_dimensions(self, sample_embedding_vectors):
        """Test clustering pipeline with different PCA dimensions."""
        pca_dimensions_list = [10, 20, 50]

        for pca_dimensions in pca_dimensions_list:
            config = {
                "algorithm": "kmeans",
                "pca_dimensions": pca_dimensions,
                "n_clusters": 3,
                "seed": 42,
            }

            result = cluster_embeddings(sample_embedding_vectors, config)

            # Validate result
            assert result is not None
            assert len(result.cluster_labels) == len(sample_embedding_vectors)

    def test_clustering_result_consistency(self, sample_embedding_vectors):
        """Test consistency of clustering results."""
        config = {
            "algorithm": "kmeans",
            "pca_dimensions": 20,
            "n_clusters": 4,
            "seed": 42,
        }

        # Run clustering multiple times
        results = []
        for _ in range(3):
            result = cluster_embeddings(sample_embedding_vectors, config)
            results.append(result)

        # Validate consistency
        for i in range(1, len(results)):
            assert torch.equal(
                results[0].cluster_labels, results[i].cluster_labels
            ), "Results not consistent"
            assert results[0].metrics == results[i].metrics

    def test_clustering_with_metadata_preservation(self, sample_embedding_vectors):
        """Test that clustering preserves embedding metadata."""
        config = {
            "algorithm": "kmeans",
            "pca_dimensions": 20,
            "n_clusters": 4,
            "seed": 42,
        }

        result = cluster_embeddings(sample_embedding_vectors, config)

        # Validate embedding IDs are preserved
        assert len(result.embedding_ids) == len(sample_embedding_vectors)
        assert result.embedding_ids[0] == sample_embedding_vectors[0].image_id
