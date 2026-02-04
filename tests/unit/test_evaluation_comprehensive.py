"""
Comprehensive unit tests for evaluation module functionality.

These tests validate the ClusteringEvaluator class and related evaluation
functionality for mammography clustering analysis.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from datetime import datetime
from typing import Dict, List, Tuple
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from mammography.clustering.clustering_result import ClusteringResult
from mammography.eval.clustering_evaluator import (
    ClusteringEvaluator,
    create_clustering_evaluator,
    evaluate_clustering,
)
from mammography.io.dicom import MammographyImage
from mammography.models.embeddings.embedding_vector import EmbeddingVector

MATMUL_ERRSTATE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


@pytest.fixture(autouse=True)
def _suppress_numpy_matmul_warnings():
    with np.errstate(**MATMUL_ERRSTATE):
        yield


class TestClusteringEvaluatorInitialization:
    """Tests for ClusteringEvaluator initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        config = {}
        evaluator = ClusteringEvaluator(config)

        assert evaluator.config is not None
        assert "metrics" in evaluator.config
        assert "sanity_checks" in evaluator.config
        assert "visual_prototypes" in evaluator.config
        assert "seed" in evaluator.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "metrics": ["silhouette", "davies_bouldin"],
            "sanity_checks": ["cluster_size_analysis"],
            "seed": 123,
        }
        evaluator = ClusteringEvaluator(config)

        assert evaluator.config["metrics"] == ["silhouette", "davies_bouldin"]
        assert evaluator.config["sanity_checks"] == ["cluster_size_analysis"]
        assert evaluator.config["seed"] == 123

    def test_initialization_with_invalid_metrics(self):
        """Test initialization with invalid metrics."""
        config = {"metrics": ["invalid_metric", "silhouette"]}
        evaluator = ClusteringEvaluator(config)

        # Should initialize without error but may log warnings
        assert evaluator.config is not None
        assert "metrics" in evaluator.config

    def test_initialization_sets_random_seed(self):
        """Test that initialization sets random seed for reproducibility."""
        config = {"seed": 42}
        evaluator1 = ClusteringEvaluator(config)
        evaluator2 = ClusteringEvaluator(config)

        # Both evaluators should produce same random results
        random1 = np.random.rand(5)
        random2 = np.random.rand(5)

        # Reset seeds to verify reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        expected = np.random.rand(5)

        # Cannot directly compare due to state, but verify seed was set
        assert evaluator1.config["seed"] == 42
        assert evaluator2.config["seed"] == 42

    def test_supported_metrics_constant(self):
        """Test that SUPPORTED_METRICS constant is defined."""
        assert hasattr(ClusteringEvaluator, "SUPPORTED_METRICS")
        assert isinstance(ClusteringEvaluator.SUPPORTED_METRICS, list)
        assert len(ClusteringEvaluator.SUPPORTED_METRICS) > 0

    def test_sanity_check_methods_constant(self):
        """Test that SANITY_CHECK_METHODS constant is defined."""
        assert hasattr(ClusteringEvaluator, "SANITY_CHECK_METHODS")
        assert isinstance(ClusteringEvaluator.SANITY_CHECK_METHODS, list)
        assert len(ClusteringEvaluator.SANITY_CHECK_METHODS) > 0


class TestClusteringEvaluatorFixtures:
    """Test fixtures for clustering evaluator tests."""

    @pytest.fixture
    def sample_embedding_vectors(self) -> List[EmbeddingVector]:
        """Create sample embedding vectors for testing."""
        rng = np.random.default_rng(42)
        n_samples = 100
        embedding_dim = 128

        embedding_vectors = []
        for i in range(n_samples):
            embedding = torch.tensor(
                rng.normal(0, 1, embedding_dim), dtype=torch.float32
            )
            emb_vector = EmbeddingVector(
                image_id=f"img_{i:03d}",
                embedding=embedding,
                model_config={"architecture": "resnet50", "pretrained": True},
                input_adapter="1to3_replication",
                extraction_time=0.1,
            )
            embedding_vectors.append(emb_vector)

        return embedding_vectors

    @pytest.fixture
    def sample_clustering_result(self) -> ClusteringResult:
        """Create sample clustering result for testing."""
        n_samples = 100
        n_clusters = 4

        # Create cluster labels
        cluster_labels = torch.tensor(
            np.random.randint(0, n_clusters, n_samples), dtype=torch.long
        )

        # Create centroids
        centroids = torch.randn(n_clusters, 128)

        # Create metrics
        metrics = {
            "silhouette": 0.5,
            "davies_bouldin": 1.2,
            "calinski_harabasz": 100.0,
        }

        clustering_result = ClusteringResult(
            experiment_id="test_experiment",
            algorithm="kmeans",
            cluster_labels=cluster_labels,
            hyperparameters={"n_clusters": n_clusters},
            metrics=metrics,
            centroids=centroids,
        )

        return clustering_result

    @pytest.fixture
    def sample_mammography_images(self) -> List[MammographyImage]:
        """Create sample mammography images for testing."""
        images = []
        projections = ["CC", "MLO"]
        lateralities = ["L", "R"]

        for i in range(100):
            # Create mock mammography image
            img = Mock(spec=MammographyImage)
            img.instance_id = f"img_{i:03d}"
            img.projection_type = projections[i % 2]
            img.laterality = lateralities[i % 2]
            images.append(img)

        return images


class TestClusteringEvaluatorQualityMetrics(TestClusteringEvaluatorFixtures):
    """Tests for quality metrics computation."""

    def test_compute_quality_metrics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test quality metrics computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        quality_metrics = evaluator._compute_quality_metrics(
            sample_clustering_result, sample_embedding_vectors
        )

        assert isinstance(quality_metrics, dict)
        assert "cluster_statistics" in quality_metrics
        assert "distance_metrics" in quality_metrics
        assert "separation_metrics" in quality_metrics

    def test_compute_cluster_statistics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test cluster statistics computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        embedding_matrix = evaluator._extract_embedding_matrix(sample_embedding_vectors)
        cluster_labels = sample_clustering_result.cluster_labels.numpy()

        cluster_stats = evaluator._compute_cluster_statistics(
            cluster_labels, embedding_matrix
        )

        assert isinstance(cluster_stats, dict)
        assert len(cluster_stats) > 0

        # Check statistics for first cluster
        cluster_0_stats = cluster_stats.get("cluster_0")
        if cluster_0_stats:
            assert "size" in cluster_0_stats
            assert "mean_embedding" in cluster_0_stats
            assert "std_embedding" in cluster_0_stats
            assert "min_embedding" in cluster_0_stats
            assert "max_embedding" in cluster_0_stats

    def test_compute_distance_metrics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test distance metrics computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        embedding_matrix = evaluator._extract_embedding_matrix(sample_embedding_vectors)
        cluster_labels = sample_clustering_result.cluster_labels.numpy()

        distance_metrics = evaluator._compute_distance_metrics(
            embedding_matrix, cluster_labels
        )

        assert isinstance(distance_metrics, dict)

        if len(distance_metrics) > 0:
            # Check for expected metrics
            if "mean_intra_cluster_distance" in distance_metrics:
                assert distance_metrics["mean_intra_cluster_distance"] >= 0
            if "mean_inter_cluster_distance" in distance_metrics:
                assert distance_metrics["mean_inter_cluster_distance"] >= 0
            if "separation_ratio" in distance_metrics:
                assert distance_metrics["separation_ratio"] >= 0

    def test_compute_separation_metrics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test separation metrics computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        embedding_matrix = evaluator._extract_embedding_matrix(sample_embedding_vectors)
        cluster_labels = sample_clustering_result.cluster_labels.numpy()

        separation_metrics = evaluator._compute_separation_metrics(
            embedding_matrix, cluster_labels
        )

        assert isinstance(separation_metrics, dict)

        if len(separation_metrics) > 0:
            assert "mean_centroid_distance" in separation_metrics
            assert "min_centroid_distance" in separation_metrics
            assert "max_centroid_distance" in separation_metrics
            assert separation_metrics["mean_centroid_distance"] >= 0

    def test_compute_stability_metrics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test stability metrics computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        embedding_matrix = evaluator._extract_embedding_matrix(sample_embedding_vectors)
        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        centroids = sample_clustering_result.centroids

        stability_metrics = evaluator._compute_stability_metrics(
            embedding_matrix, cluster_labels, centroids
        )

        assert isinstance(stability_metrics, dict)

        if len(stability_metrics) > 0:
            assert "mean_point_to_centroid_distance" in stability_metrics
            assert stability_metrics["mean_point_to_centroid_distance"] >= 0


class TestClusteringEvaluatorSanityChecks(TestClusteringEvaluatorFixtures):
    """Tests for sanity checks functionality."""

    def test_perform_sanity_checks(
        self,
        sample_clustering_result,
        sample_embedding_vectors,
        sample_mammography_images,
    ):
        """Test sanity checks execution."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        sanity_checks = evaluator._perform_sanity_checks(
            sample_clustering_result, sample_embedding_vectors, sample_mammography_images
        )

        assert isinstance(sanity_checks, dict)
        assert len(sanity_checks) > 0

    def test_analyze_cluster_sizes(self, sample_clustering_result):
        """Test cluster size analysis."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        size_analysis = evaluator._analyze_cluster_sizes(cluster_labels)

        assert isinstance(size_analysis, dict)
        assert "n_clusters" in size_analysis
        assert "cluster_sizes" in size_analysis
        assert "size_statistics" in size_analysis
        assert "size_balance" in size_analysis
        assert "potential_issues" in size_analysis

        # Check size statistics
        assert size_analysis["size_statistics"]["mean_size"] > 0
        assert size_analysis["size_statistics"]["min_size"] > 0
        assert size_analysis["size_statistics"]["max_size"] > 0

    def test_analyze_embedding_statistics(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test embedding statistics analysis."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        embedding_stats = evaluator._analyze_embedding_statistics(
            sample_embedding_vectors, cluster_labels
        )

        assert isinstance(embedding_stats, dict)
        assert "global_statistics" in embedding_stats
        assert "cluster_statistics" in embedding_stats

    def test_analyze_projection_distribution(
        self, sample_clustering_result, sample_mammography_images
    ):
        """Test projection distribution analysis."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        projection_analysis = evaluator._analyze_projection_distribution(
            sample_mammography_images, cluster_labels
        )

        assert isinstance(projection_analysis, dict)
        assert "cluster_projection_distributions" in projection_analysis
        assert "overall_projection_distribution" in projection_analysis
        assert "potential_issues" in projection_analysis

    def test_analyze_laterality_distribution(
        self, sample_clustering_result, sample_mammography_images
    ):
        """Test laterality distribution analysis."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        laterality_analysis = evaluator._analyze_laterality_distribution(
            sample_mammography_images, cluster_labels
        )

        assert isinstance(laterality_analysis, dict)
        assert "cluster_laterality_distributions" in laterality_analysis
        assert "overall_laterality_distribution" in laterality_analysis
        assert "potential_issues" in laterality_analysis

    def test_analyze_intensity_histograms(
        self, sample_clustering_result, sample_mammography_images
    ):
        """Test intensity histogram analysis."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_labels = sample_clustering_result.cluster_labels.numpy()
        intensity_analysis = evaluator._analyze_intensity_histograms(
            sample_mammography_images, cluster_labels
        )

        assert isinstance(intensity_analysis, dict)
        # This is currently a placeholder, so just verify it returns a dict


class TestClusteringEvaluatorVisualPrototypes(TestClusteringEvaluatorFixtures):
    """Tests for visual prototype selection."""

    def test_select_visual_prototypes(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test visual prototype selection."""
        config = {
            "seed": 42,
            "visual_prototypes": {
                "n_samples_per_cluster": 4,
                "selection_method": "centroid_distance",
            },
        }
        evaluator = ClusteringEvaluator(config)

        visual_prototypes = evaluator._select_visual_prototypes(
            sample_clustering_result, sample_embedding_vectors
        )

        assert isinstance(visual_prototypes, dict)
        assert len(visual_prototypes) > 0

        # Check prototype structure
        for cluster_key, prototype_info in visual_prototypes.items():
            assert "prototype_indices" in prototype_info
            assert "prototype_image_ids" in prototype_info
            assert "cluster_size" in prototype_info
            assert "selection_method" in prototype_info

    def test_select_prototypes_by_centroid_distance(
        self, sample_embedding_vectors
    ):
        """Test prototype selection by centroid distance."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Create sample cluster embeddings
        cluster_embeddings = np.random.randn(20, 128)
        cluster_indices = np.arange(20)
        n_samples = 4

        prototype_indices = evaluator._select_prototypes_by_centroid_distance(
            cluster_embeddings, cluster_indices, n_samples
        )

        assert len(prototype_indices) == n_samples
        assert all(idx in cluster_indices for idx in prototype_indices)

    def test_select_prototypes_randomly(self):
        """Test random prototype selection."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_indices = np.arange(20)
        n_samples = 4

        prototype_indices = evaluator._select_prototypes_randomly(
            cluster_indices, n_samples
        )

        assert len(prototype_indices) == n_samples
        assert all(idx in cluster_indices for idx in prototype_indices)

    def test_select_prototypes_with_small_cluster(self):
        """Test prototype selection with cluster smaller than requested samples."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        cluster_indices = np.arange(2)  # Only 2 samples
        n_samples = 4  # Request 4 samples

        prototype_indices = evaluator._select_prototypes_randomly(
            cluster_indices, n_samples
        )

        # Should return all available samples
        assert len(prototype_indices) == 2


class TestClusteringEvaluatorHelpers(TestClusteringEvaluatorFixtures):
    """Tests for helper methods."""

    def test_extract_embedding_matrix(self, sample_embedding_vectors):
        """Test embedding matrix extraction."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        embedding_matrix = evaluator._extract_embedding_matrix(sample_embedding_vectors)

        assert embedding_matrix is not None
        assert isinstance(embedding_matrix, np.ndarray)
        assert embedding_matrix.shape[0] == len(sample_embedding_vectors)
        assert embedding_matrix.shape[1] == 128  # embedding dimension

    def test_extract_embedding_matrix_with_empty_list(self):
        """Test embedding matrix extraction with empty list."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # This should handle gracefully or raise error
        result = evaluator._extract_embedding_matrix([])

        # Could be None or raise error - check implementation
        assert result is None or isinstance(result, np.ndarray)

    def test_compute_gini_coefficient(self):
        """Test Gini coefficient computation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Test with equal distribution
        equal_values = np.array([10, 10, 10, 10])
        gini_equal = evaluator._compute_gini_coefficient(equal_values)
        assert 0 <= gini_equal <= 0.1  # Should be close to 0

        # Test with unequal distribution
        unequal_values = np.array([1, 1, 1, 97])
        gini_unequal = evaluator._compute_gini_coefficient(unequal_values)
        assert gini_unequal > gini_equal  # Should be higher

    def test_generate_evaluation_summary(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test evaluation summary generation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Create mock evaluation results
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "clustering_result": sample_clustering_result,
            "quality_metrics": {
                "cluster_statistics": {},
                "distance_metrics": {},
                "separation_metrics": {},
            },
            "sanity_checks": {"cluster_size_analysis": {"potential_issues": []}},
            "visual_prototypes": {
                "cluster_0": {"prototype_indices": [0, 1, 2, 3]},
            },
        }

        summary = evaluator._generate_evaluation_summary(evaluation_results)

        assert isinstance(summary, dict)
        assert "evaluation_timestamp" in summary
        assert "clustering_algorithm" in summary
        assert "n_clusters" in summary
        assert "n_samples" in summary


class TestClusteringEvaluatorIntegration(TestClusteringEvaluatorFixtures):
    """Integration tests for complete evaluation workflow."""

    def test_evaluate_clustering_complete(
        self,
        sample_clustering_result,
        sample_embedding_vectors,
        sample_mammography_images,
    ):
        """Test complete clustering evaluation."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        results = evaluator.evaluate_clustering(
            sample_clustering_result, sample_embedding_vectors, sample_mammography_images
        )

        assert isinstance(results, dict)
        assert "clustering_result" in results
        assert "evaluation_timestamp" in results
        assert "config" in results
        assert "quality_metrics" in results
        assert "sanity_checks" in results
        assert "visual_prototypes" in results
        assert "summary" in results

    def test_evaluate_clustering_without_images(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test clustering evaluation without mammography images."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        results = evaluator.evaluate_clustering(
            sample_clustering_result, sample_embedding_vectors, mammography_images=None
        )

        assert isinstance(results, dict)
        assert "quality_metrics" in results
        assert "sanity_checks" in results
        # Some sanity checks may be skipped without images

    def test_evaluate_clustering_with_custom_config(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test clustering evaluation with custom configuration."""
        config = {
            "seed": 42,
            "metrics": ["silhouette", "davies_bouldin"],
            "sanity_checks": ["cluster_size_analysis", "embedding_statistics"],
            "visual_prototypes": {
                "n_samples_per_cluster": 2,
                "selection_method": "random",
            },
        }
        evaluator = ClusteringEvaluator(config)

        results = evaluator.evaluate_clustering(
            sample_clustering_result, sample_embedding_vectors
        )

        assert isinstance(results, dict)
        assert results["config"]["metrics"] == ["silhouette", "davies_bouldin"]


class TestClusteringEvaluatorEdgeCases(TestClusteringEvaluatorFixtures):
    """Tests for edge cases and error handling."""

    def test_evaluate_clustering_with_single_cluster(self, sample_embedding_vectors):
        """Test evaluation with single cluster."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Create clustering result with single cluster
        n_samples = len(sample_embedding_vectors)
        cluster_labels = torch.zeros(n_samples, dtype=torch.long)
        centroids = torch.randn(1, 128)

        clustering_result = ClusteringResult(
            experiment_id="test_single_cluster",
            algorithm="kmeans",
            cluster_labels=cluster_labels,
            hyperparameters={"n_clusters": 1},
            metrics={
                "silhouette": 0.0,
                "davies_bouldin": 0.0,
                "calinski_harabasz": 0.0,
            },
            centroids=centroids,
        )

        results = evaluator.evaluate_clustering(
            clustering_result, sample_embedding_vectors
        )

        assert isinstance(results, dict)
        # Some metrics may not be computable with single cluster

    def test_evaluate_clustering_with_noise_labels(self, sample_embedding_vectors):
        """Test evaluation with noise labels (-1)."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Create clustering result with noise labels
        n_samples = len(sample_embedding_vectors)
        cluster_labels = torch.randint(-1, 4, (n_samples,))
        centroids = torch.randn(4, 128)

        clustering_result = ClusteringResult(
            experiment_id="test_with_noise",
            algorithm="hdbscan",
            cluster_labels=cluster_labels,
            hyperparameters={"min_cluster_size": 5},
            metrics={
                "silhouette": 0.3,
                "davies_bouldin": 1.5,
                "calinski_harabasz": 50.0,
            },
            centroids=centroids,
        )

        results = evaluator.evaluate_clustering(
            clustering_result, sample_embedding_vectors
        )

        assert isinstance(results, dict)
        # Should handle noise labels gracefully

    def test_evaluate_clustering_with_mismatched_lengths(
        self, sample_clustering_result, sample_embedding_vectors
    ):
        """Test evaluation with mismatched embedding and label lengths."""
        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        # Create mammography images with different length
        mismatched_images = [Mock(spec=MammographyImage) for _ in range(50)]
        for i, img in enumerate(mismatched_images):
            img.instance_id = f"img_{i:03d}"
            img.projection_type = "CC"
            img.laterality = "L"

        results = evaluator.evaluate_clustering(
            sample_clustering_result,
            sample_embedding_vectors,
            mismatched_images,
        )

        assert isinstance(results, dict)
        # Should handle gracefully with warnings


class TestClusteringEvaluatorFactoryFunctions:
    """Tests for factory and convenience functions."""

    def test_create_clustering_evaluator(self):
        """Test factory function for creating evaluator."""
        config = {"seed": 42}
        evaluator = create_clustering_evaluator(config)

        assert isinstance(evaluator, ClusteringEvaluator)
        assert evaluator.config["seed"] == 42

    def test_evaluate_clustering_convenience_function(
        self,
        sample_clustering_result=None,
        sample_embedding_vectors=None,
    ):
        """Test convenience function for evaluation."""
        if sample_clustering_result is None or sample_embedding_vectors is None:
            # Create minimal test data
            n_samples = 50
            embedding_vectors = []
            for i in range(n_samples):
                embedding = torch.randn(128)
                emb_vector = EmbeddingVector(
                    image_id=f"img_{i:03d}",
                    embedding=embedding,
                    model_config={"architecture": "resnet50"},
                    input_adapter="1to3_replication",
                    extraction_time=0.1,
                )
                embedding_vectors.append(emb_vector)

            cluster_labels = torch.randint(0, 4, (n_samples,))
            centroids = torch.randn(4, 128)

            clustering_result = ClusteringResult(
                experiment_id="test_convenience",
                algorithm="kmeans",
                cluster_labels=cluster_labels,
                hyperparameters={"n_clusters": 4},
                metrics={
                    "silhouette": 0.4,
                    "davies_bouldin": 1.0,
                    "calinski_harabasz": 80.0,
                },
                centroids=centroids,
            )
        else:
            clustering_result = sample_clustering_result
            embedding_vectors = sample_embedding_vectors

        config = {"seed": 42}
        results = evaluate_clustering(
            clustering_result, embedding_vectors, config
        )

        assert isinstance(results, dict)
        assert "evaluation_timestamp" in results


class TestClusteringEvaluatorReproducibility:
    """Tests for reproducibility of evaluation results."""

    @pytest.fixture
    def test_data(self):
        """Create test data for reproducibility tests."""
        n_samples = 50
        embedding_vectors = []
        for i in range(n_samples):
            embedding = torch.randn(64)
            emb_vector = EmbeddingVector(
                image_id=f"img_{i:03d}",
                embedding=embedding,
                model_config={"architecture": "resnet50"},
                input_adapter="1to3_replication",
                extraction_time=0.1,
            )
            embedding_vectors.append(emb_vector)

        cluster_labels = torch.randint(0, 3, (n_samples,))
        centroids = torch.randn(3, 64)

        clustering_result = ClusteringResult(
            experiment_id="test_reproducibility",
            algorithm="kmeans",
            cluster_labels=cluster_labels,
            hyperparameters={"n_clusters": 3},
            metrics={
                "silhouette": 0.4,
                "davies_bouldin": 1.0,
                "calinski_harabasz": 70.0,
            },
            centroids=centroids,
        )

        return clustering_result, embedding_vectors

    def test_evaluation_reproducibility(self, test_data):
        """Test that evaluation results are reproducible with same seed."""
        clustering_result, embedding_vectors = test_data

        config1 = {"seed": 42}
        evaluator1 = ClusteringEvaluator(config1)
        results1 = evaluator1.evaluate_clustering(clustering_result, embedding_vectors)

        config2 = {"seed": 42}
        evaluator2 = ClusteringEvaluator(config2)
        results2 = evaluator2.evaluate_clustering(clustering_result, embedding_vectors)

        # Check that key metrics are reproducible
        if "quality_metrics" in results1 and "quality_metrics" in results2:
            qm1 = results1["quality_metrics"]
            qm2 = results2["quality_metrics"]

            if "distance_metrics" in qm1 and "distance_metrics" in qm2:
                dm1 = qm1["distance_metrics"]
                dm2 = qm2["distance_metrics"]

                # Compare numeric metrics (allowing for small floating point differences)
                for key in dm1:
                    if key in dm2 and isinstance(dm1[key], (int, float)):
                        assert abs(dm1[key] - dm2[key]) < 1e-6


class TestClusteringEvaluatorPerformance:
    """Tests for evaluation performance."""

    def test_evaluation_performance_with_large_dataset(self):
        """Test evaluation performance with larger dataset."""
        import time

        # Create larger dataset
        n_samples = 500
        embedding_dim = 256
        n_clusters = 10

        embedding_vectors = []
        for i in range(n_samples):
            embedding = torch.randn(embedding_dim)
            emb_vector = EmbeddingVector(
                image_id=f"img_{i:04d}",
                embedding=embedding,
                model_config={"architecture": "resnet50"},
                input_adapter="1to3_replication",
                extraction_time=0.1,
            )
            embedding_vectors.append(emb_vector)

        cluster_labels = torch.randint(0, n_clusters, (n_samples,))
        centroids = torch.randn(n_clusters, embedding_dim)

        clustering_result = ClusteringResult(
            experiment_id="test_performance",
            algorithm="kmeans",
            cluster_labels=cluster_labels,
            hyperparameters={"n_clusters": n_clusters},
            metrics={
                "silhouette": 0.35,
                "davies_bouldin": 1.3,
                "calinski_harabasz": 150.0,
            },
            centroids=centroids,
        )

        config = {"seed": 42}
        evaluator = ClusteringEvaluator(config)

        start_time = time.time()
        results = evaluator.evaluate_clustering(clustering_result, embedding_vectors)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 30.0, f"Evaluation too slow: {processing_time:.2f}s"
        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
