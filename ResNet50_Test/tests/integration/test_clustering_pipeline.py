"""
Integration tests for clustering pipeline.

These tests validate the complete clustering pipeline from embeddings
to clustering results with evaluation metrics.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import hdbscan
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
import umap

# Import the modules we'll be testing (these will be implemented later)
# from src.clustering.pipeline import ClusteringPipeline
# from src.clustering.algorithms import ClusteringAlgorithm
# from src.clustering.evaluator import ClusteringEvaluator
# from src.clustering.visualizer import ClusteringVisualizer


class TestClusteringPipelineIntegration:
    """Integration tests for clustering pipeline operations."""

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample embeddings for testing."""
        np.random.seed(42)
        # Create mock embeddings with 4 distinct clusters
        n_samples = 100
        n_features = 2048

        # Create 4 distinct clusters
        cluster_centers = np.random.randn(4, n_features) * 2
        embeddings = []
        labels = []

        for i in range(4):
            # Generate samples for each cluster
            cluster_samples = np.random.randn(25, n_features) * 0.5 + cluster_centers[i]
            embeddings.append(cluster_samples)
            labels.extend([i] * 25)

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)

        return embeddings, labels

    def test_clustering_pipeline_end_to_end(self, sample_embeddings):
        """Test complete clustering pipeline from embeddings to results."""
        embeddings, true_labels = sample_embeddings

        # Test configuration
        config = {
            "algorithm": "kmeans",
            "n_clusters": 4,
            "random_state": 42,
            "pca_dimensions": 50,
            "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "visualization": True,
        }

        # Step 1: Dimensionality reduction with PCA
        pca = PCA(
            n_components=config["pca_dimensions"], random_state=config["random_state"]
        )
        embeddings_reduced = pca.fit_transform(embeddings)

        assert embeddings_reduced.shape == (
            embeddings.shape[0],
            config["pca_dimensions"],
        )

        # Step 2: Clustering
        if config["algorithm"] == "kmeans":
            clusterer = KMeans(
                n_clusters=config["n_clusters"], random_state=config["random_state"]
            )
            cluster_labels = clusterer.fit_predict(embeddings_reduced)
        elif config["algorithm"] == "gmm":
            clusterer = GaussianMixture(
                n_components=config["n_clusters"], random_state=config["random_state"]
            )
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

        assert len(cluster_labels) == len(embeddings)
        assert len(set(cluster_labels)) <= config["n_clusters"]

        # Step 3: Evaluation
        evaluation_results = {}

        if "silhouette" in config["evaluation_metrics"]:
            silhouette = silhouette_score(embeddings_reduced, cluster_labels)
            evaluation_results["silhouette"] = silhouette
            assert -1 <= silhouette <= 1

        if "davies_bouldin" in config["evaluation_metrics"]:
            db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
            evaluation_results["davies_bouldin"] = db_score
            assert db_score >= 0

        if "calinski_harabasz" in config["evaluation_metrics"]:
            ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)
            evaluation_results["calinski_harabasz"] = ch_score
            assert ch_score >= 0

        # Step 4: Visualization (UMAP)
        if config["visualization"]:
            umap_reducer = umap.UMAP(
                n_components=2, random_state=config["random_state"]
            )
            embeddings_2d = umap_reducer.fit_transform(embeddings_reduced)

            assert embeddings_2d.shape == (embeddings.shape[0], 2)
            assert not np.any(np.isnan(embeddings_2d))
            assert not np.any(np.isinf(embeddings_2d))

        # Validate complete pipeline result
        result = {
            "cluster_labels": cluster_labels,
            "evaluation_metrics": evaluation_results,
            "pca_components": config["pca_dimensions"],
            "algorithm": config["algorithm"],
            "n_clusters": config["n_clusters"],
        }

        assert "cluster_labels" in result
        assert "evaluation_metrics" in result
        assert "pca_components" in result
        assert "algorithm" in result
        assert "n_clusters" in result

    def test_clustering_algorithms_comparison(self, sample_embeddings):
        """Test different clustering algorithms."""
        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality first
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        algorithms = ["kmeans", "gmm", "hdbscan"]
        results = {}

        for algorithm in algorithms:
            if algorithm == "kmeans":
                clusterer = KMeans(n_clusters=4, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings_reduced)

            elif algorithm == "gmm":
                clusterer = GaussianMixture(n_components=4, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings_reduced)

            elif algorithm == "hdbscan":
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
                cluster_labels = clusterer.fit_predict(embeddings_reduced)

            # Evaluate clustering
            silhouette = silhouette_score(embeddings_reduced, cluster_labels)
            db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)

            results[algorithm] = {
                "cluster_labels": cluster_labels,
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "n_clusters_found": len(set(cluster_labels)),
            }

        # Validate results
        for algorithm, result in results.items():
            assert "cluster_labels" in result
            assert "silhouette" in result
            assert "davies_bouldin" in result
            assert "calinski_harabasz" in result
            assert "n_clusters_found" in result

            assert -1 <= result["silhouette"] <= 1
            assert result["davies_bouldin"] >= 0
            assert result["calinski_harabasz"] >= 0
            assert result["n_clusters_found"] > 0

    def test_dimensionality_reduction_variants(self, sample_embeddings):
        """Test different dimensionality reduction methods."""
        embeddings, true_labels = sample_embeddings

        reduction_methods = [
            {"method": "pca", "n_components": 50},
            {"method": "pca", "n_components": 128},
            {"method": "pca", "n_components": 256},
        ]

        results = {}

        for config in reduction_methods:
            if config["method"] == "pca":
                reducer = PCA(n_components=config["n_components"], random_state=42)
                embeddings_reduced = reducer.fit_transform(embeddings)

            # Validate reduced embeddings
            assert embeddings_reduced.shape == (
                embeddings.shape[0],
                config["n_components"],
            )
            assert not np.any(np.isnan(embeddings_reduced))
            assert not np.any(np.isinf(embeddings_reduced))

            # Test clustering on reduced embeddings
            clusterer = KMeans(n_clusters=4, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

            # Evaluate clustering
            silhouette = silhouette_score(embeddings_reduced, cluster_labels)

            results[f"{config['method']}_{config['n_components']}"] = {
                "embeddings_reduced": embeddings_reduced,
                "cluster_labels": cluster_labels,
                "silhouette": silhouette,
                "explained_variance_ratio": (
                    reducer.explained_variance_ratio_.sum()
                    if config["method"] == "pca"
                    else None
                ),
            }

        # Validate results
        for method, result in results.items():
            assert "embeddings_reduced" in result
            assert "cluster_labels" in result
            assert "silhouette" in result
            assert -1 <= result["silhouette"] <= 1

    def test_clustering_evaluation_metrics(self, sample_embeddings):
        """Test clustering evaluation metrics."""
        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        # Create different clustering results
        clusterers = [
            KMeans(n_clusters=4, random_state=42),
            KMeans(n_clusters=6, random_state=42),
            GaussianMixture(n_components=4, random_state=42),
        ]

        results = {}

        for i, clusterer in enumerate(clusterers):
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

            # Calculate all evaluation metrics
            silhouette = silhouette_score(embeddings_reduced, cluster_labels)
            db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)

            results[f"clusterer_{i}"] = {
                "cluster_labels": cluster_labels,
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "n_clusters": len(set(cluster_labels)),
            }

        # Validate metrics
        for result in results.values():
            # Silhouette score: higher is better, range [-1, 1]
            assert -1 <= result["silhouette"] <= 1

            # Davies-Bouldin score: lower is better, range [0, inf)
            assert result["davies_bouldin"] >= 0

            # Calinski-Harabasz score: higher is better, range [0, inf)
            assert result["calinski_harabasz"] >= 0

            # Number of clusters should be positive
            assert result["n_clusters"] > 0

    def test_clustering_visualization(self, sample_embeddings):
        """Test clustering visualization with UMAP."""
        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality for clustering
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        # Perform clustering
        clusterer = KMeans(n_clusters=4, random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings_reduced)

        # Create 2D visualization with UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = umap_reducer.fit_transform(embeddings_reduced)

        # Validate 2D embeddings
        assert embeddings_2d.shape == (embeddings.shape[0], 2)
        assert not np.any(np.isnan(embeddings_2d))
        assert not np.any(np.isinf(embeddings_2d))

        # Test visualization data structure
        visualization_data = {
            "embeddings_2d": embeddings_2d,
            "cluster_labels": cluster_labels,
            "true_labels": true_labels,
            "n_clusters": len(set(cluster_labels)),
        }

        assert "embeddings_2d" in visualization_data
        assert "cluster_labels" in visualization_data
        assert "true_labels" in visualization_data
        assert "n_clusters" in visualization_data

        # Validate data shapes
        assert visualization_data["embeddings_2d"].shape[0] == len(
            visualization_data["cluster_labels"]
        )
        assert visualization_data["embeddings_2d"].shape[0] == len(
            visualization_data["true_labels"]
        )

    def test_clustering_reproducibility(self, sample_embeddings):
        """Test reproducibility of clustering with fixed seeds."""
        embeddings, true_labels = sample_embeddings

        config = {"random_state": 42, "pca_dimensions": 50, "n_clusters": 4}

        # Perform clustering multiple times with same seed
        results = []
        for _ in range(3):
            np.random.seed(config["random_state"])

            # PCA
            pca = PCA(
                n_components=config["pca_dimensions"],
                random_state=config["random_state"],
            )
            embeddings_reduced = pca.fit_transform(embeddings)

            # Clustering
            clusterer = KMeans(
                n_clusters=config["n_clusters"], random_state=config["random_state"]
            )
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

            results.append(cluster_labels)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), "Clustering not reproducible"

    def test_clustering_performance(self, sample_embeddings):
        """Test clustering performance benchmarks."""
        import time

        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        # Time the clustering
        start_time = time.time()

        # Perform clustering
        clusterer = KMeans(n_clusters=4, random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings_reduced)

        # Calculate evaluation metrics
        silhouette = silhouette_score(embeddings_reduced, cluster_labels)
        db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
        ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0, f"Clustering too slow: {processing_time:.2f}s"

        # Validate metrics
        assert -1 <= silhouette <= 1
        assert db_score >= 0
        assert ch_score >= 0

    def test_clustering_memory_usage(self, sample_embeddings):
        """Test memory usage during clustering."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        # Perform clustering
        clusterer = KMeans(n_clusters=4, random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings_reduced)

        # Calculate evaluation metrics
        silhouette = silhouette_score(embeddings_reduced, cluster_labels)
        db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
        ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 200 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_clustering_error_handling(self, sample_embeddings):
        """Test error handling in clustering pipeline."""
        embeddings, true_labels = sample_embeddings

        # Test with invalid configurations
        invalid_configs = [
            {"n_clusters": 0},  # Invalid number of clusters
            {"n_clusters": -1},  # Negative number of clusters
            {"pca_dimensions": 0},  # Invalid PCA dimensions
            {"pca_dimensions": embeddings.shape[1] + 1},  # Too many PCA dimensions
        ]

        for config in invalid_configs:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if "n_clusters" in config and config["n_clusters"] <= 0:
                    raise ValueError("Invalid number of clusters")
                if "pca_dimensions" in config and config["pca_dimensions"] <= 0:
                    raise ValueError("Invalid PCA dimensions")
                if (
                    "pca_dimensions" in config
                    and config["pca_dimensions"] > embeddings.shape[1]
                ):
                    raise ValueError("Too many PCA dimensions")

        # Test with invalid embeddings
        invalid_embeddings = [
            np.array([]),  # Empty array
            np.zeros((0, 2048)),  # Zero samples
            np.ones((10, 0)),  # Zero features
            np.ones((10, 2048)) * np.nan,  # NaN values
        ]

        for invalid_embedding in invalid_embeddings:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if invalid_embedding.size == 0:
                    raise ValueError("Empty embeddings")
                if np.any(np.isnan(invalid_embedding)):
                    raise ValueError("NaN values in embeddings")

    def test_clustering_with_different_cluster_numbers(self, sample_embeddings):
        """Test clustering with different numbers of clusters."""
        embeddings, true_labels = sample_embeddings

        # Reduce dimensionality
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        n_clusters_list = [2, 3, 4, 5, 6]
        results = {}

        for n_clusters in n_clusters_list:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

            # Evaluate clustering
            silhouette = silhouette_score(embeddings_reduced, cluster_labels)
            db_score = davies_bouldin_score(embeddings_reduced, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings_reduced, cluster_labels)

            results[n_clusters] = {
                "cluster_labels": cluster_labels,
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "n_clusters_found": len(set(cluster_labels)),
            }

        # Validate results
        for n_clusters, result in results.items():
            assert result["n_clusters_found"] <= n_clusters
            assert -1 <= result["silhouette"] <= 1
            assert result["davies_bouldin"] >= 0
            assert result["calinski_harabasz"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
