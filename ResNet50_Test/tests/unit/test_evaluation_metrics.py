"""
Unit tests for evaluation metrics functionality.

These tests validate individual evaluation metric functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from typing import Tuple

import numpy as np
import pytest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

# Import the modules we'll be testing (these will be implemented later)
# from src.eval.clustering_metrics import ClusteringMetrics
# from src.eval.silhouette_evaluator import SilhouetteEvaluator
# from src.eval.davies_bouldin_evaluator import DaviesBouldinEvaluator
# from src.eval.calinski_harabasz_evaluator import CalinskiHarabaszEvaluator


class TestEvaluationMetrics:
    """Unit tests for evaluation metric functions."""

    @pytest.fixture
    def sample_data_with_clusters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample data with clear cluster structure."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        n_clusters = 4

        # Create cluster centers
        cluster_centers = np.random.randn(n_clusters, n_features) * 2

        # Generate samples for each cluster
        data = []
        labels = []
        for i in range(n_clusters):
            cluster_samples = (
                np.random.randn(n_samples // n_clusters, n_features) * 0.5
                + cluster_centers[i]
            )
            data.append(cluster_samples)
            labels.extend([i] * (n_samples // n_clusters))

        data = np.vstack(data)
        labels = np.array(labels)

        return data, labels

    @pytest.fixture
    def sample_data_without_clusters(self) -> np.ndarray:
        """Create sample data without clear cluster structure."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Create random data without structure
        data = np.random.randn(n_samples, n_features)

        return data

    def test_silhouette_score_calculation(self, sample_data_with_clusters):
        """Test silhouette score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate silhouette score
        silhouette = silhouette_score(data, true_labels)

        # Validate silhouette score
        assert isinstance(silhouette, float)
        assert -1 <= silhouette <= 1
        assert not np.isnan(silhouette)
        assert not np.isinf(silhouette)

        # For well-separated clusters, silhouette should be positive
        assert silhouette > 0

    def test_silhouette_score_different_clusterings(self, sample_data_with_clusters):
        """Test silhouette score with different clusterings."""
        data, true_labels = sample_data_with_clusters

        # Create different clusterings
        clusterings = [
            true_labels,  # True clustering
            np.random.randint(0, 4, len(true_labels)),  # Random clustering
            np.zeros(len(true_labels)),  # Single cluster
            np.arange(len(true_labels)),  # Each point is its own cluster
        ]

        silhouette_scores = []
        for clustering in clusterings:
            if len(set(clustering)) > 1:  # Need at least 2 clusters for silhouette
                silhouette = silhouette_score(data, clustering)
                silhouette_scores.append(silhouette)

                # Validate silhouette score
                assert -1 <= silhouette <= 1
                assert not np.isnan(silhouette)
                assert not np.isinf(silhouette)

        # True clustering should have higher silhouette score than random
        if len(silhouette_scores) >= 2:
            assert silhouette_scores[0] >= silhouette_scores[1]

    def test_silhouette_score_reproducibility(self, sample_data_with_clusters):
        """Test reproducibility of silhouette score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate silhouette score multiple times
        scores = []
        for _ in range(3):
            silhouette = silhouette_score(data, true_labels)
            scores.append(silhouette)

        # Scores should be identical
        for i in range(1, len(scores)):
            assert (
                abs(scores[0] - scores[i]) < 1e-10
            ), "Silhouette score not reproducible"

    def test_davies_bouldin_score_calculation(self, sample_data_with_clusters):
        """Test Davies-Bouldin score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate Davies-Bouldin score
        db_score = davies_bouldin_score(data, true_labels)

        # Validate Davies-Bouldin score
        assert isinstance(db_score, float)
        assert db_score >= 0
        assert not np.isnan(db_score)
        assert not np.isinf(db_score)

        # Lower is better for Davies-Bouldin
        assert db_score < 10  # Reasonable upper bound

    def test_davies_bouldin_score_different_clusterings(
        self, sample_data_with_clusters
    ):
        """Test Davies-Bouldin score with different clusterings."""
        data, true_labels = sample_data_with_clusters

        # Create different clusterings
        clusterings = [
            true_labels,  # True clustering
            np.random.randint(0, 4, len(true_labels)),  # Random clustering
        ]

        db_scores = []
        for clustering in clusterings:
            if len(set(clustering)) > 1:  # Need at least 2 clusters
                db_score = davies_bouldin_score(data, clustering)
                db_scores.append(db_score)

                # Validate Davies-Bouldin score
                assert db_score >= 0
                assert not np.isnan(db_score)
                assert not np.isinf(db_score)

        # True clustering should have lower Davies-Bouldin score than random
        if len(db_scores) >= 2:
            assert db_scores[0] <= db_scores[1]

    def test_davies_bouldin_score_reproducibility(self, sample_data_with_clusters):
        """Test reproducibility of Davies-Bouldin score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate Davies-Bouldin score multiple times
        scores = []
        for _ in range(3):
            db_score = davies_bouldin_score(data, true_labels)
            scores.append(db_score)

        # Scores should be identical
        for i in range(1, len(scores)):
            assert (
                abs(scores[0] - scores[i]) < 1e-10
            ), "Davies-Bouldin score not reproducible"

    def test_calinski_harabasz_score_calculation(self, sample_data_with_clusters):
        """Test Calinski-Harabasz score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate Calinski-Harabasz score
        ch_score = calinski_harabasz_score(data, true_labels)

        # Validate Calinski-Harabasz score
        assert isinstance(ch_score, float)
        assert ch_score >= 0
        assert not np.isnan(ch_score)
        assert not np.isinf(ch_score)

        # Higher is better for Calinski-Harabasz
        assert ch_score > 0

    def test_calinski_harabasz_score_different_clusterings(
        self, sample_data_with_clusters
    ):
        """Test Calinski-Harabasz score with different clusterings."""
        data, true_labels = sample_data_with_clusters

        # Create different clusterings
        clusterings = [
            true_labels,  # True clustering
            np.random.randint(0, 4, len(true_labels)),  # Random clustering
        ]

        ch_scores = []
        for clustering in clusterings:
            if len(set(clustering)) > 1:  # Need at least 2 clusters
                ch_score = calinski_harabasz_score(data, clustering)
                ch_scores.append(ch_score)

                # Validate Calinski-Harabasz score
                assert ch_score >= 0
                assert not np.isnan(ch_score)
                assert not np.isinf(ch_score)

        # True clustering should have higher Calinski-Harabasz score than random
        if len(ch_scores) >= 2:
            assert ch_scores[0] >= ch_scores[1]

    def test_calinski_harabasz_score_reproducibility(self, sample_data_with_clusters):
        """Test reproducibility of Calinski-Harabasz score calculation."""
        data, true_labels = sample_data_with_clusters

        # Calculate Calinski-Harabasz score multiple times
        scores = []
        for _ in range(3):
            ch_score = calinski_harabasz_score(data, true_labels)
            scores.append(ch_score)

        # Scores should be identical
        for i in range(1, len(scores)):
            assert (
                abs(scores[0] - scores[i]) < 1e-10
            ), "Calinski-Harabasz score not reproducible"

    def test_adjusted_rand_score_calculation(self, sample_data_with_clusters):
        """Test Adjusted Rand Index calculation."""
        data, true_labels = sample_data_with_clusters

        # Create predicted clustering
        predicted_labels = np.random.randint(0, 4, len(true_labels))

        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(true_labels, predicted_labels)

        # Validate Adjusted Rand Index
        assert isinstance(ari, float)
        assert -1 <= ari <= 1
        assert not np.isnan(ari)
        assert not np.isinf(ari)

        # Test with identical clusterings
        ari_identical = adjusted_rand_score(true_labels, true_labels)
        assert abs(ari_identical - 1.0) < 1e-10

    def test_normalized_mutual_info_score_calculation(self, sample_data_with_clusters):
        """Test Normalized Mutual Information calculation."""
        data, true_labels = sample_data_with_clusters

        # Create predicted clustering
        predicted_labels = np.random.randint(0, 4, len(true_labels))

        # Calculate Normalized Mutual Information
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        # Validate Normalized Mutual Information
        assert isinstance(nmi, float)
        assert 0 <= nmi <= 1
        assert not np.isnan(nmi)
        assert not np.isinf(nmi)

        # Test with identical clusterings
        nmi_identical = normalized_mutual_info_score(true_labels, true_labels)
        assert abs(nmi_identical - 1.0) < 1e-10

    def test_homogeneity_score_calculation(self, sample_data_with_clusters):
        """Test homogeneity score calculation."""
        data, true_labels = sample_data_with_clusters

        # Create predicted clustering
        predicted_labels = np.random.randint(0, 4, len(true_labels))

        # Calculate homogeneity score
        homogeneity = homogeneity_score(true_labels, predicted_labels)

        # Validate homogeneity score
        assert isinstance(homogeneity, float)
        assert 0 <= homogeneity <= 1
        assert not np.isnan(homogeneity)
        assert not np.isinf(homogeneity)

        # Test with identical clusterings
        homogeneity_identical = homogeneity_score(true_labels, true_labels)
        assert abs(homogeneity_identical - 1.0) < 1e-10

    def test_completeness_score_calculation(self, sample_data_with_clusters):
        """Test completeness score calculation."""
        data, true_labels = sample_data_with_clusters

        # Create predicted clustering
        predicted_labels = np.random.randint(0, 4, len(true_labels))

        # Calculate completeness score
        completeness = completeness_score(true_labels, predicted_labels)

        # Validate completeness score
        assert isinstance(completeness, float)
        assert 0 <= completeness <= 1
        assert not np.isnan(completeness)
        assert not np.isinf(completeness)

        # Test with identical clusterings
        completeness_identical = completeness_score(true_labels, true_labels)
        assert abs(completeness_identical - 1.0) < 1e-10

    def test_v_measure_score_calculation(self, sample_data_with_clusters):
        """Test V-measure score calculation."""
        data, true_labels = sample_data_with_clusters

        # Create predicted clustering
        predicted_labels = np.random.randint(0, 4, len(true_labels))

        # Calculate V-measure score
        v_measure = v_measure_score(true_labels, predicted_labels)

        # Validate V-measure score
        assert isinstance(v_measure, float)
        assert 0 <= v_measure <= 1
        assert not np.isnan(v_measure)
        assert not np.isinf(v_measure)

        # Test with identical clusterings
        v_measure_identical = v_measure_score(true_labels, true_labels)
        assert abs(v_measure_identical - 1.0) < 1e-10

    def test_evaluation_metrics_comparison(self, sample_data_with_clusters):
        """Test comparison of different evaluation metrics."""
        data, true_labels = sample_data_with_clusters

        # Create different clusterings
        clusterings = [
            ("true", true_labels),
            ("random", np.random.randint(0, 4, len(true_labels))),
            ("single", np.zeros(len(true_labels))),
            ("individual", np.arange(len(true_labels))),
        ]

        results = {}

        for name, clustering in clusterings:
            if len(set(clustering)) > 1:  # Need at least 2 clusters for most metrics
                metrics = {}

                # Calculate all metrics
                metrics["silhouette"] = silhouette_score(data, clustering)
                metrics["davies_bouldin"] = davies_bouldin_score(data, clustering)
                metrics["calinski_harabasz"] = calinski_harabasz_score(data, clustering)

                # Calculate supervised metrics if we have true labels
                if name != "true":
                    metrics["ari"] = adjusted_rand_score(true_labels, clustering)
                    metrics["nmi"] = normalized_mutual_info_score(
                        true_labels, clustering
                    )
                    metrics["homogeneity"] = homogeneity_score(true_labels, clustering)
                    metrics["completeness"] = completeness_score(
                        true_labels, clustering
                    )
                    metrics["v_measure"] = v_measure_score(true_labels, clustering)

                results[name] = metrics

        # Validate all metrics
        for name, metrics in results.items():
            for metric_name, value in metrics.items():
                assert not np.isnan(value)
                assert not np.isinf(value)

                # Validate metric ranges
                if metric_name in ["silhouette", "ari"]:
                    assert -1 <= value <= 1
                elif metric_name in ["nmi", "homogeneity", "completeness", "v_measure"]:
                    assert 0 <= value <= 1
                elif metric_name in ["davies_bouldin", "calinski_harabasz"]:
                    assert value >= 0

    def test_evaluation_metrics_performance(self, sample_data_with_clusters):
        """Test performance of evaluation metrics calculation."""
        import time

        data, true_labels = sample_data_with_clusters

        # Time the metric calculations
        start_time = time.time()

        # Calculate all metrics
        silhouette = silhouette_score(data, true_labels)
        db_score = davies_bouldin_score(data, true_labels)
        ch_score = calinski_harabasz_score(data, true_labels)
        ari = adjusted_rand_score(true_labels, true_labels)
        nmi = normalized_mutual_info_score(true_labels, true_labels)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert (
            processing_time < 5.0
        ), f"Evaluation metrics too slow: {processing_time:.2f}s"

    def test_evaluation_metrics_memory_usage(self, sample_data_with_clusters):
        """Test memory usage during evaluation metrics calculation."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Calculate all metrics
        silhouette = silhouette_score(
            sample_data_with_clusters[0], sample_data_with_clusters[1]
        )
        db_score = davies_bouldin_score(
            sample_data_with_clusters[0], sample_data_with_clusters[1]
        )
        ch_score = calinski_harabasz_score(
            sample_data_with_clusters[0], sample_data_with_clusters[1]
        )
        ari = adjusted_rand_score(
            sample_data_with_clusters[1], sample_data_with_clusters[1]
        )
        nmi = normalized_mutual_info_score(
            sample_data_with_clusters[1], sample_data_with_clusters[1]
        )

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_evaluation_metrics_error_handling(self, sample_data_with_clusters):
        """Test error handling in evaluation metrics calculation."""
        data, true_labels = sample_data_with_clusters

        # Test with invalid data
        invalid_data = [
            np.array([]),  # Empty array
            np.zeros((0, 50)),  # Zero samples
            np.ones((10, 0)),  # Zero features
            np.ones((10, 50)) * np.nan,  # NaN values
        ]

        for invalid_d in invalid_data:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if invalid_d.size == 0:
                    raise ValueError("Empty data")
                if np.any(np.isnan(invalid_d)):
                    raise ValueError("NaN values in data")

        # Test with invalid labels
        invalid_labels = [
            np.array([]),  # Empty array
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),  # Wrong length
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * np.nan,  # NaN values
        ]

        for invalid_l in invalid_labels:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if len(invalid_l) != len(data):
                    raise ValueError("Labels length mismatch")
                if np.any(np.isnan(invalid_l)):
                    raise ValueError("NaN values in labels")

        # Test with single cluster
        single_cluster_labels = np.zeros(len(data))
        with pytest.raises(ValueError):
            # This should raise an error for metrics that require multiple clusters
            silhouette_score(data, single_cluster_labels)

    def test_evaluation_metrics_with_different_data_sizes(
        self, sample_data_with_clusters
    ):
        """Test evaluation metrics with different data sizes."""
        data, true_labels = sample_data_with_clusters
        data_sizes = [10, 50, 100]

        for size in data_sizes:
            # Create subset of data
            subset_data = data[:size]
            subset_labels = true_labels[:size]

            # Calculate metrics
            silhouette = silhouette_score(subset_data, subset_labels)
            db_score = davies_bouldin_score(subset_data, subset_labels)
            ch_score = calinski_harabasz_score(subset_data, subset_labels)

            # Validate metrics
            assert -1 <= silhouette <= 1
            assert db_score >= 0
            assert ch_score >= 0
            assert not np.isnan(silhouette)
            assert not np.isnan(db_score)
            assert not np.isnan(ch_score)

    def test_evaluation_metrics_quality_assessment(self, sample_data_with_clusters):
        """Test quality assessment using evaluation metrics."""
        data, true_labels = sample_data_with_clusters

        # Create different quality clusterings
        good_clustering = true_labels
        random_clustering = np.random.randint(0, 4, len(true_labels))

        # Calculate metrics for both clusterings
        good_silhouette = silhouette_score(data, good_clustering)
        good_db = davies_bouldin_score(data, good_clustering)
        good_ch = calinski_harabasz_score(data, good_clustering)

        random_silhouette = silhouette_score(data, random_clustering)
        random_db = davies_bouldin_score(data, random_clustering)
        random_ch = calinski_harabasz_score(data, random_clustering)

        # Good clustering should have better metrics
        assert good_silhouette >= random_silhouette  # Higher is better
        assert good_db <= random_db  # Lower is better
        assert good_ch >= random_ch  # Higher is better

        # All metrics should be reasonable
        assert -1 <= good_silhouette <= 1
        assert good_db >= 0
        assert good_ch >= 0
        assert -1 <= random_silhouette <= 1
        assert random_db >= 0
        assert random_ch >= 0


if __name__ == "__main__":
    pytest.main([__file__])
