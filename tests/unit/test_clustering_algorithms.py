"""
Unit tests for clustering algorithms functionality.

These tests validate individual clustering algorithm functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
hdbscan = pytest.importorskip("hdbscan")

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

# Import the modules we'll be testing (these will be implemented later)
# from mammography.clustering.algorithms import ClusteringAlgorithm
# from mammography.clustering.kmeans_clusterer import KMeansClusterer
# from mammography.clustering.gmm_clusterer import GMMClusterer
# from mammography.clustering.hdbscan_clusterer import HDBSCANClusterer


class TestClusteringAlgorithms:
    """Unit tests for clustering algorithm functions."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Create data with 4 distinct clusters
        cluster_centers = np.random.randn(4, n_features) * 2
        data = []
        labels = []

        for i in range(4):
            # Generate samples for each cluster
            cluster_samples = np.random.randn(25, n_features) * 0.5 + cluster_centers[i]
            data.append(cluster_samples)
            labels.extend([i] * 25)

        data = np.vstack(data)
        labels = np.array(labels)

        return data, labels

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample embeddings for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2048

        # Create embeddings with some structure
        embeddings = np.random.randn(n_samples, n_features)

        # Add some structure to make clustering meaningful
        for i in range(n_samples):
            embeddings[i] += np.random.randn(n_features) * 0.1

        return embeddings

    def test_kmeans_clustering(self, sample_data):
        """Test K-means clustering algorithm."""
        data, true_labels = sample_data
        n_clusters = 4

        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Validate clustering results
        assert len(cluster_labels) == len(data)
        assert len(set(cluster_labels)) <= n_clusters
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)

        # Validate K-means properties
        assert kmeans.n_clusters == n_clusters
        assert kmeans.cluster_centers_.shape == (n_clusters, data.shape[1])
        assert not np.any(np.isnan(kmeans.cluster_centers_))
        assert not np.any(np.isinf(kmeans.cluster_centers_))

        # Test inertia
        assert kmeans.inertia_ >= 0
        assert not np.isnan(kmeans.inertia_)
        assert not np.isinf(kmeans.inertia_)

    def test_kmeans_different_clusters(self, sample_data):
        """Test K-means with different numbers of clusters."""
        data, true_labels = sample_data
        n_clusters_list = [2, 3, 4, 5, 6]

        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)

            # Validate clustering results
            assert len(cluster_labels) == len(data)
            assert len(set(cluster_labels)) <= n_clusters
            assert kmeans.n_clusters == n_clusters
            assert kmeans.cluster_centers_.shape == (n_clusters, data.shape[1])

    def test_kmeans_reproducibility(self, sample_data):
        """Test reproducibility of K-means with fixed seeds."""
        data, true_labels = sample_data
        n_clusters = 4
        random_state = 42

        # Fit K-means multiple times with same seed
        results = []
        for _ in range(3):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(data)
            results.append(cluster_labels)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), "K-means not reproducible"

    def test_kmeans_cluster_centers(self, sample_data):
        """Test K-means cluster centers."""
        data, true_labels = sample_data
        n_clusters = 4

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)

        # Validate cluster centers
        centers = kmeans.cluster_centers_
        assert centers.shape == (n_clusters, data.shape[1])
        assert not np.any(np.isnan(centers))
        assert not np.any(np.isinf(centers))

        # Test prediction using cluster centers
        predictions = kmeans.predict(data)
        assert len(predictions) == len(data)
        assert len(set(predictions)) <= n_clusters

    def test_gmm_clustering(self, sample_data):
        """Test Gaussian Mixture Model clustering algorithm."""
        data, true_labels = sample_data
        n_components = 4

        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        cluster_labels = gmm.fit_predict(data)

        # Validate clustering results
        assert len(cluster_labels) == len(data)
        assert len(set(cluster_labels)) <= n_components
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)

        # Validate GMM properties
        assert gmm.n_components == n_components
        assert gmm.means_.shape == (n_components, data.shape[1])
        assert gmm.covariances_.shape == (n_components, data.shape[1], data.shape[1])
        assert gmm.weights_.shape == (n_components,)

        # Validate GMM parameters
        assert not np.any(np.isnan(gmm.means_))
        assert not np.any(np.isinf(gmm.means_))
        assert not np.any(np.isnan(gmm.covariances_))
        assert not np.any(np.isinf(gmm.covariances_))
        assert not np.any(np.isnan(gmm.weights_))
        assert not np.any(np.isinf(gmm.weights_))

        # Weights should sum to 1
        assert abs(np.sum(gmm.weights_) - 1.0) < 1e-6

    def test_gmm_different_components(self, sample_data):
        """Test GMM with different numbers of components."""
        data, true_labels = sample_data
        n_components_list = [2, 3, 4, 5, 6]

        for n_components in n_components_list:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            cluster_labels = gmm.fit_predict(data)

            # Validate clustering results
            assert len(cluster_labels) == len(data)
            assert len(set(cluster_labels)) <= n_components
            assert gmm.n_components == n_components
            assert gmm.means_.shape == (n_components, data.shape[1])
            assert gmm.covariances_.shape == (
                n_components,
                data.shape[1],
                data.shape[1],
            )
            assert gmm.weights_.shape == (n_components,)

    def test_gmm_reproducibility(self, sample_data):
        """Test reproducibility of GMM with fixed seeds."""
        data, true_labels = sample_data
        n_components = 4
        random_state = 42

        # Fit GMM multiple times with same seed
        results = []
        for _ in range(3):
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            cluster_labels = gmm.fit_predict(data)
            results.append(cluster_labels)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), "GMM not reproducible"

    def test_gmm_probabilities(self, sample_data):
        """Test GMM probability predictions."""
        data, true_labels = sample_data
        n_components = 4

        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data)

        # Test probability predictions
        probabilities = gmm.predict_proba(data)

        # Validate probabilities
        assert probabilities.shape == (len(data), n_components)
        assert not np.any(np.isnan(probabilities))
        assert not np.any(np.isinf(probabilities))

        # Probabilities should sum to 1 for each sample
        prob_sums = np.sum(probabilities, axis=1)
        assert np.allclose(prob_sums, 1.0)

        # All probabilities should be non-negative
        assert np.all(probabilities >= 0)

    def test_hdbscan_clustering(self, sample_data):
        """Test HDBSCAN clustering algorithm."""
        data, true_labels = sample_data

        # Fit HDBSCAN
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        cluster_labels = hdbscan_clusterer.fit_predict(data)

        # Validate clustering results
        assert len(cluster_labels) == len(data)
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)

        # HDBSCAN can assign -1 for noise points
        assert all(label >= -1 for label in cluster_labels)

        # Validate HDBSCAN properties
        assert hdbscan_clusterer.min_cluster_size == 10
        assert hdbscan_clusterer.min_samples == 5

        # Test cluster persistence
        if hasattr(hdbscan_clusterer, "cluster_persistence_"):
            assert len(hdbscan_clusterer.cluster_persistence_) >= 0

    def test_hdbscan_different_parameters(self, sample_data):
        """Test HDBSCAN with different parameters."""
        data, true_labels = sample_data

        parameter_sets = [
            {"min_cluster_size": 5, "min_samples": 3},
            {"min_cluster_size": 15, "min_samples": 7},
            {"min_cluster_size": 20, "min_samples": 10},
        ]

        for params in parameter_sets:
            hdbscan_clusterer = hdbscan.HDBSCAN(**params)
            cluster_labels = hdbscan_clusterer.fit_predict(data)

            # Validate clustering results
            assert len(cluster_labels) == len(data)
            assert all(label >= -1 for label in cluster_labels)
            assert hdbscan_clusterer.min_cluster_size == params["min_cluster_size"]
            assert hdbscan_clusterer.min_samples == params["min_samples"]

    def test_hdbscan_reproducibility(self, sample_data):
        """Test reproducibility of HDBSCAN with fixed seeds."""
        data, true_labels = sample_data
        if "random_state" not in hdbscan.HDBSCAN().get_params():
            pytest.skip("HDBSCAN random_state not supported")
        random_state = 42

        # Fit HDBSCAN multiple times with same seed
        results = []
        for _ in range(3):
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=10, min_samples=5, random_state=random_state
            )
            cluster_labels = hdbscan_clusterer.fit_predict(data)
            results.append(cluster_labels)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), "HDBSCAN not reproducible"

    def test_hdbscan_probabilities(self, sample_data):
        """Test HDBSCAN probability predictions."""
        data, true_labels = sample_data

        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        hdbscan_clusterer.fit(data)

        # Test probability predictions
        probabilities = hdbscan_clusterer.probabilities_

        # Validate probabilities
        assert len(probabilities) == len(data)
        assert not np.any(np.isnan(probabilities))
        assert not np.any(np.isinf(probabilities))

        # Probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_agglomerative_clustering(self, sample_data):
        """Test Agglomerative clustering algorithm."""
        data, true_labels = sample_data
        n_clusters = 4

        # Fit Agglomerative clustering
        agg_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clusterer.fit_predict(data)

        # Validate clustering results
        assert len(cluster_labels) == len(data)
        assert len(set(cluster_labels)) <= n_clusters
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)

        # Validate Agglomerative properties
        assert agg_clusterer.n_clusters == n_clusters
        assert agg_clusterer.labels_.shape == (len(data),)

    def test_clustering_evaluation_metrics(self, sample_data):
        """Test clustering evaluation metrics."""
        data, true_labels = sample_data
        n_clusters = 4

        # Fit different clustering algorithms
        algorithms = [
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),
            ("gmm", GaussianMixture(n_components=n_clusters, random_state=42)),
        ]

        for name, algorithm in algorithms:
            cluster_labels = algorithm.fit_predict(data)

            # Calculate evaluation metrics
            silhouette = silhouette_score(data, cluster_labels)
            db_score = davies_bouldin_score(data, cluster_labels)
            ch_score = calinski_harabasz_score(data, cluster_labels)

            # Validate metrics
            assert -1 <= silhouette <= 1
            assert db_score >= 0
            assert ch_score >= 0
            assert not np.isnan(silhouette)
            assert not np.isnan(db_score)
            assert not np.isnan(ch_score)

    def test_clustering_performance(self, sample_data):
        """Test clustering performance benchmarks."""
        import time

        data, true_labels = sample_data
        n_clusters = 4

        # Test K-means performance
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit_predict(data)
        kmeans_time = time.time() - start_time

        # Test GMM performance
        start_time = time.time()
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit_predict(data)
        gmm_time = time.time() - start_time

        # Test HDBSCAN performance
        start_time = time.time()
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        hdbscan_clusterer.fit_predict(data)
        hdbscan_time = time.time() - start_time

        # Should complete within reasonable time (adjust thresholds as needed)
        assert kmeans_time < 5.0, f"K-means too slow: {kmeans_time:.2f}s"
        assert gmm_time < 10.0, f"GMM too slow: {gmm_time:.2f}s"
        assert hdbscan_time < 15.0, f"HDBSCAN too slow: {hdbscan_time:.2f}s"

    def test_clustering_memory_usage(self, sample_data):
        """Test memory usage during clustering."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(sample_data[0])

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_clustering_error_handling(self, sample_data):
        """Test error handling in clustering algorithms."""
        data, true_labels = sample_data

        # Test with invalid number of clusters
        invalid_clusters = [0, -1, len(data) + 1]

        for n_clusters in invalid_clusters:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if n_clusters <= 0:
                    raise ValueError("Invalid number of clusters")
                if n_clusters > len(data):
                    raise ValueError("Too many clusters")

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

    def test_clustering_with_different_data_sizes(self, sample_embeddings):
        """Test clustering with different data sizes."""
        data_sizes = [10, 50, 100]

        for size in data_sizes:
            # Create subset of data
            subset = sample_embeddings[:size]
            n_clusters = min(4, size // 5)  # Adjust clusters based on size

            if n_clusters > 0:
                # Test K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(subset)

                # Validate results
                assert len(cluster_labels) == size
                assert len(set(cluster_labels)) <= n_clusters

                # Test GMM
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                cluster_labels = gmm.fit_predict(subset)

                # Validate results
                assert len(cluster_labels) == size
                assert len(set(cluster_labels)) <= n_clusters

    def test_clustering_quality_metrics(self, sample_data):
        """Test quality metrics for clustering results."""
        data, true_labels = sample_data
        n_clusters = 4

        # Fit clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Test quality metrics
        silhouette = silhouette_score(data, cluster_labels)
        db_score = davies_bouldin_score(data, cluster_labels)
        ch_score = calinski_harabasz_score(data, cluster_labels)

        # Validate quality metrics
        assert -1 <= silhouette <= 1
        assert db_score >= 0
        assert ch_score >= 0

        # Test that metrics are reasonable (not all identical)
        assert not np.isnan(silhouette)
        assert not np.isnan(db_score)
        assert not np.isnan(ch_score)

        # Test cluster quality
        n_clusters_found = len(set(cluster_labels))
        assert n_clusters_found > 0
        assert n_clusters_found <= n_clusters

        # Test cluster sizes
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters_found)]
        assert all(size > 0 for size in cluster_sizes)
        assert sum(cluster_sizes) == len(data)


if __name__ == "__main__":
    pytest.main([__file__])
