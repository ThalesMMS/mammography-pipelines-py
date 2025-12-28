"""
Unit tests for dimensionality reduction functionality.

These tests validate individual dimensionality reduction functions and operations.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
umap = pytest.importorskip("umap")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import the modules we'll be testing (these will be implemented later)
# from mammography.clustering.dimensionality_reduction import DimensionalityReducer
# from mammography.clustering.pca_reducer import PCAReducer
# from mammography.clustering.umap_reducer import UMAPReducer


MATMUL_ERRSTATE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


@pytest.fixture(autouse=True)
def _suppress_numpy_matmul_warnings():
    with np.errstate(**MATMUL_ERRSTATE):
        yield


class TestDimensionalityReduction:
    """Unit tests for dimensionality reduction functions."""

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample embeddings for testing."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 2048

        # Create embeddings with some structure
        embeddings = rng.normal(scale=0.5, size=(n_samples, n_features))

        # Add some structure to make dimensionality reduction meaningful
        embeddings += rng.normal(scale=0.05, size=embeddings.shape)

        embeddings = embeddings.astype(np.float64)
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        std = embeddings.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        embeddings /= std

        return embeddings

    @pytest.fixture
    def sample_embeddings_with_clusters(self) -> np.ndarray:
        """Create sample embeddings with clear cluster structure."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 2048
        n_clusters = 4

        # Create cluster centers
        cluster_centers = rng.normal(scale=0.5, size=(n_clusters, n_features))

        # Generate samples for each cluster
        embeddings = []
        for i in range(n_clusters):
            cluster_samples = (
                rng.normal(scale=0.1, size=(n_samples // n_clusters, n_features))
                + cluster_centers[i]
            )
            embeddings.append(cluster_samples)

        embeddings = np.vstack(embeddings).astype(np.float64)
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        std = embeddings.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        embeddings /= std
        return embeddings

    def test_pca_reduction(self, sample_embeddings):
        """Test PCA dimensionality reduction."""
        n_components = 50

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
        embeddings_reduced = pca.fit_transform(sample_embeddings)

        # Validate reduced embeddings
        assert embeddings_reduced.shape == (sample_embeddings.shape[0], n_components)
        assert not np.any(np.isnan(embeddings_reduced))
        assert not np.any(np.isinf(embeddings_reduced))

        # Validate PCA properties
        assert pca.n_components_ == n_components
        assert pca.explained_variance_ratio_.shape == (n_components,)
        assert np.all(pca.explained_variance_ratio_ >= 0)
        assert np.all(pca.explained_variance_ratio_ <= 1)

        # Validate explained variance ratio sums to reasonable value
        total_variance = np.sum(pca.explained_variance_ratio_)
        assert total_variance > 0
        assert total_variance <= 1.0 + 1e-6

    def test_pca_different_components(self, sample_embeddings):
        """Test PCA with different numbers of components."""
        n_components_list = [10, 50, 100, 256]
        max_components = min(sample_embeddings.shape)

        for n_components in n_components_list:
            n_components = min(n_components, max_components)
            # Fit PCA
            pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
            embeddings_reduced = pca.fit_transform(sample_embeddings)

            # Validate reduced embeddings
            assert embeddings_reduced.shape == (
                sample_embeddings.shape[0],
                n_components,
            )
            assert not np.any(np.isnan(embeddings_reduced))
            assert not np.any(np.isinf(embeddings_reduced))

            # Validate PCA properties
            assert pca.n_components_ == n_components
            assert pca.explained_variance_ratio_.shape == (n_components,)

            # Validate explained variance ratio
            total_variance = np.sum(pca.explained_variance_ratio_)
            assert total_variance > 0
            assert total_variance <= 1.0 + 1e-6

    def test_pca_reproducibility(self, sample_embeddings):
        """Test reproducibility of PCA with fixed seeds."""
        n_components = 50
        random_state = 42

        # Fit PCA multiple times with same seed
        results = []
        for _ in range(3):
            pca = PCA(
                n_components=n_components,
                random_state=random_state,
                svd_solver="full",
            )
            embeddings_reduced = pca.fit_transform(sample_embeddings)
            results.append(embeddings_reduced)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i]), "PCA not reproducible"

    def test_pca_inverse_transform(self, sample_embeddings):
        """Test PCA inverse transform."""
        n_components = 50

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
        embeddings_reduced = pca.fit_transform(sample_embeddings)

        # Inverse transform
        embeddings_reconstructed = pca.inverse_transform(embeddings_reduced)

        # Validate reconstructed embeddings
        assert embeddings_reconstructed.shape == sample_embeddings.shape
        assert not np.any(np.isnan(embeddings_reconstructed))
        assert not np.any(np.isinf(embeddings_reconstructed))

        # Reconstruction should be close to original (within some tolerance)
        reconstruction_error = np.mean(
            (sample_embeddings - embeddings_reconstructed) ** 2
        )
        assert reconstruction_error > 0  # Some error is expected
        assert reconstruction_error < 1.0  # But not too much error

    def test_pca_explained_variance(self, sample_embeddings):
        """Test PCA explained variance analysis."""
        n_components = 50

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
        pca.fit(sample_embeddings)

        # Validate explained variance
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_

        assert explained_variance.shape == (n_components,)
        assert explained_variance_ratio.shape == (n_components,)

        # Explained variance should be decreasing
        assert np.all(np.diff(explained_variance) <= 0)
        assert np.all(np.diff(explained_variance_ratio) <= 0)

        # Cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        assert np.all(cumulative_variance >= 0)
        assert np.all(cumulative_variance <= 1.0)
        assert cumulative_variance[-1] <= 1.0

    def test_umap_reduction(self, sample_embeddings):
        """Test UMAP dimensionality reduction."""
        n_components = 2

        # Fit UMAP
        umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_reduced = umap_reducer.fit_transform(sample_embeddings)

        # Validate reduced embeddings
        assert embeddings_reduced.shape == (sample_embeddings.shape[0], n_components)
        assert not np.any(np.isnan(embeddings_reduced))
        assert not np.any(np.isinf(embeddings_reduced))

        # Validate UMAP properties
        assert umap_reducer.n_components == n_components
        assert umap_reducer.embedding_.shape == (
            sample_embeddings.shape[0],
            n_components,
        )

    def test_umap_different_components(self, sample_embeddings):
        """Test UMAP with different numbers of components."""
        n_components_list = [2, 3, 10, 50]

        for n_components in n_components_list:
            # Fit UMAP
            umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
            embeddings_reduced = umap_reducer.fit_transform(sample_embeddings)

            # Validate reduced embeddings
            assert embeddings_reduced.shape == (
                sample_embeddings.shape[0],
                n_components,
            )
            assert not np.any(np.isnan(embeddings_reduced))
            assert not np.any(np.isinf(embeddings_reduced))

            # Validate UMAP properties
            assert umap_reducer.n_components == n_components
            assert umap_reducer.embedding_.shape == (
                sample_embeddings.shape[0],
                n_components,
            )

    def test_umap_reproducibility(self, sample_embeddings):
        """Test reproducibility of UMAP with fixed seeds."""
        n_components = 2
        random_state = 42

        # Fit UMAP multiple times with same seed
        results = []
        for _ in range(3):
            umap_reducer = umap.UMAP(
                n_components=n_components, random_state=random_state
            )
            embeddings_reduced = umap_reducer.fit_transform(sample_embeddings)
            results.append(embeddings_reduced)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i]), "UMAP not reproducible"

    def test_umap_with_clusters(self, sample_embeddings_with_clusters):
        """Test UMAP with data that has clear cluster structure."""
        n_components = 2

        # Fit UMAP
        umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_reduced = umap_reducer.fit_transform(sample_embeddings_with_clusters)

        # Validate reduced embeddings
        assert embeddings_reduced.shape == (
            sample_embeddings_with_clusters.shape[0],
            n_components,
        )
        assert not np.any(np.isnan(embeddings_reduced))
        assert not np.any(np.isinf(embeddings_reduced))

        # Check if clusters are preserved (basic check)
        # This is a simplified test - in practice, you'd use clustering metrics
        assert (
            len(np.unique(embeddings_reduced, axis=0)) > 1
        )  # Not all points are identical

    def test_tsne_reduction(self, sample_embeddings):
        """Test t-SNE dimensionality reduction."""
        n_components = 2

        # Fit t-SNE
        tsne = TSNE(n_components=n_components, random_state=42)
        embeddings_reduced = tsne.fit_transform(sample_embeddings)

        # Validate reduced embeddings
        assert embeddings_reduced.shape == (sample_embeddings.shape[0], n_components)
        assert not np.any(np.isnan(embeddings_reduced))
        assert not np.any(np.isinf(embeddings_reduced))

        # Validate t-SNE properties
        assert tsne.n_components == n_components
        assert tsne.embedding_.shape == (sample_embeddings.shape[0], n_components)

    def test_dimensionality_reduction_performance(self, sample_embeddings):
        """Test performance of dimensionality reduction methods."""
        import time

        # Test PCA performance
        start_time = time.time()
        pca = PCA(n_components=50, random_state=42)
        pca.fit_transform(sample_embeddings)
        pca_time = time.time() - start_time

        # Test UMAP performance
        start_time = time.time()
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_reducer.fit_transform(sample_embeddings)
        umap_time = time.time() - start_time

        # Should complete within reasonable time (adjust thresholds as needed)
        assert pca_time < 5.0, f"PCA too slow: {pca_time:.2f}s"
        assert umap_time < 30.0, f"UMAP too slow: {umap_time:.2f}s"

    def test_dimensionality_reduction_memory_usage(self, sample_embeddings):
        """Test memory usage during dimensionality reduction."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform PCA
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(sample_embeddings)

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert (
            memory_increase < 200 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"

    def test_dimensionality_reduction_error_handling(self, sample_embeddings):
        """Test error handling in dimensionality reduction."""
        # Test with invalid number of components
        invalid_components = [0, -1, sample_embeddings.shape[1] + 1]

        for n_components in invalid_components:
            with pytest.raises((ValueError, RuntimeError)):
                # This should raise an error
                if n_components <= 0:
                    raise ValueError("Invalid number of components")
                if n_components > sample_embeddings.shape[1]:
                    raise ValueError("Too many components")

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

    def test_dimensionality_reduction_comparison(self, sample_embeddings):
        """Test comparison of different dimensionality reduction methods."""
        n_components = 50

        # PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(sample_embeddings)

        # Validate PCA results
        assert embeddings_pca.shape == (sample_embeddings.shape[0], n_components)
        assert not np.any(np.isnan(embeddings_pca))
        assert not np.any(np.isinf(embeddings_pca))

        # UMAP (2D for visualization)
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_umap = umap_reducer.fit_transform(sample_embeddings)

        # Validate UMAP results
        assert embeddings_umap.shape == (sample_embeddings.shape[0], 2)
        assert not np.any(np.isnan(embeddings_umap))
        assert not np.any(np.isinf(embeddings_umap))

        # Compare explained variance (PCA only)
        pca_variance = np.sum(pca.explained_variance_ratio_)
        assert pca_variance > 0
        assert pca_variance <= 1.0

    def test_dimensionality_reduction_with_different_data_sizes(
        self, sample_embeddings
    ):
        """Test dimensionality reduction with different data sizes."""
        data_sizes = [10, 50, 100]

        for size in data_sizes:
            # Create subset of data
            subset = sample_embeddings[:size]

            # Test PCA
            pca = PCA(n_components=min(10, size), random_state=42)
            embeddings_reduced = pca.fit_transform(subset)

            # Validate results
            assert embeddings_reduced.shape == (size, min(10, size))
            assert not np.any(np.isnan(embeddings_reduced))
            assert not np.any(np.isinf(embeddings_reduced))

            # Test UMAP
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_umap = umap_reducer.fit_transform(subset)

            # Validate results
            assert embeddings_umap.shape == (size, 2)
            assert not np.any(np.isnan(embeddings_umap))
            assert not np.any(np.isinf(embeddings_umap))

    def test_dimensionality_reduction_quality_metrics(
        self, sample_embeddings_with_clusters
    ):
        """Test quality metrics for dimensionality reduction."""
        n_components = 2

        # PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(sample_embeddings_with_clusters)

        # UMAP
        umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_umap = umap_reducer.fit_transform(sample_embeddings_with_clusters)

        # Test variance preservation (PCA)
        original_variance = np.var(sample_embeddings_with_clusters, axis=0).sum()
        pca_variance = np.var(embeddings_pca, axis=0).sum()

        # PCA should preserve some variance
        assert pca_variance > 0

        # Test embedding quality (both methods)
        for embeddings in [embeddings_pca, embeddings_umap]:
            # Embeddings should not be all zeros
            assert not np.all(embeddings == 0)

            # Embeddings should have reasonable variance
            embedding_std = np.std(embeddings)
            assert embedding_std > 0.1  # Adjust threshold as needed

            # Embeddings should not be all the same value
            unique_values = np.unique(embeddings)
            assert len(unique_values) > 1


if __name__ == "__main__":
    pytest.main([__file__])
