#!/usr/bin/env python3
#
# test_clustering_evaluator.py
# mammography-pipelines
#
# Unit tests for ClusteringEvaluator and related components in eval/clustering_evaluator.py.
#
"""Unit tests for mammography.eval.clustering_evaluator module."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestClusteringEvaluatorConstants:
    """Tests for class-level constants on ClusteringEvaluator."""

    def test_supported_metrics_is_list(self):
        """SUPPORTED_METRICS is a list."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert isinstance(ClusteringEvaluator.SUPPORTED_METRICS, list)

    def test_supported_metrics_contains_silhouette(self):
        """SUPPORTED_METRICS includes 'silhouette'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "silhouette" in ClusteringEvaluator.SUPPORTED_METRICS

    def test_supported_metrics_contains_davies_bouldin(self):
        """SUPPORTED_METRICS includes 'davies_bouldin'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "davies_bouldin" in ClusteringEvaluator.SUPPORTED_METRICS

    def test_supported_metrics_contains_calinski_harabasz(self):
        """SUPPORTED_METRICS includes 'calinski_harabasz'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "calinski_harabasz" in ClusteringEvaluator.SUPPORTED_METRICS

    def test_supported_metrics_contains_ari(self):
        """SUPPORTED_METRICS includes 'ari'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "ari" in ClusteringEvaluator.SUPPORTED_METRICS

    def test_supported_metrics_contains_nmi(self):
        """SUPPORTED_METRICS includes 'nmi'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "nmi" in ClusteringEvaluator.SUPPORTED_METRICS

    def test_sanity_check_methods_is_list(self):
        """SANITY_CHECK_METHODS is a list."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert isinstance(ClusteringEvaluator.SANITY_CHECK_METHODS, list)

    def test_sanity_check_methods_contains_cluster_size_analysis(self):
        """SANITY_CHECK_METHODS includes 'cluster_size_analysis'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "cluster_size_analysis" in ClusteringEvaluator.SANITY_CHECK_METHODS

    def test_sanity_check_methods_contains_embedding_statistics(self):
        """SANITY_CHECK_METHODS includes 'embedding_statistics'."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert "embedding_statistics" in ClusteringEvaluator.SANITY_CHECK_METHODS

    def test_sanity_check_methods_nonempty(self):
        """SANITY_CHECK_METHODS contains at least one entry."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert len(ClusteringEvaluator.SANITY_CHECK_METHODS) > 0

    def test_supported_metrics_nonempty(self):
        """SUPPORTED_METRICS contains at least one entry."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        assert len(ClusteringEvaluator.SUPPORTED_METRICS) > 0


class TestClusteringEvaluatorValidateConfig:
    """Tests for ClusteringEvaluator._validate_config."""

    def _make_evaluator_with_config(self, config: Dict[str, Any]):
        """Create a ClusteringEvaluator bypassing __init__ to call _validate_config directly."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        evaluator = ClusteringEvaluator.__new__(ClusteringEvaluator)
        return evaluator._validate_config(config)

    def test_empty_config_gets_defaults(self):
        """Empty config is filled with all required defaults."""
        config = self._make_evaluator_with_config({})
        assert "metrics" in config
        assert "sanity_checks" in config
        assert "visual_prototypes" in config
        assert "seed" in config

    def test_default_seed_is_42(self):
        """Default seed is 42."""
        config = self._make_evaluator_with_config({})
        assert config["seed"] == 42

    def test_custom_seed_preserved(self):
        """Custom seed is preserved."""
        config = self._make_evaluator_with_config({"seed": 123})
        assert config["seed"] == 123

    def test_default_visual_prototypes_config(self):
        """Default visual_prototypes config has expected structure."""
        config = self._make_evaluator_with_config({})
        vp_config = config["visual_prototypes"]
        assert "n_samples_per_cluster" in vp_config
        assert "selection_method" in vp_config

    def test_default_metrics_contains_supported_metrics(self):
        """Default metrics list uses SUPPORTED_METRICS."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        config = self._make_evaluator_with_config({})
        assert config["metrics"] == ClusteringEvaluator.SUPPORTED_METRICS

    def test_custom_metrics_preserved(self):
        """Custom metrics list is preserved."""
        custom_metrics = ["silhouette", "ari"]
        config = self._make_evaluator_with_config({"metrics": custom_metrics})
        assert config["metrics"] == custom_metrics

    def test_unknown_metric_does_not_raise(self):
        """Unknown metric in config does not raise; it is logged as warning."""
        # Should not raise, just log
        config = self._make_evaluator_with_config({"metrics": ["nonexistent_metric"]})
        assert "nonexistent_metric" in config["metrics"]


class TestClusteringEvaluatorGenerateSummary:
    """Tests for ClusteringEvaluator._generate_evaluation_summary."""

    def _make_clustering_result(self, labels):
        """Create a minimal mock ClusteringResult."""
        import torch
        mock_result = MagicMock()
        mock_result.algorithm = "kmeans"
        mock_result.cluster_labels = torch.tensor(labels)
        return mock_result

    def _make_evaluator(self):
        """Create ClusteringEvaluator with minimal config."""
        from mammography.eval.clustering_evaluator import ClusteringEvaluator
        evaluator = ClusteringEvaluator.__new__(ClusteringEvaluator)
        evaluator.config = {"metrics": [], "sanity_checks": [], "seed": 42}
        return evaluator

    def test_summary_has_expected_keys(self):
        """Summary dict has expected top-level keys."""
        evaluator = self._make_evaluator()
        clustering_result = self._make_clustering_result([0, 0, 1, 1, 2])
        evaluation_results = {
            "evaluation_timestamp": "2026-01-01T00:00:00",
            "clustering_result": clustering_result,
        }
        summary = evaluator._generate_evaluation_summary(evaluation_results)
        assert "evaluation_timestamp" in summary
        assert "clustering_algorithm" in summary
        assert "n_clusters" in summary
        assert "n_samples" in summary

    def test_n_samples_is_length_of_labels(self):
        """n_samples matches the number of labels."""
        evaluator = self._make_evaluator()
        labels = [0, 0, 1, 1, 2, 2]
        clustering_result = self._make_clustering_result(labels)
        evaluation_results = {
            "evaluation_timestamp": "2026-01-01T00:00:00",
            "clustering_result": clustering_result,
        }
        summary = evaluator._generate_evaluation_summary(evaluation_results)
        assert summary["n_samples"] == len(labels)

    def test_clustering_algorithm_is_set(self):
        """clustering_algorithm matches the result's algorithm."""
        evaluator = self._make_evaluator()
        clustering_result = self._make_clustering_result([0, 1, 2])
        clustering_result.algorithm = "dbscan"
        evaluation_results = {
            "evaluation_timestamp": "2026-01-01T00:00:00",
            "clustering_result": clustering_result,
        }
        summary = evaluator._generate_evaluation_summary(evaluation_results)
        assert summary["clustering_algorithm"] == "dbscan"

    def test_quality_metrics_section_when_present(self):
        """Summary includes quality_metrics section when quality_metrics is in evaluation_results."""
        evaluator = self._make_evaluator()
        clustering_result = self._make_clustering_result([0, 1])
        evaluation_results = {
            "evaluation_timestamp": "2026-01-01T00:00:00",
            "clustering_result": clustering_result,
            "quality_metrics": {
                "cluster_statistics": {},
                "distance_metrics": {},
            },
        }
        summary = evaluator._generate_evaluation_summary(evaluation_results)
        assert "quality_metrics" in summary
        assert summary["quality_metrics"]["cluster_statistics_available"] is True
        assert summary["quality_metrics"]["distance_metrics_available"] is True

    def test_visual_prototypes_section_when_present(self):
        """Summary includes visual_prototypes section when visual_prototypes is in evaluation_results."""
        evaluator = self._make_evaluator()
        clustering_result = self._make_clustering_result([0, 0, 1])
        evaluation_results = {
            "evaluation_timestamp": "2026-01-01T00:00:00",
            "clustering_result": clustering_result,
            "visual_prototypes": {
                "cluster_0": {"prototype_indices": [0, 1]},
                "cluster_1": {"prototype_indices": [2]},
            },
        }
        summary = evaluator._generate_evaluation_summary(evaluation_results)
        assert "visual_prototypes" in summary
        assert summary["visual_prototypes"]["clusters_with_prototypes"] == 2
        assert summary["visual_prototypes"]["total_prototypes"] == 3


class TestCreateClusteringEvaluator:
    """Tests for create_clustering_evaluator factory function."""

    def test_returns_clustering_evaluator_instance(self):
        """create_clustering_evaluator returns a ClusteringEvaluator."""
        from mammography.eval.clustering_evaluator import (
            ClusteringEvaluator,
            create_clustering_evaluator,
        )
        evaluator = create_clustering_evaluator({})
        assert isinstance(evaluator, ClusteringEvaluator)

    def test_config_applied_to_evaluator(self):
        """Evaluator receives and validates the provided config."""
        from mammography.eval.clustering_evaluator import create_clustering_evaluator
        evaluator = create_clustering_evaluator({"seed": 99})
        assert evaluator.config["seed"] == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])