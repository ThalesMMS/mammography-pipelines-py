#!/usr/bin/env python3
#
# test_pipeline_storage.py
# mammography-pipelines
#
# Unit tests for PipelineStorageMixin in pipeline/storage.py.
#
"""Unit tests for mammography.pipeline.storage module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class _StorageTestHelper:
    """Concrete helper that inherits PipelineStorageMixin for testing."""

    def __init__(self, config=None):
        self.config = config or {}


def _make_storage(config=None):
    """Create a minimal object with PipelineStorageMixin methods."""
    from mammography.pipeline.storage import PipelineStorageMixin

    class ConcreteStorage(PipelineStorageMixin):
        def __init__(self):
            self.config = config or {}

    return ConcreteStorage()


class TestGetPreprocessingOutputPath:
    """Tests for PipelineStorageMixin._get_preprocessing_output_path."""

    def test_returns_path_with_pt_suffix(self, tmp_path):
        """Output path has .pt suffix."""
        storage = _make_storage()
        # Create a nested input path at depth >= 2 so relative_to works
        input_dir = tmp_path / "archive" / "patient1"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "image.dcm"
        input_path.touch()
        output_dir = tmp_path / "preprocessed"
        result = storage._get_preprocessing_output_path(input_path, output_dir)
        assert result.suffix == ".pt"

    def test_output_is_under_output_dir(self, tmp_path):
        """Output path is located under the specified output_dir."""
        storage = _make_storage()
        input_dir = tmp_path / "archive" / "p1"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "img.dcm"
        input_path.touch()
        output_dir = tmp_path / "out"
        result = storage._get_preprocessing_output_path(input_path, output_dir)
        assert str(result).startswith(str(output_dir))

    def test_parent_directories_are_created(self, tmp_path):
        """Ensures parent directory of output path is created."""
        storage = _make_storage()
        input_dir = tmp_path / "archive" / "patient_x"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "slice.dcm"
        input_path.touch()
        output_dir = tmp_path / "tensors"
        result = storage._get_preprocessing_output_path(input_path, output_dir)
        assert result.parent.exists()

    def test_preserves_stem_from_input(self, tmp_path):
        """Output path stem matches the input file stem."""
        storage = _make_storage()
        input_dir = tmp_path / "data" / "patient2"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "unique_name.dcm"
        input_path.touch()
        output_dir = tmp_path / "output"
        result = storage._get_preprocessing_output_path(input_path, output_dir)
        assert result.stem == "unique_name"


class TestGetEmbeddingOutputPath:
    """Tests for PipelineStorageMixin._get_embedding_output_path."""

    def test_returns_path_with_embedding_pt_suffix(self, tmp_path):
        """Output path ends with '.embedding.pt'."""
        storage = _make_storage()
        input_dir = tmp_path / "archive" / "patient1"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "image.dcm"
        input_path.touch()
        output_dir = tmp_path / "embeddings"
        result = storage._get_embedding_output_path(input_path, output_dir)
        assert str(result).endswith(".embedding.pt")

    def test_output_is_under_output_dir(self, tmp_path):
        """Output path is located under the specified output_dir."""
        storage = _make_storage()
        input_dir = tmp_path / "archive" / "p1"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "img.dcm"
        input_path.touch()
        output_dir = tmp_path / "embeddings_out"
        result = storage._get_embedding_output_path(input_path, output_dir)
        assert str(result).startswith(str(output_dir))

    def test_parent_directories_are_created(self, tmp_path):
        """Ensures parent directory of embedding output path is created."""
        storage = _make_storage()
        input_dir = tmp_path / "archive" / "patX"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "frame.dcm"
        input_path.touch()
        output_dir = tmp_path / "emb"
        result = storage._get_embedding_output_path(input_path, output_dir)
        assert result.parent.exists()


class TestSavePreprocessedTensor:
    """Tests for PipelineStorageMixin._save_preprocessed_tensor."""

    def _make_preprocessed_tensor(self):
        """Create a mock PreprocessedTensor for testing."""
        import torch
        mock_tensor = MagicMock()
        mock_tensor.tensor_data = torch.zeros(3, 224, 224)
        mock_tensor.image_id = "test_image_001"
        mock_tensor.preprocessing_config = {"normalize": True}
        mock_tensor.normalization_method = "z_score_per_image"
        mock_tensor.target_size = (224, 224)
        mock_tensor.input_adapter = "1to3_replication"
        mock_tensor.border_removed = False
        mock_tensor.original_shape = (512, 512)
        mock_tensor.processing_time = 0.5
        mock_tensor.created_at = datetime(2026, 1, 1, 12, 0, 0)
        return mock_tensor

    def test_returns_true_on_success(self, tmp_path):
        """Returns True when tensor is saved successfully."""
        storage = _make_storage()
        mock_tensor_obj = self._make_preprocessed_tensor()
        output_path = tmp_path / "tensor.pt"
        with patch("mammography.pipeline.storage.torch.save"):
            with patch("builtins.open", create=True) as mock_open:
                import io
                mock_open.return_value.__enter__ = lambda s: io.StringIO()
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                with patch("mammography.pipeline.storage.yaml.dump"):
                    result = storage._save_preprocessed_tensor(mock_tensor_obj, output_path)
        assert result is True

    def test_returns_false_on_error(self, tmp_path):
        """Returns False when an exception occurs during saving."""
        storage = _make_storage()
        mock_tensor_obj = self._make_preprocessed_tensor()
        output_path = tmp_path / "tensor.pt"
        with patch("mammography.pipeline.storage.torch.save", side_effect=OSError("disk full")):
            result = storage._save_preprocessed_tensor(mock_tensor_obj, output_path)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])