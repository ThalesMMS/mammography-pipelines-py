# ruff: noqa
#
# test_config.py
# mammography-pipelines
#
# Tests for Pydantic config validation in config.py
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from mammography.config import (
    HP,
    BaseConfig,
    BatchInferenceConfig,
    ExtractConfig,
    InferenceConfig,
    PreprocessConfig,
    TrainConfig,
)

class TestExtractConfig:
    """Test ExtractConfig validation."""

    def test_extract_config_defaults(self):
        """Test that ExtractConfig initializes with defaults."""
        config = ExtractConfig(dataset="mamografias")
        assert config.dataset == "mamografias"
        assert config.arch == "resnet50"
        assert config.batch_size == 32
        assert config.img_size == HP.IMG_SIZE

    def test_extract_config_requires_dataset_or_csv(self):
        """Test that either dataset or csv must be provided."""
        # No dataset or csv - should fail
        with pytest.raises(
            ValueError, match="Informe --csv ou --dataset para extrair embeddings"
        ):
            ExtractConfig()

        # With dataset - should succeed
        config = ExtractConfig(dataset="patches_completo")
        assert config.dataset == "patches_completo"

    def test_extract_config_csv_validation(self):
        """Test CSV path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Valid CSV
            config = ExtractConfig(csv=csv_path, dicom_root=Path(tmpdir))
            assert config.csv == csv_path

        # Invalid CSV path
        with pytest.raises(ValueError, match="csv_path nao encontrado"):
            ExtractConfig(csv=Path("/nonexistent.csv"), dataset=None)

    def test_extract_config_numeric_constraints(self):
        """Test numeric field constraints."""
        config = ExtractConfig(dataset="mamografias", batch_size=16, img_size=224)
        assert config.batch_size == 16
        assert config.img_size == 224

        # Invalid batch_size
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", batch_size=0)

        # Invalid img_size
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", img_size=0)

    def test_extract_config_architecture_options(self):
        """Test architecture configuration."""
        for arch in ["resnet50", "efficientnet_b0", "vit_base_patch16_224"]:
            config = ExtractConfig(dataset="mamografias", arch=arch)
            assert config.arch == arch

    def test_extract_config_layer_name(self):
        """Test layer name configuration for feature extraction."""
        config = ExtractConfig(dataset="mamografias", layer_name="layer4")
        assert config.layer_name == "layer4"

        # Default layer_name
        config = ExtractConfig(dataset="mamografias")
        assert config.layer_name == "avgpool"

    def test_extract_config_reduction_flags(self):
        """Test dimensionality reduction flags."""
        config = ExtractConfig(
            dataset="mamografias",
            run_reduction=True,
            pca=True,
            tsne=True,
            umap=True,
        )
        assert config.run_reduction is True
        assert config.pca is True
        assert config.tsne is True
        assert config.umap is True

    def test_extract_config_pca_parameters(self):
        """Test PCA-specific configuration."""
        for svd_solver in ["auto", "full", "arpack", "randomized"]:
            config = ExtractConfig(
                dataset="mamografias", pca=True, pca_svd_solver=svd_solver
            )
            assert config.pca_svd_solver == svd_solver

    def test_extract_config_clustering_flags(self):
        """Test clustering configuration."""
        config = ExtractConfig(
            dataset="mamografias",
            run_clustering=True,
            cluster_auto=True,
            cluster_k=5,
            n_clusters=4,
        )
        assert config.run_clustering is True
        assert config.cluster_auto is True
        assert config.cluster_k == 5
        assert config.n_clusters == 4

        # Invalid cluster_k (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", cluster_k=-1)

        # Invalid n_clusters (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", n_clusters=-1)

    def test_extract_config_sample_grid(self):
        """Test sample grid parameter."""
        config = ExtractConfig(dataset="mamografias", sample_grid=32)
        assert config.sample_grid == 32

        # Invalid sample_grid (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", sample_grid=-1)

    def test_extract_config_output_flags(self):
        """Test output configuration flags."""
        config = ExtractConfig(dataset="mamografias", save_csv=True)
        assert config.save_csv is True

    def test_extract_config_loader_parameters(self):
        """Test data loader configuration."""
        config = ExtractConfig(
            dataset="mamografias",
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=False,
            loader_heuristics=False,
        )
        assert config.num_workers == 8
        assert config.prefetch_factor == 8
        assert config.persistent_workers is False
        assert config.loader_heuristics is False

        # Invalid num_workers (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", num_workers=-1)

        # Invalid prefetch_factor (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", prefetch_factor=-1)

    def test_extract_config_cache_mode(self):
        """Test cache mode configuration."""
        for cache_mode in ["auto", "memory", "disk", "tensor-disk", "tensor-memmap"]:
            config = ExtractConfig(dataset="mamografias", cache_mode=cache_mode)
            assert config.cache_mode == cache_mode

    def test_extract_config_performance_flags(self):
        """Test performance optimization flags."""
        config = ExtractConfig(
            dataset="mamografias",
            amp=True,
            deterministic=True,
            allow_tf32=False,
        )
        assert config.amp is True
        assert config.deterministic is True
        assert config.allow_tf32 is False

    def test_extract_config_normalization_parameters(self):
        """Test normalization configuration."""
        config = ExtractConfig(
            dataset="mamografias", mean="0.5,0.5,0.5", std="0.25,0.25,0.25"
        )
        assert config.mean == "0.5,0.5,0.5"
        assert config.std == "0.25,0.25,0.25"

    def test_extract_config_seed_parameter(self):
        """Test seed configuration for reproducibility."""
        config = ExtractConfig(dataset="mamografias", seed=123)
        assert config.seed == 123

        # Invalid seed (< 0)
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", seed=-1)
