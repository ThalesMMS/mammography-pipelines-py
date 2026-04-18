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

class TestTrainConfig:
    """Test TrainConfig validation."""

    def test_train_config_defaults(self):
        """Test that TrainConfig initializes with defaults."""
        config = TrainConfig(dataset="mamografias")
        assert config.dataset == "mamografias"
        assert config.epochs == HP.EPOCHS
        assert config.batch_size == HP.BATCH_SIZE
        assert config.lr == HP.LR
        assert config.arch == "efficientnet_b0"

    def test_train_config_validation_bounds(self):
        """Test that Field constraints are validated."""
        # Valid config
        config = TrainConfig(dataset="mamografias", epochs=10, batch_size=16)
        assert config.epochs == 10

        # Invalid epochs (< 1)
        with pytest.raises(Exception):  # Pydantic ValidationError
            TrainConfig(dataset="mamografias", epochs=0)

        # Invalid batch_size (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", batch_size=0)

        # Invalid val_frac (not in range 0-1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", val_frac=1.5)

    def test_train_config_requires_dataset_or_csv(self):
        """Test that either dataset or csv must be provided."""
        # No dataset or csv - should fail validation
        with pytest.raises(ValueError, match="Informe --csv ou --dataset para treinar"):
            TrainConfig()

        # With dataset - should succeed
        config = TrainConfig(dataset="mamografias")
        assert config.dataset == "mamografias"

    def test_train_config_csv_path_validation(self):
        """Test that csv path is validated if provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Valid CSV path
            config = TrainConfig(csv=csv_path, dicom_root=Path(tmpdir))
            assert config.csv == csv_path

        # Invalid CSV path - should raise error
        with pytest.raises(ValueError, match="Caminho critico nao encontrado"):
            TrainConfig(csv=Path("/nonexistent/path.csv"), dataset=None)

    def test_train_config_numeric_constraints(self):
        """Test numeric field constraints."""
        # Valid lr (gt=0)
        config = TrainConfig(dataset="mamografias", lr=1e-5)
        assert config.lr == 1e-5

        # Invalid lr (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr=0)

        # Valid weight_decay (ge=0)
        config = TrainConfig(dataset="mamografias", weight_decay=0)
        assert config.weight_decay == 0

        # Invalid weight_decay (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", weight_decay=-0.1)

    def test_train_config_rejects_transformer_with_non_224_img_size(self):
        """Transformer backbones should require 224x224 inputs."""
        with pytest.raises(ValueError, match="requires img_size=224"):
            TrainConfig(dataset="mamografias", arch="vit_b_16", img_size=64)

    def test_train_config_view_specific_training(self):
        """Test view-specific training configuration."""
        # Valid view-specific training config
        config = TrainConfig(
            dataset="mamografias",
            view_specific_training=True,
            views_to_train=["CC", "MLO"],
        )
        assert config.view_specific_training is True
        assert config.views_to_train == ["CC", "MLO"]

        # Without view-specific training
        config = TrainConfig(dataset="mamografias", view_specific_training=False)
        assert config.view_specific_training is False
        assert config.views_to_train is None

    def test_train_config_scheduler_parameters(self):
        """Test scheduler configuration parameters."""
        config = TrainConfig(
            dataset="mamografias",
            scheduler="cosine",
            scheduler_min_lr=1e-6,
            scheduler_step_size=10,
            scheduler_gamma=0.9,
        )
        assert config.scheduler == "cosine"
        assert config.scheduler_min_lr == 1e-6
        assert config.scheduler_step_size == 10
        assert config.scheduler_gamma == 0.9

    def test_train_config_early_stopping_parameters(self):
        """Test early stopping configuration."""
        config = TrainConfig(
            dataset="mamografias",
            early_stop_patience=10,
            early_stop_min_delta=0.001,
        )
        assert config.early_stop_patience == 10
        assert config.early_stop_min_delta == 0.001

        # Invalid patience (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", early_stop_patience=-1)

        # Invalid min_delta (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", early_stop_min_delta=-0.01)

    def test_train_config_lr_reduction_parameters(self):
        """Test learning rate reduction configuration."""
        config = TrainConfig(
            dataset="mamografias",
            lr_reduce_patience=5,
            lr_reduce_factor=0.3,
            lr_reduce_min_lr=1e-6,
            lr_reduce_cooldown=2,
        )
        assert config.lr_reduce_patience == 5
        assert config.lr_reduce_factor == 0.3
        assert config.lr_reduce_min_lr == 1e-6
        assert config.lr_reduce_cooldown == 2

    def test_train_config_augmentation_parameters(self):
        """Test augmentation configuration."""
        config = TrainConfig(
            dataset="mamografias",
            augment=True,
            augment_vertical=True,
            augment_color=True,
            augment_rotation_deg=15.0,
        )
        assert config.augment is True
        assert config.augment_vertical is True
        assert config.augment_color is True
        assert config.augment_rotation_deg == 15.0

        # Invalid rotation degree (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", augment_rotation_deg=-5.0)

    def test_train_config_augmentation_warning(self):
        """Test that augmentation warning is raised when augment=False but parameters set."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainConfig(
                dataset="mamografias",
                augment=False,
                augment_vertical=True,
                augment_color=True,
            )
            assert len(w) == 1
            assert "augment=False" in str(w[0].message)
            assert config.augment is False

    def test_train_config_gradcam_parameters(self):
        """Test Grad-CAM configuration."""
        config = TrainConfig(dataset="mamografias", gradcam=True, gradcam_limit=10)
        assert config.gradcam is True
        assert config.gradcam_limit == 10

        # Invalid gradcam_limit (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", gradcam_limit=0)

    def test_train_config_gradcam_warning(self):
        """Test that gradcam warning is raised when gradcam=False but limit set."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainConfig(
                dataset="mamografias", gradcam=False, gradcam_limit=10
            )
            assert len(w) == 1
            assert "gradcam=False" in str(w[0].message)
            assert config.gradcam is False

    def test_train_config_output_directories(self):
        """Test output directory validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = str(Path(tmpdir) / "output")
            config = TrainConfig(dataset="mamografias", outdir=outdir)
            assert config.outdir == outdir

            # Profile directory
            profile_dir = str(Path(tmpdir) / "profiling")
            config = TrainConfig(
                dataset="mamografias", profile=True, profile_dir=profile_dir
            )
            assert config.profile_dir == profile_dir

    def test_train_config_embeddings_dir_validation(self):
        """Test embeddings directory validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_dir = Path(tmpdir) / "embeddings"
            embeddings_dir.mkdir()

            # Valid embeddings_dir
            config = TrainConfig(dataset="mamografias", embeddings_dir=embeddings_dir)
            assert config.embeddings_dir == embeddings_dir

        # Invalid embeddings_dir
        with pytest.raises(ValueError, match="embeddings_dir nao encontrado"):
            TrainConfig(
                dataset="mamografias", embeddings_dir=Path("/nonexistent/embeddings")
            )

    def test_train_config_cache_mode_options(self):
        """Test cache mode configuration options."""
        for cache_mode in ["auto", "memory", "disk", "tensor-disk", "tensor-memmap"]:
            config = TrainConfig(dataset="mamografias", cache_mode=cache_mode)
            assert config.cache_mode == cache_mode

    def test_train_config_class_weights_options(self):
        """Test class weights configuration."""
        for class_weights in ["none", "balanced", "inverse", "effective"]:
            config = TrainConfig(dataset="mamografias", class_weights=class_weights)
            assert config.class_weights == class_weights

        # Test class_weights_alpha
        config = TrainConfig(dataset="mamografias", class_weights_alpha=0.5)
        assert config.class_weights_alpha == 0.5

        # Invalid class_weights_alpha (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", class_weights_alpha=0)

    def test_train_config_sampler_configuration(self):
        """Test weighted sampler configuration."""
        config = TrainConfig(
            dataset="mamografias", sampler_weighted=True, sampler_alpha=2.0
        )
        assert config.sampler_weighted is True
        assert config.sampler_alpha == 2.0

        # Invalid sampler_alpha (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", sampler_alpha=0)

    def test_train_config_backbone_training(self):
        """Test backbone training configuration."""
        config = TrainConfig(
            dataset="mamografias",
            train_backbone=True,
            unfreeze_last_block=False,
            backbone_lr=1e-6,
        )
        assert config.train_backbone is True
        assert config.unfreeze_last_block is False
        assert config.backbone_lr == 1e-6

        # Invalid backbone_lr (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", backbone_lr=0)

    def test_train_config_warmup_epochs(self):
        """Test warmup epochs configuration."""
        config = TrainConfig(dataset="mamografias", warmup_epochs=5)
        assert config.warmup_epochs == 5

        # Invalid warmup_epochs (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", warmup_epochs=-1)

    def test_train_config_performance_flags(self):
        """Test performance optimization flags."""
        config = TrainConfig(
            dataset="mamografias",
            amp=True,
            deterministic=True,
            allow_tf32=False,
            fused_optim=True,
            torch_compile=True,
        )
        assert config.amp is True
        assert config.deterministic is True
        assert config.allow_tf32 is False
        assert config.fused_optim is True
        assert config.torch_compile is True

    def test_train_config_loader_parameters(self):
        """Test data loader configuration."""
        config = TrainConfig(
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
            TrainConfig(dataset="mamografias", num_workers=-1)

        # Invalid prefetch_factor (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", prefetch_factor=-1)

    def test_train_config_validation_output_flags(self):
        """Test validation output configuration."""
        config = TrainConfig(
            dataset="mamografias",
            save_val_preds=True,
            export_val_embeddings=True,
        )
        assert config.save_val_preds is True
        assert config.export_val_embeddings is True

    def test_train_config_subset_parameter(self):
        """Test subset parameter for limited training."""
        config = TrainConfig(dataset="mamografias", subset=100)
        assert config.subset == 100

        # Invalid subset (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", subset=-1)

    def test_train_config_split_parameters(self):
        """Test dataset split configuration."""
        config = TrainConfig(
            dataset="mamografias",
            val_frac=0.25,
            split_ensure_all_classes=False,
            split_max_tries=500,
        )
        assert config.val_frac == 0.25
        assert config.split_ensure_all_classes is False
        assert config.split_max_tries == 500

        # Invalid split_max_tries (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", split_max_tries=0)
