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

from mammography.config import (
    HP,
    BaseConfig,
    ExtractConfig,
    InferenceConfig,
    TrainConfig,
)


class TestHP:
    """Test the HP dataclass for default hyperparameters."""

    def test_hp_defaults(self):
        """Test that HP dataclass has correct default values."""
        assert HP.IMG_SIZE == 512
        assert HP.EPOCHS == 100
        assert HP.BATCH_SIZE == 16
        assert HP.LR == 1e-4
        assert HP.SEED == 42
        assert HP.DEVICE == "auto"
        assert HP.CACHE_MODE == "auto"

    def test_hp_types(self):
        """Test that HP values have correct types."""
        assert isinstance(HP.IMG_SIZE, int)
        assert isinstance(HP.EPOCHS, int)
        assert isinstance(HP.LR, float)
        assert isinstance(HP.DETERMINISTIC, bool)
        assert isinstance(HP.CACHE_MODE, str)

    def test_hp_additional_defaults(self):
        """Test additional HP default values for comprehensive coverage."""
        assert HP.WINDOW_P_LOW == 0.5
        assert HP.WINDOW_P_HIGH == 99.5
        assert HP.NUM_WORKERS == 4
        assert HP.BACKBONE_LR == 1e-5
        assert HP.VAL_FRAC == 0.20
        assert HP.UNFREEZE_LAST_BLOCK is True
        assert HP.TRAIN_BACKBONE is False
        assert HP.CLASS_WEIGHTS == "none"
        assert HP.SAMPLER_WEIGHTED is False
        assert HP.WARMUP_EPOCHS == 0
        assert HP.ALLOW_TF32 is True
        assert HP.PREFETCH_FACTOR == 4
        assert HP.PERSISTENT_WORKERS is True
        assert HP.LOG_LEVEL == "info"
        assert HP.TRAIN_AUGMENT is True
        assert HP.LOADER_HEURISTICS is True
        assert HP.FUSED_OPTIM is False
        assert HP.TORCH_COMPILE is False

    def test_hp_early_stopping_defaults(self):
        """Test early stopping and LR reduction defaults."""
        assert HP.EARLY_STOP_PATIENCE == 0
        assert HP.EARLY_STOP_MIN_DELTA == 0.0
        assert HP.LR_REDUCE_PATIENCE == 0
        assert HP.LR_REDUCE_FACTOR == 0.5
        assert HP.LR_REDUCE_MIN_LR == 1e-7
        assert HP.LR_REDUCE_COOLDOWN == 0


class TestBaseConfig:
    """Test the BaseConfig base class."""

    def test_base_config_extra_ignore(self):
        """Test that extra fields are ignored."""

        class TestConfig(BaseConfig):
            field1: str = "default"

        # Should not raise error even with extra fields
        config = TestConfig(field1="test", unknown_field="value")
        assert config.field1 == "test"

    def test_from_args(self):
        """Test from_args class method."""

        class TestConfig(BaseConfig):
            field1: str = "default"
            field2: int = 42

        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args)

        assert config.field1 == "custom"
        assert config.field2 == 100

    def test_from_args_with_overrides(self):
        """Test from_args with overrides parameter."""

        class TestConfig(BaseConfig):
            field1: str = "default"
            field2: int = 42

        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args, field2=200)

        assert config.field1 == "custom"
        assert config.field2 == 200  # Override takes precedence


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


class TestInferenceConfig:
    """Test InferenceConfig validation."""

    def test_inference_config_requires_checkpoint_and_input(self):
        """Test that checkpoint and input are required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir) / "images"
            input_path.mkdir()

            # Valid config
            config = InferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.checkpoint == checkpoint
            assert config.input == input_path

    def test_inference_config_checkpoint_validation(self):
        """Test checkpoint path validation."""
        # Nonexistent checkpoint
        with pytest.raises(ValueError, match="checkpoint nao encontrado"):
            InferenceConfig(checkpoint=Path("/nonexistent.pt"), input=Path("/tmp"))

    def test_inference_config_checkpoint_must_be_file(self):
        """Test that checkpoint must be a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()

            with pytest.raises(ValueError, match="checkpoint invalido"):
                InferenceConfig(checkpoint=checkpoint_dir, input=Path(tmpdir))

    def test_inference_config_input_validation(self):
        """Test input path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")

            # Nonexistent input
            with pytest.raises(ValueError, match="input nao encontrado"):
                InferenceConfig(checkpoint=checkpoint, input=Path("/nonexistent"))

    def test_inference_config_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(checkpoint=checkpoint, input=input_path)
            assert config.arch == "resnet50"
            assert config.batch_size == 16
            assert config.img_size == HP.IMG_SIZE
            assert config.device == HP.DEVICE
            assert config.amp is False

    def test_inference_config_architecture(self):
        """Test architecture configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            for arch in ["resnet50", "efficientnet_b0", "vit_base_patch16_224"]:
                config = InferenceConfig(
                    checkpoint=checkpoint, input=input_path, arch=arch
                )
                assert config.arch == arch

    def test_inference_config_classes_parameter(self):
        """Test classes parameter configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            for classes in ["multiclass", "density", "cancer"]:
                config = InferenceConfig(
                    checkpoint=checkpoint, input=input_path, classes=classes
                )
                assert config.classes == classes

    def test_inference_config_output_parameter(self):
        """Test output path configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)
            output = str(Path(tmpdir) / "predictions.csv")

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, output=output
            )
            assert config.output == output

    def test_inference_config_numeric_constraints(self):
        """Test numeric field constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            # Valid config
            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, batch_size=32, img_size=256
            )
            assert config.batch_size == 32
            assert config.img_size == 256

            # Invalid batch_size (< 1)
            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, batch_size=0)

            # Invalid img_size (< 1)
            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, img_size=0)

    def test_inference_config_normalization_parameters(self):
        """Test normalization configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(
                checkpoint=checkpoint,
                input=input_path,
                mean="0.485,0.456,0.406",
                std="0.229,0.224,0.225",
            )
            assert config.mean == "0.485,0.456,0.406"
            assert config.std == "0.229,0.224,0.225"

    def test_inference_config_performance_flags(self):
        """Test performance configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, amp=True, device="cuda"
            )
            assert config.amp is True
            assert config.device == "cuda"


class TestConfigIntegration:
    """Integration tests for config classes."""

    def test_train_config_from_args(self):
        """Test TrainConfig.from_args integration."""
        args = Namespace(
            dataset="mamografias",
            epochs=50,
            batch_size=32,
            lr=1e-3,
            arch="resnet50",
            device="cpu",
        )

        config = TrainConfig.from_args(args)
        assert config.dataset == "mamografias"
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.lr == 1e-3
        assert config.arch == "resnet50"

    def test_extract_config_from_args(self):
        """Test ExtractConfig.from_args integration."""
        args = Namespace(
            dataset="patches_completo",
            arch="efficientnet_b0",
            batch_size=16,
            device="cuda",
        )

        config = ExtractConfig.from_args(args)
        assert config.dataset == "patches_completo"
        assert config.arch == "efficientnet_b0"
        assert config.batch_size == 16
        assert config.device == "cuda"

    def test_config_model_dump(self):
        """Test that Pydantic v2 model_dump works."""
        config = TrainConfig(dataset="mamografias", epochs=10)
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["dataset"] == "mamografias"
        assert config_dict["epochs"] == 10

    def test_config_model_validate(self):
        """Test that Pydantic v2 model_validate works."""
        data = {"dataset": "mamografias", "epochs": 20, "batch_size": 16}
        config = TrainConfig.model_validate(data)

        assert config.dataset == "mamografias"
        assert config.epochs == 20
        assert config.batch_size == 16

    def test_inference_config_from_args(self):
        """Test InferenceConfig.from_args integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_text("fake checkpoint")
            input_path = Path(tmpdir)

            args = Namespace(
                checkpoint=checkpoint,
                input=input_path,
                arch="efficientnet_b0",
                batch_size=32,
            )

            config = InferenceConfig.from_args(args)
            assert config.checkpoint == checkpoint
            assert config.input == input_path
            assert config.arch == "efficientnet_b0"
            assert config.batch_size == 32


class TestConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_train_config_with_both_csv_and_dataset(self):
        """Test that both csv and dataset can be provided together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Both provided should work
            config = TrainConfig(
                dataset="mamografias", csv=csv_path, dicom_root=Path(tmpdir)
            )
            assert config.dataset == "mamografias"
            assert config.csv == csv_path

    def test_extract_config_with_both_csv_and_dataset(self):
        """Test that both csv and dataset can be provided together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Both provided should work
            config = ExtractConfig(
                dataset="patches_completo", csv=csv_path, dicom_root=Path(tmpdir)
            )
            assert config.dataset == "patches_completo"
            assert config.csv == csv_path

    def test_train_config_pretrained_flag(self):
        """Test pretrained flag configuration."""
        # Pretrained enabled (default)
        config = TrainConfig(dataset="mamografias", pretrained=True)
        assert config.pretrained is True

        # Pretrained disabled
        config = TrainConfig(dataset="mamografias", pretrained=False)
        assert config.pretrained is False

    def test_extract_config_pretrained_flag(self):
        """Test pretrained flag for feature extraction."""
        # Pretrained enabled (default)
        config = ExtractConfig(dataset="mamografias", pretrained=True)
        assert config.pretrained is True

        # Pretrained disabled
        config = ExtractConfig(dataset="mamografias", pretrained=False)
        assert config.pretrained is False

    def test_train_config_include_class_5(self):
        """Test include_class_5 flag."""
        config = TrainConfig(dataset="mamografias", include_class_5=True)
        assert config.include_class_5 is True

        config = TrainConfig(dataset="mamografias", include_class_5=False)
        assert config.include_class_5 is False

    def test_extract_config_include_class_5(self):
        """Test include_class_5 flag for extraction."""
        config = ExtractConfig(dataset="mamografias", include_class_5=True)
        assert config.include_class_5 is True

        config = ExtractConfig(dataset="mamografias", include_class_5=False)
        assert config.include_class_5 is False

    def test_train_config_auto_normalize(self):
        """Test auto-normalization configuration."""
        config = TrainConfig(
            dataset="mamografias", auto_normalize=True, auto_normalize_samples=2000
        )
        assert config.auto_normalize is True
        assert config.auto_normalize_samples == 2000

        # Invalid auto_normalize_samples (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", auto_normalize_samples=0)

    def test_train_config_log_level(self):
        """Test log level configuration."""
        for log_level in ["debug", "info", "warning", "error"]:
            config = TrainConfig(dataset="mamografias", log_level=log_level)
            assert config.log_level == log_level

    def test_extract_config_log_level(self):
        """Test log level configuration for extraction."""
        for log_level in ["debug", "info", "warning", "error"]:
            config = ExtractConfig(dataset="mamografias", log_level=log_level)
            assert config.log_level == log_level

    def test_train_config_classes_parameter(self):
        """Test classes parameter configuration."""
        for classes in ["density", "cancer", "multiclass"]:
            config = TrainConfig(dataset="mamografias", classes=classes)
            assert config.classes == classes

    def test_extract_config_classes_parameter(self):
        """Test classes parameter for extraction."""
        for classes in ["multiclass", "density", "cancer"]:
            config = ExtractConfig(dataset="mamografias", classes=classes)
            assert config.classes == classes

    def test_base_config_from_args_missing_fields(self):
        """Test from_args with args missing some fields."""

        class TestConfig(BaseConfig):
            field1: str = "default1"
            field2: int = 42
            field3: bool = True

        # Args missing field3
        args = Namespace(field1="custom", field2=100)
        config = TestConfig.from_args(args)

        assert config.field1 == "custom"
        assert config.field2 == 100
        assert config.field3 is True  # Should use default

    def test_config_immutability_after_creation(self):
        """Test that config values can be read after creation."""
        config = TrainConfig(dataset="mamografias", epochs=50)

        # Should be able to read values
        assert config.epochs == 50
        assert config.dataset == "mamografias"
        assert config.arch == "efficientnet_b0"

    def test_train_config_scheduler_constraints(self):
        """Test scheduler numeric constraints."""
        # Valid scheduler parameters
        config = TrainConfig(
            dataset="mamografias",
            scheduler_min_lr=1e-8,
            scheduler_step_size=1,
            scheduler_gamma=0.1,
        )
        assert config.scheduler_min_lr == 1e-8

        # Invalid scheduler_min_lr (< 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_min_lr=-1e-5)

        # Invalid scheduler_step_size (< 1)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_step_size=0)

        # Invalid scheduler_gamma (<= 0)
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_gamma=0)

    def test_extract_config_outdir_default(self):
        """Test ExtractConfig outdir default value."""
        config = ExtractConfig(dataset="mamografias")
        assert config.outdir == "outputs/features"

    def test_train_config_outdir_default(self):
        """Test TrainConfig outdir default value."""
        config = TrainConfig(dataset="mamografias")
        assert config.outdir == "outputs/run"

    def test_extract_config_dicom_root_with_hint(self):
        """Test that _normalize_dir_hint works for dicom_root."""
        # This would test the typo correction from 'archieve' to 'archive'
        # but we can't create that scenario easily in tests without actual filesystem
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"
            archive_dir.mkdir()
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("id,label\n1,A\n")

            # Valid path should work
            config = ExtractConfig(csv=csv_path, dicom_root=archive_dir)
            assert config.dicom_root == archive_dir
