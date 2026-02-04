#
# test_boundary_conditions.py
# mammography-pipelines
#
# Tests for boundary conditions and edge values in configuration parameters.
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
import tempfile
from pathlib import Path

import pytest

from mammography.config import ExtractConfig, InferenceConfig, TrainConfig


class TestTrainConfigBoundaries:
    """Test boundary conditions for TrainConfig numeric fields."""

    def test_epochs_minimum_valid(self):
        """Test that epochs=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", epochs=1)
        assert config.epochs == 1

    def test_epochs_zero_invalid(self):
        """Test that epochs=0 is invalid (ge=1)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TrainConfig(dataset="mamografias", epochs=0)

    def test_epochs_negative_invalid(self):
        """Test that negative epochs are invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", epochs=-1)

    def test_batch_size_minimum_valid(self):
        """Test that batch_size=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", batch_size=1)
        assert config.batch_size == 1

    def test_batch_size_zero_invalid(self):
        """Test that batch_size=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", batch_size=0)

    def test_batch_size_negative_invalid(self):
        """Test that negative batch_size is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", batch_size=-1)

    def test_lr_very_small_valid(self):
        """Test that very small positive lr is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", lr=1e-10)
        assert config.lr == 1e-10

    def test_lr_zero_invalid(self):
        """Test that lr=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr=0)

    def test_lr_negative_invalid(self):
        """Test that negative lr is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr=-0.001)

    def test_backbone_lr_very_small_valid(self):
        """Test that very small positive backbone_lr is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", backbone_lr=1e-10)
        assert config.backbone_lr == 1e-10

    def test_backbone_lr_zero_invalid(self):
        """Test that backbone_lr=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", backbone_lr=0)

    def test_backbone_lr_negative_invalid(self):
        """Test that negative backbone_lr is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", backbone_lr=-1e-5)

    def test_weight_decay_zero_valid(self):
        """Test that weight_decay=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", weight_decay=0)
        assert config.weight_decay == 0

    def test_weight_decay_negative_invalid(self):
        """Test that negative weight_decay is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", weight_decay=-0.1)

    def test_img_size_minimum_valid(self):
        """Test that img_size=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", img_size=1)
        assert config.img_size == 1

    def test_img_size_zero_invalid(self):
        """Test that img_size=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", img_size=0)

    def test_img_size_negative_invalid(self):
        """Test that negative img_size is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", img_size=-1)

    def test_seed_zero_valid(self):
        """Test that seed=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", seed=0)
        assert config.seed == 0

    def test_seed_negative_invalid(self):
        """Test that negative seed is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", seed=-1)

    def test_val_frac_lower_bound_invalid(self):
        """Test that val_frac=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", val_frac=0)

    def test_val_frac_upper_bound_invalid(self):
        """Test that val_frac=1 is invalid (lt=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", val_frac=1)

    def test_val_frac_very_small_valid(self):
        """Test that very small positive val_frac is valid (gt=0, lt=1)."""
        config = TrainConfig(dataset="mamografias", val_frac=0.001)
        assert config.val_frac == 0.001

    def test_val_frac_near_one_valid(self):
        """Test that val_frac near 1.0 is valid (gt=0, lt=1)."""
        config = TrainConfig(dataset="mamografias", val_frac=0.999)
        assert config.val_frac == 0.999

    def test_val_frac_above_one_invalid(self):
        """Test that val_frac>1 is invalid (lt=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", val_frac=1.5)

    def test_val_frac_negative_invalid(self):
        """Test that negative val_frac is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", val_frac=-0.1)

    def test_split_max_tries_minimum_valid(self):
        """Test that split_max_tries=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", split_max_tries=1)
        assert config.split_max_tries == 1

    def test_split_max_tries_zero_invalid(self):
        """Test that split_max_tries=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", split_max_tries=0)

    def test_num_workers_zero_valid(self):
        """Test that num_workers=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", num_workers=0)
        assert config.num_workers == 0

    def test_num_workers_negative_invalid(self):
        """Test that negative num_workers is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", num_workers=-1)

    def test_prefetch_factor_zero_valid(self):
        """Test that prefetch_factor=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", prefetch_factor=0)
        assert config.prefetch_factor == 0

    def test_prefetch_factor_negative_invalid(self):
        """Test that negative prefetch_factor is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", prefetch_factor=-1)

    def test_class_weights_alpha_very_small_valid(self):
        """Test that very small positive class_weights_alpha is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", class_weights_alpha=1e-10)
        assert config.class_weights_alpha == 1e-10

    def test_class_weights_alpha_zero_invalid(self):
        """Test that class_weights_alpha=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", class_weights_alpha=0)

    def test_sampler_alpha_very_small_valid(self):
        """Test that very small positive sampler_alpha is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", sampler_alpha=1e-10)
        assert config.sampler_alpha == 1e-10

    def test_sampler_alpha_zero_invalid(self):
        """Test that sampler_alpha=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", sampler_alpha=0)

    def test_warmup_epochs_zero_valid(self):
        """Test that warmup_epochs=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", warmup_epochs=0)
        assert config.warmup_epochs == 0

    def test_warmup_epochs_negative_invalid(self):
        """Test that negative warmup_epochs is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", warmup_epochs=-1)

    def test_lr_reduce_patience_zero_valid(self):
        """Test that lr_reduce_patience=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", lr_reduce_patience=0)
        assert config.lr_reduce_patience == 0

    def test_lr_reduce_patience_negative_invalid(self):
        """Test that negative lr_reduce_patience is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr_reduce_patience=-1)

    def test_lr_reduce_factor_very_small_valid(self):
        """Test that very small positive lr_reduce_factor is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", lr_reduce_factor=1e-10)
        assert config.lr_reduce_factor == 1e-10

    def test_lr_reduce_factor_zero_invalid(self):
        """Test that lr_reduce_factor=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr_reduce_factor=0)

    def test_lr_reduce_min_lr_zero_valid(self):
        """Test that lr_reduce_min_lr=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", lr_reduce_min_lr=0)
        assert config.lr_reduce_min_lr == 0

    def test_lr_reduce_min_lr_negative_invalid(self):
        """Test that negative lr_reduce_min_lr is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr_reduce_min_lr=-1e-7)

    def test_lr_reduce_cooldown_zero_valid(self):
        """Test that lr_reduce_cooldown=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", lr_reduce_cooldown=0)
        assert config.lr_reduce_cooldown == 0

    def test_lr_reduce_cooldown_negative_invalid(self):
        """Test that negative lr_reduce_cooldown is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", lr_reduce_cooldown=-1)

    def test_scheduler_min_lr_zero_valid(self):
        """Test that scheduler_min_lr=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", scheduler_min_lr=0)
        assert config.scheduler_min_lr == 0

    def test_scheduler_min_lr_negative_invalid(self):
        """Test that negative scheduler_min_lr is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_min_lr=-1e-7)

    def test_scheduler_step_size_minimum_valid(self):
        """Test that scheduler_step_size=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", scheduler_step_size=1)
        assert config.scheduler_step_size == 1

    def test_scheduler_step_size_zero_invalid(self):
        """Test that scheduler_step_size=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_step_size=0)

    def test_scheduler_gamma_very_small_valid(self):
        """Test that very small positive scheduler_gamma is valid (gt=0)."""
        config = TrainConfig(dataset="mamografias", scheduler_gamma=1e-10)
        assert config.scheduler_gamma == 1e-10

    def test_scheduler_gamma_zero_invalid(self):
        """Test that scheduler_gamma=0 is invalid (gt=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", scheduler_gamma=0)

    def test_early_stop_patience_zero_valid(self):
        """Test that early_stop_patience=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", early_stop_patience=0)
        assert config.early_stop_patience == 0

    def test_early_stop_patience_negative_invalid(self):
        """Test that negative early_stop_patience is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", early_stop_patience=-1)

    def test_early_stop_min_delta_zero_valid(self):
        """Test that early_stop_min_delta=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", early_stop_min_delta=0)
        assert config.early_stop_min_delta == 0

    def test_early_stop_min_delta_negative_invalid(self):
        """Test that negative early_stop_min_delta is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", early_stop_min_delta=-0.001)

    def test_augment_rotation_deg_zero_valid(self):
        """Test that augment_rotation_deg=0 is valid (ge=0)."""
        config = TrainConfig(dataset="mamografias", augment_rotation_deg=0)
        assert config.augment_rotation_deg == 0

    def test_augment_rotation_deg_negative_invalid(self):
        """Test that negative augment_rotation_deg is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", augment_rotation_deg=-5.0)

    def test_gradcam_limit_minimum_valid(self):
        """Test that gradcam_limit=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", gradcam_limit=1)
        assert config.gradcam_limit == 1

    def test_gradcam_limit_zero_invalid(self):
        """Test that gradcam_limit=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", gradcam_limit=0)

    def test_subset_zero_valid(self):
        """Test that subset=0 is valid (ge=0) and means no subset."""
        config = TrainConfig(dataset="mamografias", subset=0)
        assert config.subset == 0

    def test_subset_negative_invalid(self):
        """Test that negative subset is invalid (ge=0)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", subset=-1)

    def test_auto_normalize_samples_minimum_valid(self):
        """Test that auto_normalize_samples=1 is the minimum valid value (ge=1)."""
        config = TrainConfig(dataset="mamografias", auto_normalize_samples=1)
        assert config.auto_normalize_samples == 1

    def test_auto_normalize_samples_zero_invalid(self):
        """Test that auto_normalize_samples=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            TrainConfig(dataset="mamografias", auto_normalize_samples=0)


class TestExtractConfigBoundaries:
    """Test boundary conditions for ExtractConfig numeric fields."""

    def test_seed_zero_valid(self):
        """Test that seed=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", seed=0)
        assert config.seed == 0

    def test_seed_negative_invalid(self):
        """Test that negative seed is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", seed=-1)

    def test_img_size_minimum_valid(self):
        """Test that img_size=1 is the minimum valid value (ge=1)."""
        config = ExtractConfig(dataset="mamografias", img_size=1)
        assert config.img_size == 1

    def test_img_size_zero_invalid(self):
        """Test that img_size=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", img_size=0)

    def test_batch_size_minimum_valid(self):
        """Test that batch_size=1 is the minimum valid value (ge=1)."""
        config = ExtractConfig(dataset="mamografias", batch_size=1)
        assert config.batch_size == 1

    def test_batch_size_zero_invalid(self):
        """Test that batch_size=0 is invalid (ge=1)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", batch_size=0)

    def test_num_workers_zero_valid(self):
        """Test that num_workers=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", num_workers=0)
        assert config.num_workers == 0

    def test_num_workers_negative_invalid(self):
        """Test that negative num_workers is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", num_workers=-1)

    def test_prefetch_factor_zero_valid(self):
        """Test that prefetch_factor=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", prefetch_factor=0)
        assert config.prefetch_factor == 0

    def test_prefetch_factor_negative_invalid(self):
        """Test that negative prefetch_factor is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", prefetch_factor=-1)

    def test_cluster_k_zero_valid(self):
        """Test that cluster_k=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", cluster_k=0)
        assert config.cluster_k == 0

    def test_cluster_k_negative_invalid(self):
        """Test that negative cluster_k is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", cluster_k=-1)

    def test_n_clusters_zero_valid(self):
        """Test that n_clusters=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", n_clusters=0)
        assert config.n_clusters == 0

    def test_n_clusters_negative_invalid(self):
        """Test that negative n_clusters is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", n_clusters=-1)

    def test_sample_grid_zero_valid(self):
        """Test that sample_grid=0 is valid (ge=0)."""
        config = ExtractConfig(dataset="mamografias", sample_grid=0)
        assert config.sample_grid == 0

    def test_sample_grid_negative_invalid(self):
        """Test that negative sample_grid is invalid (ge=0)."""
        with pytest.raises(Exception):
            ExtractConfig(dataset="mamografias", sample_grid=-1)


class TestInferenceConfigBoundaries:
    """Test boundary conditions for InferenceConfig numeric fields."""

    def test_img_size_minimum_valid(self):
        """Test that img_size=1 is the minimum valid value (ge=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pth"
            checkpoint.write_text("dummy")
            input_path = Path(tmpdir) / "input.dcm"
            input_path.write_text("dummy")

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, img_size=1
            )
            assert config.img_size == 1

    def test_img_size_zero_invalid(self):
        """Test that img_size=0 is invalid (ge=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pth"
            checkpoint.write_text("dummy")
            input_path = Path(tmpdir) / "input.dcm"
            input_path.write_text("dummy")

            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, img_size=0)

    def test_batch_size_minimum_valid(self):
        """Test that batch_size=1 is the minimum valid value (ge=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pth"
            checkpoint.write_text("dummy")
            input_path = Path(tmpdir) / "input.dcm"
            input_path.write_text("dummy")

            config = InferenceConfig(
                checkpoint=checkpoint, input=input_path, batch_size=1
            )
            assert config.batch_size == 1

    def test_batch_size_zero_invalid(self):
        """Test that batch_size=0 is invalid (ge=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pth"
            checkpoint.write_text("dummy")
            input_path = Path(tmpdir) / "input.dcm"
            input_path.write_text("dummy")

            with pytest.raises(Exception):
                InferenceConfig(checkpoint=checkpoint, input=input_path, batch_size=0)


class TestLargeValueBoundaries:
    """Test very large values to ensure no overflow or unexpected behavior."""

    def test_epochs_very_large(self):
        """Test that very large epochs value is accepted."""
        config = TrainConfig(dataset="mamografias", epochs=1000000)
        assert config.epochs == 1000000

    def test_batch_size_very_large(self):
        """Test that very large batch_size value is accepted."""
        config = TrainConfig(dataset="mamografias", batch_size=100000)
        assert config.batch_size == 100000

    def test_lr_very_large(self):
        """Test that very large lr value is accepted."""
        config = TrainConfig(dataset="mamografias", lr=1000.0)
        assert config.lr == 1000.0

    def test_img_size_very_large(self):
        """Test that very large img_size value is accepted."""
        config = TrainConfig(dataset="mamografias", img_size=100000)
        assert config.img_size == 100000

    def test_num_workers_very_large(self):
        """Test that very large num_workers value is accepted."""
        config = TrainConfig(dataset="mamografias", num_workers=1000)
        assert config.num_workers == 1000


class TestFloatPrecisionBoundaries:
    """Test floating point precision edge cases."""

    def test_val_frac_machine_epsilon_above_zero(self):
        """Test val_frac with value just above zero."""
        import sys

        epsilon = sys.float_info.epsilon
        config = TrainConfig(dataset="mamografias", val_frac=epsilon)
        assert config.val_frac == epsilon

    def test_val_frac_machine_epsilon_below_one(self):
        """Test val_frac with value just below one."""
        import sys

        epsilon = sys.float_info.epsilon
        config = TrainConfig(dataset="mamografias", val_frac=1.0 - epsilon)
        assert config.val_frac == 1.0 - epsilon

    def test_lr_smallest_normal_float(self):
        """Test lr with smallest normal float value."""
        import sys

        min_normal = sys.float_info.min
        config = TrainConfig(dataset="mamografias", lr=min_normal)
        assert config.lr == min_normal

    def test_weight_decay_smallest_denormal_float(self):
        """Test weight_decay with smallest representable float."""
        import sys

        smallest = sys.float_info.min * sys.float_info.epsilon
        config = TrainConfig(dataset="mamografias", weight_decay=smallest)
        assert config.weight_decay >= 0
