#
# config.py
# mammography-pipelines
#
# Centralizes default hyperparameters used across the mammography training pipelines.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
# Thales Matheus Mendonça Santos - November 2025
#
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from mammography.utils.pydantic_fallback import (
        BaseModel,
        ConfigDict,
        Field,
        field_validator,
        model_validator,
    )

from mammography.utils.class_modes import normalize_classes_mode
from mammography.models.nets import validate_arch_img_size


def _validate_path_exists(value: Path, name: str) -> Path:
    """Validate that a path exists, raising ValueError if not."""
    if not value.exists():
        raise ValueError(f"{name} nao encontrado: {value}")
    return value


def _validate_checkpoint_path(value: Path) -> Path:
    """Validate that a checkpoint path exists and is a regular file."""
    if not value.exists():
        raise ValueError(f"checkpoint nao encontrado: {value}")
    resolved_path = value.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"checkpoint invalido: {value}")
    return value


def _normalize_dir_hint(path: Path) -> Path:
    """Apply simple path hints before validating existence."""
    if path.exists():
        return path
    candidate = Path(str(path).replace("archieve", "archive"))
    if candidate.exists():
        return candidate
    return path


def _validate_arch_img_size_pair(arch: str, img_size: int) -> None:
    """Validate architecture/image-size compatibility for model-based workflows."""
    validate_arch_img_size(arch, img_size)


@dataclass
class HP:
    """Centralized training hyperparameters for the pipelines."""

    IMG_SIZE: int = 512
    WINDOW_P_LOW: float = 0.5
    WINDOW_P_HIGH: float = 99.5
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    LR: float = 1e-4
    BACKBONE_LR: float = 1e-5
    VAL_FRAC: float = 0.20
    SEED: int = 42
    DEVICE: str = "auto"
    UNFREEZE_LAST_BLOCK: bool = True
    TRAIN_BACKBONE: bool = False
    CLASS_WEIGHTS: str = "none"
    SAMPLER_WEIGHTED: bool = False
    WARMUP_EPOCHS: int = 0
    DETERMINISTIC: bool = False
    ALLOW_TF32: bool = True
    PREFETCH_FACTOR: int = 4
    PERSISTENT_WORKERS: bool = True
    PIN_MEMORY: bool = True
    CACHE_MODE: str = "auto"
    LOG_LEVEL: str = "info"
    TRAIN_AUGMENT: bool = True
    LOADER_HEURISTICS: bool = True
    FUSED_OPTIM: bool = False
    TORCH_COMPILE: bool = False
    EARLY_STOP_PATIENCE: int = 0
    EARLY_STOP_MIN_DELTA: float = 0.0
    LR_REDUCE_PATIENCE: int = 0
    LR_REDUCE_FACTOR: float = 0.5
    LR_REDUCE_MIN_LR: float = 1e-7
    LR_REDUCE_COOLDOWN: int = 0
    N_FOLDS: int = 0
    CV_STRATIFIED: bool = True


class BaseConfig(BaseModel):
    """Shared behavior for validated CLI configuration models."""

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_args(cls, args: Any, **overrides: Any) -> "BaseConfig":
        payload = {
            name: getattr(args, name)
            for name in cls.model_fields
            if hasattr(args, name)
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class TrainConfig(BaseConfig):
    dataset: Optional[str] = None
    csv: Optional[Path] = None
    dicom_root: Optional[Path] = None
    include_class_5: bool = False
    outdir: str = "outputs/run"
    cache_mode: str = HP.CACHE_MODE
    cache_dir: Optional[Path] = None
    embeddings_dir: Optional[Path] = None
    mean: Optional[str] = None
    std: Optional[str] = None
    auto_normalize: bool = False
    auto_normalize_samples: int = Field(default=1000, ge=1)
    log_level: str = HP.LOG_LEVEL
    subset: int = Field(default=0, ge=0)

    arch: str = "efficientnet_b0"
    classes: str = "multiclass"
    pretrained: bool = True

    view_specific_training: bool = False
    views_to_train: Optional[list[str]] = None

    epochs: int = Field(default=HP.EPOCHS, ge=1)
    batch_size: int = Field(default=HP.BATCH_SIZE, ge=1)
    lr: float = Field(default=HP.LR, gt=0)
    backbone_lr: float = Field(default=HP.BACKBONE_LR, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    img_size: int = Field(default=HP.IMG_SIZE, ge=1)
    seed: int = Field(default=HP.SEED, ge=0)
    device: str = HP.DEVICE
    val_frac: float = Field(default=HP.VAL_FRAC, gt=0, lt=1)
    split_ensure_all_classes: bool = True
    split_max_tries: int = Field(default=200, ge=1)
    num_workers: int = Field(default=HP.NUM_WORKERS, ge=0)
    prefetch_factor: int = Field(default=HP.PREFETCH_FACTOR, ge=0)
    persistent_workers: bool = HP.PERSISTENT_WORKERS
    loader_heuristics: bool = HP.LOADER_HEURISTICS
    amp: bool = False
    class_weights: str = HP.CLASS_WEIGHTS
    class_weights_alpha: float = Field(default=1.0, gt=0)
    sampler_weighted: bool = HP.SAMPLER_WEIGHTED
    sampler_alpha: float = Field(default=1.0, gt=0)
    train_backbone: bool = HP.TRAIN_BACKBONE
    unfreeze_last_block: bool = HP.UNFREEZE_LAST_BLOCK
    warmup_epochs: int = Field(default=HP.WARMUP_EPOCHS, ge=0)
    deterministic: bool = HP.DETERMINISTIC
    allow_tf32: bool = HP.ALLOW_TF32
    fused_optim: bool = HP.FUSED_OPTIM
    torch_compile: bool = HP.TORCH_COMPILE
    lr_reduce_patience: int = Field(default=HP.LR_REDUCE_PATIENCE, ge=0)
    lr_reduce_factor: float = Field(default=HP.LR_REDUCE_FACTOR, gt=0)
    lr_reduce_min_lr: float = Field(default=HP.LR_REDUCE_MIN_LR, ge=0)
    lr_reduce_cooldown: int = Field(default=HP.LR_REDUCE_COOLDOWN, ge=0)
    scheduler: str = "auto"
    scheduler_min_lr: float = Field(default=HP.LR_REDUCE_MIN_LR, ge=0)
    scheduler_step_size: int = Field(default=5, ge=1)
    scheduler_gamma: float = Field(default=0.5, gt=0)
    profile: bool = False
    profile_dir: str = "outputs/profiler"
    early_stop_patience: int = Field(default=HP.EARLY_STOP_PATIENCE, ge=0)
    early_stop_min_delta: float = Field(default=HP.EARLY_STOP_MIN_DELTA, ge=0)
    augment: bool = HP.TRAIN_AUGMENT
    augment_vertical: bool = False
    augment_color: bool = False
    augment_rotation_deg: float = Field(default=5.0, ge=0)

    n_folds: int = Field(default=HP.N_FOLDS, ge=0)
    cv_fold: Optional[int] = None
    cv_stratified: bool = HP.CV_STRATIFIED

    gradcam: bool = False
    gradcam_limit: int = Field(default=4, ge=1)
    save_val_preds: bool = False
    export_val_embeddings: bool = False

    @field_validator("csv", "dicom_root")
    @classmethod
    def _check_path_exists(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        value = _normalize_dir_hint(value)
        if not value.exists():
            if not value.is_absolute() and not value.anchor:
                return value
            raise ValueError(f"Caminho critico nao encontrado: {value}")
        return value

    @field_validator("embeddings_dir")
    @classmethod
    def _validate_embeddings_dir(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        if not value.exists():
            raise ValueError(f"embeddings_dir nao encontrado: {value}")
        return value

    @field_validator("classes")
    @classmethod
    def _normalize_classes(cls, value: str) -> str:
        return normalize_classes_mode(
            value,
            warn=True,
            source=f"{cls.__name__}.classes",
            allow_unknown=True,
        )

    @field_validator("scheduler", mode="before")
    @classmethod
    def _normalize_scheduler(cls, value: Optional[str]) -> str:
        return "none" if value is None else value

    @field_validator("outdir", "profile_dir")
    @classmethod
    def _validate_output_dir(cls, value: str) -> str:
        """Validate that output directory parent exists and is writable."""
        from pathlib import Path
        import os

        output_path = Path(value)
        parent = output_path.parent

        # If parent directory doesn't exist, try to create it
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                raise ValueError(
                    f"Cannot create parent directory for output: {parent}. Error: {e}"
                )

        # Check if parent is writable
        if not os.access(parent, os.W_OK):
            raise ValueError(
                f"Output directory parent is not writable: {parent}"
            )

        return value

    @model_validator(mode="after")
    def _validate_source(self) -> "TrainConfig":
        if not self.csv and not self.dataset:
            raise ValueError("Informe --csv ou --dataset para treinar.")
        return self

    @model_validator(mode="after")
    def _validate_augmentation_consistency(self) -> "TrainConfig":
        """Validate that augmentation parameters are only set when augment=True."""
        if not self.augment:
            if self.augment_vertical or self.augment_color or self.augment_rotation_deg != 5.0:
                import warnings
                warnings.warn(
                    "Augmentation is disabled (augment=False) but augmentation parameters are set. "
                    "These parameters will be ignored.",
                    UserWarning
                )
        return self

    @model_validator(mode="after")
    def _validate_gradcam_consistency(self) -> "TrainConfig":
        """Validate that gradcam_limit is only meaningful when gradcam=True."""
        if not self.gradcam and self.gradcam_limit != 4:
            import warnings
            warnings.warn(
                "Grad-CAM is disabled (gradcam=False) but gradcam_limit is set. "
                "This parameter will be ignored.",
                UserWarning
            )
        return self

    @model_validator(mode="after")
    def _validate_cv_consistency(self) -> "TrainConfig":
        """Validate cross-validation parameters consistency."""
        if self.cv_fold is not None:
            if self.n_folds == 0:
                raise ValueError(
                    "cv_fold specified but n_folds=0. Set n_folds to a positive value to use cross-validation."
                )
            if self.cv_fold < 0 or self.cv_fold >= self.n_folds:
                raise ValueError(
                    f"cv_fold must be in range [0, {self.n_folds}). Got: {self.cv_fold}"
                )
        return self

    @model_validator(mode="after")
    def _validate_model_image_size(self) -> "TrainConfig":
        _validate_arch_img_size_pair(self.arch, self.img_size)
        return self


class ExtractConfig(BaseConfig):
    dataset: Optional[str] = None
    csv: Optional[Path] = None
    dicom_root: Optional[Path] = None
    outdir: str = "outputs/features"
    seed: int = Field(default=HP.SEED, ge=0)
    device: str = HP.DEVICE
    deterministic: bool = HP.DETERMINISTIC
    allow_tf32: bool = HP.ALLOW_TF32
    arch: str = "resnet50"
    pretrained: bool = True
    classes: str = "multiclass"
    img_size: int = Field(default=HP.IMG_SIZE, ge=1)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=HP.NUM_WORKERS, ge=0)
    prefetch_factor: int = Field(default=HP.PREFETCH_FACTOR, ge=0)
    persistent_workers: bool = HP.PERSISTENT_WORKERS
    loader_heuristics: bool = HP.LOADER_HEURISTICS
    cache_mode: str = HP.CACHE_MODE
    include_class_5: bool = False
    log_level: str = HP.LOG_LEVEL
    amp: bool = False
    mean: Optional[str] = None
    std: Optional[str] = None
    layer_name: str = "avgpool"
    save_csv: bool = False
    run_reduction: bool = False
    run_clustering: bool = False
    pca: bool = False
    pca_svd_solver: str = "auto"
    tsne: bool = False
    umap: bool = False
    cluster_auto: bool = False
    cluster_k: int = Field(default=0, ge=0)
    n_clusters: int = Field(default=0, ge=0)
    sample_grid: int = Field(default=16, ge=0)

    @field_validator("csv")
    @classmethod
    def _validate_csv(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        if not value.exists():
            raise ValueError(f"csv_path nao encontrado: {value}")
        return value

    @field_validator("dicom_root")
    @classmethod
    def _validate_dicom_root(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        value = _normalize_dir_hint(value)
        if not value.exists():
            raise ValueError(f"dicom_root nao encontrado: {value}")
        return value

    @field_validator("classes")
    @classmethod
    def _normalize_classes(cls, value: str) -> str:
        return normalize_classes_mode(
            value,
            warn=True,
            source=f"{cls.__name__}.classes",
            allow_unknown=True,
        )

    @model_validator(mode="after")
    def _validate_source(self) -> "ExtractConfig":
        if not self.csv and not self.dataset:
            raise ValueError("Informe --csv ou --dataset para extrair embeddings.")
        return self

    @model_validator(mode="after")
    def _validate_model_image_size(self) -> "ExtractConfig":
        _validate_arch_img_size_pair(self.arch, self.img_size)
        return self


class InferenceConfig(BaseConfig):
    checkpoint: Path
    input: Optional[Path] = None
    csv: Optional[Path] = None
    dicom_root: Optional[Path] = None
    arch: str = "resnet50"
    architecture: Optional[str] = None
    classes: str = "multiclass"
    img_size: int = Field(default=HP.IMG_SIZE, ge=1)
    batch_size: int = Field(default=16, ge=1)
    device: str = HP.DEVICE
    output: Optional[str] = None
    amp: bool = False
    mean: Optional[str] = None
    std: Optional[str] = None

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: Path) -> Path:
        return _validate_checkpoint_path(value)

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        return _validate_path_exists(value, "input")

    @field_validator("csv")
    @classmethod
    def _validate_csv(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        return _validate_path_exists(value, "csv")

    @field_validator("dicom_root")
    @classmethod
    def _validate_dicom_root(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        return _validate_path_exists(value, "dicom_root")

    @field_validator("classes")
    @classmethod
    def _normalize_classes(cls, value: str) -> str:
        return normalize_classes_mode(
            value,
            warn=True,
            source=f"{cls.__name__}.classes",
            allow_unknown=True,
        )

    @model_validator(mode="after")
    def _validate_model_image_size(self) -> "InferenceConfig":
        if self.architecture:
            self.arch = self.architecture
        _validate_arch_img_size_pair(self.arch, self.img_size)
        if not self.input and not self.csv and not self.dicom_root:
            raise ValueError("Informe --input, --csv ou --dicom-root para inferencia.")
        return self


class BatchInferenceConfig(BaseConfig):
    """Configuration for batch inference with progress tracking and resumable processing."""

    checkpoint: Path
    input: Path
    arch: str = "resnet50"
    classes: str = "multiclass"
    img_size: int = Field(default=HP.IMG_SIZE, ge=1)
    batch_size: int = Field(default=16, ge=1)
    device: str = HP.DEVICE
    output: Optional[str] = None
    output_format: str = "csv"
    resume: bool = False
    checkpoint_file: Optional[Path] = None
    checkpoint_interval: int = Field(default=100, ge=1)
    num_workers: int = Field(default=HP.NUM_WORKERS, ge=0)
    prefetch_factor: int = Field(default=HP.PREFETCH_FACTOR, ge=0)
    persistent_workers: bool = HP.PERSISTENT_WORKERS
    pin_memory: bool = HP.PIN_MEMORY
    amp: bool = False
    mean: Optional[str] = None
    std: Optional[str] = None

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: Path) -> Path:
        return _validate_checkpoint_path(value)

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: Path) -> Path:
        return _validate_path_exists(value, "input")

    @field_validator("checkpoint_file")
    @classmethod
    def _validate_checkpoint_file(cls, value: Optional[Path]) -> Optional[Path]:
        """
        Validate the checkpoint_file path by ensuring its parent directory exists.
        
        If `value` is None, returns None. If `value` is provided, verifies that its parent directory exists and returns `value`; raises ValueError if the parent directory does not exist.
        
        Parameters:
            value (Optional[Path]): Path to the checkpoint file or None.
        
        Returns:
            Optional[Path]: The original `value` if valid, or None.
        """
        if value is None:
            return value
        # For checkpoint_file, we only check if parent directory exists
        # (the file itself may not exist yet if we're starting fresh)
        parent = value.parent
        if not parent.exists():
            raise ValueError(f"checkpoint_file parent directory nao encontrado: {parent}")
        return value

    @field_validator("classes")
    @classmethod
    def _normalize_classes(cls, value: str) -> str:
        return normalize_classes_mode(
            value,
            warn=True,
            source=f"{cls.__name__}.classes",
            allow_unknown=True,
        )

    @field_validator("output_format")
    @classmethod
    def _validate_output_format(cls, value: str) -> str:
        """
        Ensure the output_format string is one of the supported formats.
        
        Supported values: "csv", "json", "jsonl".
        
        Parameters:
            value (str): Requested output format.
        
        Returns:
            str: The validated output format string.
        
        Raises:
            ValueError: If `value` is not one of the supported formats.
        """
        allowed = {"csv", "json", "jsonl"}
        if value not in allowed:
            raise ValueError(
                f"output_format deve ser um de: {sorted(allowed)}. Recebido: {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _validate_model_image_size(self) -> "BatchInferenceConfig":
        _validate_arch_img_size_pair(self.arch, self.img_size)
        return self


class PreprocessConfig(BaseConfig):
    """Configuration for the dataset preprocessing CLI command."""

    input: Path
    output: Path
    normalize: str = "per-image"
    img_size: int = Field(default=HP.IMG_SIZE, ge=1)
    resize: bool = True
    crop: bool = False
    format: str = "png"
    preview: bool = False
    preview_n: int = Field(default=8, ge=1)
    report: bool = True
    border_removal: bool = False
    log_level: str = HP.LOG_LEVEL

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"input nao encontrado: {value}")
        return value

    @field_validator("output")
    @classmethod
    def _validate_output(cls, value: Path) -> Path:
        """Attempt to create the output parent directory if it does not exist."""
        import os

        parent = value.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                raise ValueError(
                    f"Cannot create parent directory for output: {parent}. Error: {e}"
                )
        if not os.access(parent, os.W_OK):
            raise ValueError(f"Output directory parent is not writable: {parent}")
        return value

    @field_validator("normalize")
    @classmethod
    def _validate_normalize(cls, value: str) -> str:
        allowed = {"per-image", "per-dataset", "none"}
        if value not in allowed:
            raise ValueError(
                f"normalize deve ser um de: {sorted(allowed)}. Recebido: {value!r}"
            )
        return value

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        allowed = {"png", "jpg", "keep"}
        if value not in allowed:
            raise ValueError(
                f"format deve ser um de: {sorted(allowed)}. Recebido: {value!r}"
            )
        return value
