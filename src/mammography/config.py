#
# config.py
# mammography-pipelines
#
# Centralizes default hyperparameters used across the mammography training pipelines.
#
# Thales Matheus Mendonça Santos - November 2025
#
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_dir_hint(path: Path) -> Path:
    """Apply simple path hints before validating existence."""
    if path.exists():
        return path
    candidate = Path(str(path).replace("archieve", "archive"))
    if candidate.exists():
        return candidate
    return path


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
    log_level: str = HP.LOG_LEVEL
    subset: int = Field(default=0, ge=0)

    arch: str = "efficientnet_b0"
    classes: str = "density"
    pretrained: bool = True

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

    gradcam: bool = False
    gradcam_limit: int = Field(default=4, ge=1)
    save_val_preds: bool = False
    export_val_embeddings: bool = False

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

    @field_validator("embeddings_dir")
    @classmethod
    def _validate_embeddings_dir(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        if not value.exists():
            raise ValueError(f"embeddings_dir nao encontrado: {value}")
        return value

    @model_validator(mode="after")
    def _validate_source(self) -> "TrainConfig":
        if not self.csv and not self.dataset:
            raise ValueError("Informe --csv ou --dataset para treinar.")
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

    @model_validator(mode="after")
    def _validate_source(self) -> "ExtractConfig":
        if not self.csv and not self.dataset:
            raise ValueError("Informe --csv ou --dataset para extrair embeddings.")
        return self


class InferenceConfig(BaseConfig):
    checkpoint: Path
    input: Path
    arch: str = "resnet50"
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
        if not value.exists():
            raise ValueError(f"checkpoint nao encontrado: {value}")
        if not value.is_file():
            raise ValueError(f"checkpoint invalido: {value}")
        return value

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"input nao encontrado: {value}")
        return value
