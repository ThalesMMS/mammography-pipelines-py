from dataclasses import dataclass

@dataclass
class HP:
    """Hiperparâmetros centrais deste pipeline."""
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
