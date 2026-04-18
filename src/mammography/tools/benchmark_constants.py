"""Shared constants for official benchmark report generation."""

DATASET_ORDER = ("archive", "mamografias", "patches_completo")
TASK_ORDER = ("multiclass", "binary")
ARCH_ORDER = ("efficientnet_b0", "resnet50", "vit_b_16")

EXPECTED_SPLITS = {
    "archive": "patient",
    "mamografias": "random",
    "patches_completo": "random",
}

COMMON_CONFIG = {
    "seed": 42,
    "deterministic": True,
    "allow_tf32": True,
    "amp": True,
    "pretrained": True,
    "train_backbone": True,
    "unfreeze_last_block": True,
    "augment": True,
    "class_weights": "auto",
    "sampler_weighted": True,
    "test_frac": 0.1,
    "tracker": "local",
    "view_specific_training": False,
}

ARCH_CONFIG = {
    "efficientnet_b0": {
        "img_size": 512,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
    },
    "resnet50": {
        "img_size": 512,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "backbone_lr": 1e-5,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
    },
    "vit_b_16": {
        "img_size": 224,
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "warmup_epochs": 3,
        "early_stop_patience": 10,
    },
}

MASTER_COLUMNS = [
    "dataset",
    "task",
    "split_mode",
    "arch",
    "seed",
    "img_size",
    "batch_size",
    "epochs",
    "accuracy",
    "kappa",
    "macro_f1",
    "auc",
    "best_epoch",
    "run_path",
    "status",
]

ARTICLE_COLUMNS = [
    "dataset",
    "task",
    "split",
    "modelo",
    "accuracy",
    "kappa",
    "macro-F1",
    "AUC",
]
