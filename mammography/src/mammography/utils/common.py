#
# common.py
# mammography-pipelines-py
#
# Shared utilities for seeding, device/runtime setup, logging configuration, and safe path handling.
#
# Thales Matheus Mendonça Santos - November 2025
#
import os
import logging
import random
import numpy as np
import torch
try:
    from tqdm.contrib.logging import TqdmLoggingHandler
except ImportError:
    TqdmLoggingHandler = logging.StreamHandler

def seed_everything(seed: int = 42, deterministic: bool = False):
    """Make training more reproducible by seeding Python, NumPy, Torch, and CUDA/MPS."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    if deterministic:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def resolve_device(device_choice: str) -> torch.device:
    """Pick the best available device (CUDA > MPS > CPU) unless an explicit choice is provided."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_choice == "mps":
        return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if device_choice == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def configure_runtime(device: torch.device, deterministic: bool, allow_tf32: bool) -> None:
    """Tweak backend flags to balance performance and determinism for the chosen accelerator."""
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if not deterministic:
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
    elif device.type == "mps":
        if not deterministic:
            try:
                torch.set_float32_matmul_precision("medium")
            except AttributeError:
                pass


def setup_logging(outdir: str, level: str, name: str = "mammography") -> logging.Logger:
    """Configure a dual logger that streams to console and writes a rotating run.log."""
    log_dir = outdir
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_level = getattr(logging, level.upper(), logging.INFO)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def increment_path(path: str) -> str:
    """Return a free path by appending _1/_2/... if the desired folder already exists."""
    base = path.rstrip("/\\ ")
    if not os.path.exists(base):
        return base
    i = 1
    while True:
        cand = f"{base}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1
