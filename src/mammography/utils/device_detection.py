"""Device detection and configuration utilities for ResNet50_Test."""

import logging
import platform
from typing import Any, Dict

import torch

RESEARCH_DISCLAIMER = (
    """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
)

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Detect and configure the most suitable PyTorch device."""

    def __init__(self) -> None:
        self.system_info = self._get_system_info()
        self.available_devices = self._detect_available_devices()
        self.best_device = self._select_best_device()

    def _get_system_info(self) -> Dict[str, Any]:
        """Gather basic system information for logging and debugging."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }

    def _detect_available_devices(self) -> Dict[str, bool]:
        """Check which accelerators are available (CUDA, MPS, CPU)."""
        devices = {
            "cuda": False,
            "mps": False,
            "cpu": True,  # CPU is always available
        }

        if torch.cuda.is_available():
            devices["cuda"] = True
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("CUDA VRAM: %.1f GB", vram)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices["mps"] = True
            logger.info("MPS (Apple Silicon GPU) available")

        return devices

    def _select_best_device(self) -> str:
        """Return the highest-priority device available."""
        if self.available_devices["cuda"]:
            return "cuda"
        if self.available_devices["mps"]:
            return "mps"
        return "cpu"

    def get_device(self) -> torch.device:
        """Return the PyTorch device object representing the preferred backend."""
        return torch.device(self.best_device)

    def get_device_config(self) -> Dict[str, Any]:
        """Return recommended configuration defaults for the active device."""
        configs = {
            "cuda": {
                "batch_size": 16,
                "mixed_precision": True,
                "gpu_memory_limit": 16,
                "num_workers": 4,
                "pin_memory": True,
            },
            "mps": {
                "batch_size": 8,
                "mixed_precision": False,  # AMP is not supported on MPS
                "gpu_memory_limit": 8,
                "num_workers": 4,
                "pin_memory": False,
            },
            "cpu": {
                "batch_size": 4,
                "mixed_precision": False,
                "gpu_memory_limit": 0,
                "num_workers": 8,
                "pin_memory": False,
            },
        }
        return configs.get(self.best_device, configs["cpu"])

    def optimize_for_device(self) -> None:
        """Apply backend-specific optimisation flags."""
        if self.best_device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("CUDA optimisations enabled")
        elif self.best_device == "mps":
            logger.info("MPS optimisations enabled")
        elif self.best_device == "cpu":
            torch.set_num_threads(8)
            logger.info("CPU optimisations enabled")

    def get_memory_info(self) -> Dict[str, Any]:
        """Return memory statistics for the current device."""
        memory_info: Dict[str, Any] = {
            "device": self.best_device,
            "total_memory": 0,
            "allocated_memory": 0,
            "cached_memory": 0,
        }

        if self.best_device == "cuda":
            props = torch.cuda.get_device_properties(0)
            memory_info["total_memory"] = props.total_memory
            memory_info["allocated_memory"] = torch.cuda.memory_allocated()
            memory_info["cached_memory"] = torch.cuda.memory_reserved()
        elif self.best_device == "mps":
            # torch.mps currently exposes limited memory diagnostics
            memory_info.update({
                "total_memory": 0,
                "allocated_memory": 0,
                "cached_memory": 0,
            })

        return memory_info

    def print_device_status(self) -> None:
        """Convenience helper for CLI commands."""
        device = self.get_device()
        config = self.get_device_config()

        print("Selected device:", device)
        print("Suggested configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


def get_optimal_device() -> torch.device:
    """Return the preferred device for the running environment."""
    detector = DeviceDetector()
    return detector.get_device()


def get_device_config() -> Dict[str, Any]:
    """Return recommended configuration parameters for the preferred device."""
    detector = DeviceDetector()
    return detector.get_device_config()


def print_device_status() -> None:
    """Print a summary of device selection and configuration."""
    detector = DeviceDetector()
    detector.print_device_status()
