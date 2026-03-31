#
# resource_manager.py
# mammography-pipelines
#
# Resource detection and management for hyperparameter tuning with resource-aware search space adjustment.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Resource management utilities for AutoML with GPU memory detection and search space constraints."""

import logging
from typing import Optional, Dict, Any, List
import torch

logger = logging.getLogger(__name__)


class ResourceManager:
    """Detects available compute resources and adjusts search space based on hardware constraints.

    This manager identifies GPU memory, CPU cores, and other resources to ensure hyperparameter
    search does not suggest configurations that would exceed available hardware limits.

    Attributes:
        device: The detected torch device (cuda, mps, or cpu)
        gpu_memory_gb: Total GPU memory in GB (0 if CPU-only)
        cpu_count: Number of available CPU cores
        has_gpu: Whether GPU acceleration is available
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize resource manager and detect available hardware.

        Args:
            device: Optional device override ("cuda", "mps", "cpu"). If None, auto-detects.
        """
        self.device = self._detect_device(device)
        self.gpu_memory_gb = self._detect_gpu_memory()
        self.cpu_count = self._detect_cpu_count()
        self.has_gpu = self.device.type in ("cuda", "mps")

        logger.info("ResourceManager initialized:")
        logger.info("  Device: %s", self.device)
        logger.info("  GPU Memory: %.1f GB", self.gpu_memory_gb)
        logger.info("  CPU Cores: %d", self.cpu_count)

    def _detect_device(self, device_override: Optional[str]) -> torch.device:
        """Detect or resolve the compute device.

        Args:
            device_override: Optional device string to override auto-detection

        Returns:
            torch.device: The selected device
        """
        if device_override is not None:
            # Handle torch.device object
            if isinstance(device_override, torch.device):
                return device_override
            # Handle string
            device_str = device_override.lower()
            if device_str not in ("cuda", "mps", "cpu"):
                logger.warning("Invalid device '%s', falling back to auto-detection", device_override)
                device_str = None
            else:
                return torch.device(device_str)

        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _detect_gpu_memory(self) -> float:
        """Detect total GPU memory in GB.

        Returns:
            float: GPU memory in GB, or 0.0 if no GPU available
        """
        if self.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / (1024 ** 3)
                logger.debug("CUDA GPU memory detected: %.2f GB", memory_gb)
                return memory_gb
            except Exception as exc:
                logger.warning("Failed to detect CUDA memory: %s", exc)
                return 0.0
        elif self.device.type == "mps":
            # MPS does not expose memory stats via PyTorch API
            # Use conservative estimate for Apple Silicon GPUs (8-16 GB typical)
            logger.debug("MPS device detected, estimating 8 GB unified memory")
            return 8.0
        else:
            return 0.0

    def _detect_cpu_count(self) -> int:
        """Detect number of available CPU cores.

        Returns:
            int: Number of CPU cores, defaulting to 1 if detection fails
        """
        try:
            import os
            count = os.cpu_count() or 1
            return max(1, count)
        except Exception as exc:
            logger.warning("Failed to detect CPU count: %s", exc)
            return 1

    def get_max_batch_size(self) -> int:
        """Suggest maximum batch size based on available GPU memory.

        Returns:
            int: Recommended maximum batch size
        """
        if not self.has_gpu:
            return 8  # CPU default

        # Conservative estimates based on typical model sizes
        if self.gpu_memory_gb >= 24:
            return 64
        elif self.gpu_memory_gb >= 16:
            return 48
        elif self.gpu_memory_gb >= 12:
            return 32
        elif self.gpu_memory_gb >= 8:
            return 24
        elif self.gpu_memory_gb >= 6:
            return 16
        else:
            return 8

    def get_max_num_workers(self) -> int:
        """Suggest maximum number of DataLoader workers.

        Returns:
            int: Recommended maximum workers for data loading
        """
        # Cap at 8 workers to avoid excessive overhead
        return min(self.cpu_count, 8)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Return a dictionary summarizing detected resources.

        Returns:
            Dict containing device info, memory, CPU count, and suggested limits
        """
        return {
            "device": str(self.device),
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_count": self.cpu_count,
            "has_gpu": self.has_gpu,
            "max_batch_size": self.get_max_batch_size(),
            "max_num_workers": self.get_max_num_workers(),
        }

    def filter_architectures(self, architectures: List[str]) -> List[str]:
        """Filter architecture list based on available GPU memory.

        Large models like ViT-L require more memory than smaller models.
        This method filters out architectures that would exceed available GPU memory
        based on conservative training memory estimates.

        Memory requirements (approximate, for training with typical batch sizes):
        - efficientnet_b0: 4-6 GB
        - resnet50: 6-8 GB
        - vit_b_32: 8-10 GB
        - vit_b_16: 10-12 GB
        - vit_l_16: 16-24 GB
        - deit variants: Similar to corresponding ViT models

        Args:
            architectures: List of architecture names to filter

        Returns:
            List of architectures that fit within resource constraints
        """
        if not architectures:
            return []

        # Define minimum GPU memory requirements for each architecture (in GB)
        # These are conservative estimates for training with typical batch sizes
        ARCH_MEMORY_REQUIREMENTS = {
            "efficientnet_b0": 4.0,
            "resnet50": 6.0,
            "vit_b_32": 8.0,
            "vit_b_16": 10.0,
            "vit_l_16": 16.0,
            "vit_l_32": 14.0,
            # DeiT models (timm library)
            "deit_tiny_patch16_224": 6.0,
            "deit_small_patch16_224": 8.0,
            "deit_base_patch16_224": 10.0,
            "deit_base_patch16_384": 14.0,
        }

        # If no GPU, only allow most lightweight models
        if not self.has_gpu:
            cpu_compatible = ["efficientnet_b0", "resnet50"]
            filtered = [arch for arch in architectures if arch in cpu_compatible]
            if len(filtered) < len(architectures):
                removed = set(architectures) - set(filtered)
                logger.warning(
                    "CPU-only mode: filtering out GPU-only architectures %s",
                    sorted(removed)
                )
            return filtered

        # GPU available: filter based on memory
        filtered = []
        removed = []
        for arch in architectures:
            required_memory = ARCH_MEMORY_REQUIREMENTS.get(arch, 8.0)  # Default 8 GB if unknown
            # Use conservative threshold: allow if 80% of memory is sufficient
            # This accounts for gradients, optimizer states, and activations
            memory_threshold = required_memory * 1.25  # 25% safety margin
            if self.gpu_memory_gb >= memory_threshold:
                filtered.append(arch)
            else:
                removed.append(arch)
                logger.debug(
                    "Filtering out %s: requires %.1f GB, available %.1f GB",
                    arch, memory_threshold, self.gpu_memory_gb
                )

        if removed:
            logger.info(
                "Filtered %d/%d architectures due to %.1f GB GPU memory limit: %s",
                len(removed), len(architectures), self.gpu_memory_gb, sorted(removed)
            )

        # Ensure at least one architecture remains
        if not filtered and architectures:
            logger.warning(
                "All architectures filtered out! Keeping smallest: efficientnet_b0"
            )
            # Return the smallest architecture as fallback
            if "efficientnet_b0" in architectures:
                filtered = ["efficientnet_b0"]
            else:
                # If efficientnet_b0 not in list, return the first one as last resort
                filtered = [architectures[0]]
                logger.warning("efficientnet_b0 not available, using fallback: %s", filtered[0])

        return filtered

    def validate_config(self, batch_size: int, num_workers: int) -> bool:
        """Validate whether a given configuration is feasible given available resources.

        Args:
            batch_size: Proposed batch size
            num_workers: Proposed number of DataLoader workers

        Returns:
            bool: True if configuration is feasible, False otherwise
        """
        max_batch = self.get_max_batch_size()
        max_workers = self.get_max_num_workers()

        if batch_size > max_batch:
            logger.warning(
                "Batch size %d exceeds recommended max %d for %.1f GB GPU",
                batch_size, max_batch, self.gpu_memory_gb
            )
            return False

        if num_workers > max_workers:
            logger.warning(
                "num_workers %d exceeds available CPU cores %d",
                num_workers, self.cpu_count
            )
            return False

        return True
