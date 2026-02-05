"""Smart defaults calculator based on hardware detection and dataset characteristics.

Provides intelligent default values for training and inference configurations
by analyzing available hardware (GPU/CPU) and dataset properties.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from .device_detection import DeviceDetector

if TYPE_CHECKING:
    from ..data.format_detection import DatasetFormat

logger = logging.getLogger(__name__)


class SmartDefaults:
    """Calculate intelligent default values based on hardware and dataset properties.

    This class wraps DeviceDetector and provides methods for retrieving
    hardware-aware configuration defaults for training, inference, and data loading.
    Optionally accepts DatasetFormat for dataset-specific recommendations.

    Attributes:
        device_detector: DeviceDetector instance for hardware analysis
        device_type: Detected device type (cuda, mps, cpu)
        device_config: Base device configuration from DeviceDetector
        system_memory_gb: Available system RAM in GB
        dataset_format: Optional DatasetFormat for dataset-aware defaults

    Examples:
        >>> defaults = SmartDefaults()
        >>> batch_size = defaults.get_batch_size()
        >>> num_workers = defaults.get_num_workers()
        >>> cache_mode = defaults.get_cache_mode()

        >>> # With dataset format detection
        >>> from mammography.data.format_detection import detect_dataset_format
        >>> dataset_info = detect_dataset_format("/path/to/dataset")
        >>> defaults = SmartDefaults(dataset_format=dataset_info)
        >>> defaults.get_training_defaults()  # Now dataset-aware
    """

    def __init__(
        self,
        device_detector: Optional[DeviceDetector] = None,
        dataset_format: Optional[DatasetFormat] = None,
    ) -> None:
        """Initialize SmartDefaults with hardware and optional dataset detection.

        Args:
            device_detector: Optional DeviceDetector instance. If None, creates new one.
            dataset_format: Optional DatasetFormat from format detection for dataset-aware defaults.
        """
        self.device_detector = device_detector or DeviceDetector()
        self.device_type = self.device_detector.best_device
        self.device_config = self.device_detector.get_device_config()
        self.system_memory_gb = self._get_system_memory_gb()
        self.dataset_format = dataset_format

        logger.info(
            "SmartDefaults initialized: device=%s, memory=%.1f GB, dataset=%s",
            self.device_type,
            self.system_memory_gb,
            dataset_format.dataset_type if dataset_format else "none",
        )

    def _get_system_memory_gb(self) -> float:
        """Get available system RAM in GB.

        Returns:
            Available system memory in gigabytes (estimated conservatively)
        """
        # Conservative estimate based on typical systems
        # Can be enhanced if psutil is added as dependency
        try:
            # Try to get info from /proc/meminfo on Linux
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)  # Convert KB to GB
        except Exception as e:
            logger.debug("Could not read /proc/meminfo: %s", e)

        # Default conservative estimate
        return 8.0

    def get_batch_size(
        self, task: str = "train", image_size: int = 224, dataset_size: Optional[int] = None
    ) -> int:
        """Get recommended batch size based on hardware and task.

        Args:
            task: Task type - "train", "inference", or "embed"
            image_size: Input image size (default: 224)
            dataset_size: Optional dataset size for small dataset adjustment

        Returns:
            Recommended batch size (power of 2)

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_batch_size(task="train")
            16  # On CUDA
            >>> defaults.get_batch_size(task="inference")
            32  # Larger for inference
        """
        # Start with device-specific baseline
        base_batch_size = self.device_config["batch_size"]

        # Adjust for task type
        if task == "inference":
            # Inference can use larger batches (no gradients)
            base_batch_size = int(base_batch_size * 2)
        elif task == "embed":
            # Embedding extraction can also use larger batches
            base_batch_size = int(base_batch_size * 1.5)

        # Adjust for image size (larger images need smaller batches)
        if image_size > 224:
            scale_factor = (224 / image_size) ** 2
            base_batch_size = max(1, int(base_batch_size * scale_factor))

        # Adjust for small datasets (avoid overly large batches)
        if dataset_size is not None and dataset_size < 1000:
            base_batch_size = min(base_batch_size, max(4, dataset_size // 10))

        # Ensure power of 2 for optimal GPU performance
        batch_size = self._nearest_power_of_2(base_batch_size)

        # Apply device-specific limits
        if self.device_type == "cpu":
            batch_size = min(batch_size, 8)
        elif self.device_type == "mps":
            batch_size = min(batch_size, 16)
        elif self.device_type == "cuda":
            # Check VRAM availability
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 8:
                    batch_size = min(batch_size, 8)
                elif vram_gb < 16:
                    batch_size = min(batch_size, 16)
            except Exception:
                pass

        return max(1, batch_size)

    def get_num_workers(self, task: str = "train") -> int:
        """Get recommended number of data loader workers.

        Args:
            task: Task type - "train", "inference", or "embed"

        Returns:
            Recommended number of workers

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_num_workers(task="train")
            4  # On most systems
        """
        base_workers = self.device_config.get("num_workers", 4)

        # CPU can benefit from more workers
        if self.device_type == "cpu":
            cpu_count = os.cpu_count() or 4
            base_workers = min(cpu_count - 1, 8)  # Leave 1 CPU free

        # Adjust based on task
        if task == "inference":
            # Inference is less I/O bound
            base_workers = max(2, base_workers // 2)

        return max(0, base_workers)

    def get_cache_mode(self, dataset_size: Optional[int] = None) -> str:
        """Get recommended cache mode based on hardware and dataset size.

        Args:
            dataset_size: Number of samples in dataset

        Returns:
            Recommended cache mode: "auto", "memory", "disk", "tensor-disk", or "none"

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_cache_mode(dataset_size=500)
            'memory'  # Small dataset
            >>> defaults.get_cache_mode(dataset_size=10000)
            'disk'  # Large dataset
        """
        # Use dataset format info if available
        if self.dataset_format and dataset_size is None:
            dataset_size = self.dataset_format.image_count

        if dataset_size is None:
            return "auto"

        # Estimate memory requirements
        # DICOM files are larger (5-10MB) vs PNG/JPG (1-2MB)
        is_dicom = self.dataset_format and self.dataset_format.image_format == "dicom"
        memory_per_image_gb = 0.008 if is_dicom else 0.0015
        estimated_memory_gb = dataset_size * memory_per_image_gb

        # Small datasets can fit in memory
        if dataset_size < 1000 and estimated_memory_gb < self.system_memory_gb * 0.3:
            return "memory"

        # Medium datasets use disk cache
        if dataset_size < 10000:
            return "disk"

        # Large datasets use tensor-disk for efficiency
        return "tensor-disk"

    def get_epochs(
        self, dataset_size: Optional[int] = None, task: str = "train"
    ) -> int:
        """Get recommended number of training epochs.

        Args:
            dataset_size: Number of samples in dataset
            task: Task type (currently only "train" uses this)

        Returns:
            Recommended number of epochs

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_epochs(dataset_size=500)
            50  # Small dataset needs more epochs
            >>> defaults.get_epochs(dataset_size=10000)
            20  # Large dataset needs fewer epochs
        """
        if dataset_size is None:
            return 30  # Conservative default

        # Small datasets benefit from more epochs
        if dataset_size < 500:
            return 50
        elif dataset_size < 2000:
            return 30
        elif dataset_size < 5000:
            return 20
        else:
            return 15

    def get_mixed_precision(self) -> bool:
        """Get whether mixed precision training is recommended.

        Returns:
            True if mixed precision is recommended for the current device

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_mixed_precision()
            True  # On CUDA with AMP support
        """
        return self.device_config.get("mixed_precision", False)

    def get_pin_memory(self) -> bool:
        """Get whether to pin memory in data loaders.

        Returns:
            True if memory pinning is recommended

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_pin_memory()
            True  # On CUDA
        """
        return self.device_config.get("pin_memory", False)

    def get_learning_rate(self, batch_size: Optional[int] = None) -> float:
        """Get recommended learning rate based on batch size.

        Uses linear scaling rule: lr scales with batch size.

        Args:
            batch_size: Batch size (if None, uses recommended batch size)

        Returns:
            Recommended learning rate

        Examples:
            >>> defaults = SmartDefaults()
            >>> defaults.get_learning_rate(batch_size=16)
            0.001  # Base learning rate
        """
        base_lr = 1e-3  # Standard base learning rate

        if batch_size is None:
            batch_size = self.get_batch_size()

        # Linear scaling rule: scale lr with batch size
        # Reference batch size is 16
        reference_batch = 16
        scaled_lr = base_lr * (batch_size / reference_batch)

        # Clamp to reasonable range
        return max(1e-5, min(1e-2, scaled_lr))

    def get_training_defaults(
        self,
        dataset_size: Optional[int] = None,
        image_size: int = 224,
    ) -> Dict[str, Any]:
        """Get comprehensive training configuration defaults.

        Args:
            dataset_size: Number of samples in dataset (uses dataset_format if available)
            image_size: Input image size

        Returns:
            Dictionary with all recommended training defaults

        Examples:
            >>> defaults = SmartDefaults()
            >>> config = defaults.get_training_defaults(dataset_size=1000)
            >>> config['batch_size']
            16
            >>> config['epochs']
            30
        """
        # Use dataset format info if available and dataset_size not explicitly provided
        if dataset_size is None and self.dataset_format:
            dataset_size = self.dataset_format.image_count

        batch_size = self.get_batch_size(
            task="train", image_size=image_size, dataset_size=dataset_size
        )

        return {
            "device": self.device_type,
            "batch_size": batch_size,
            "num_workers": self.get_num_workers(task="train"),
            "epochs": self.get_epochs(dataset_size=dataset_size),
            "lr": self.get_learning_rate(batch_size=batch_size),
            "mixed_precision": self.get_mixed_precision(),
            "pin_memory": self.get_pin_memory(),
            "cache_mode": self.get_cache_mode(dataset_size=dataset_size),
        }

    def get_inference_defaults(
        self,
        dataset_size: Optional[int] = None,
        image_size: int = 224,
    ) -> Dict[str, Any]:
        """Get comprehensive inference configuration defaults.

        Args:
            dataset_size: Number of samples to process
            image_size: Input image size

        Returns:
            Dictionary with all recommended inference defaults

        Examples:
            >>> defaults = SmartDefaults()
            >>> config = defaults.get_inference_defaults()
            >>> config['batch_size']
            32
        """
        return {
            "device": self.device_type,
            "batch_size": self.get_batch_size(
                task="inference", image_size=image_size, dataset_size=dataset_size
            ),
            "num_workers": self.get_num_workers(task="inference"),
            "mixed_precision": self.get_mixed_precision(),
            "pin_memory": self.get_pin_memory(),
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset-specific information and recommendations.

        Returns:
            Dictionary with dataset format info, warnings, and suggestions.
            Empty dict if no dataset_format is available.

        Examples:
            >>> from mammography.data.format_detection import detect_dataset_format
            >>> dataset_info = detect_dataset_format("/path/to/dataset")
            >>> defaults = SmartDefaults(dataset_format=dataset_info)
            >>> info = defaults.get_dataset_info()
            >>> info['dataset_type']
            'archive'
            >>> info['warnings']
            []
        """
        if not self.dataset_format:
            return {}

        return {
            "dataset_type": self.dataset_format.dataset_type,
            "image_format": self.dataset_format.image_format,
            "image_count": self.dataset_format.image_count,
            "format_counts": self.dataset_format.format_counts,
            "csv_path": self.dataset_format.csv_path,
            "dicom_root": self.dataset_format.dicom_root,
            "has_features_txt": self.dataset_format.has_features_txt,
            "has_csv": self.dataset_format.has_csv,
            "warnings": self.dataset_format.warnings,
            "suggestions": self.dataset_format.suggestions,
        }

    def get_comprehensive_defaults(
        self, dataset_size: Optional[int] = None, image_size: int = 224
    ) -> Dict[str, Any]:
        """Get comprehensive defaults including hardware, training, and dataset info.

        Combines hardware detection, training defaults, and dataset-specific
        recommendations into a single configuration dictionary.

        Args:
            dataset_size: Number of samples in dataset (uses dataset_format if available)
            image_size: Input image size

        Returns:
            Dictionary with hardware, training, and dataset information

        Examples:
            >>> defaults = SmartDefaults()
            >>> config = defaults.get_comprehensive_defaults(dataset_size=1000)
            >>> config['hardware']['device']
            'cuda'
            >>> config['training']['batch_size']
            16
        """
        train_defaults = self.get_training_defaults(
            dataset_size=dataset_size, image_size=image_size
        )
        dataset_info = self.get_dataset_info()

        result: Dict[str, Any] = {
            "hardware": {
                "device": self.device_type,
                "system_memory_gb": self.system_memory_gb,
                "mixed_precision_available": self.get_mixed_precision(),
            },
            "training": train_defaults,
        }

        # Add dataset info if available
        if dataset_info:
            result["dataset"] = dataset_info

        return result

    def get_warnings(self) -> List[str]:
        """Get dataset-specific warnings if available.

        Returns:
            List of warning messages, empty list if no dataset_format

        Examples:
            >>> defaults = SmartDefaults(dataset_format=detected_format)
            >>> warnings = defaults.get_warnings()
            >>> if warnings:
            ...     print("Warnings:", warnings)
        """
        if not self.dataset_format:
            return []
        return self.dataset_format.warnings

    def get_suggestions(self) -> List[str]:
        """Get dataset-specific preprocessing suggestions if available.

        Returns:
            List of suggestion messages, empty list if no dataset_format

        Examples:
            >>> defaults = SmartDefaults(dataset_format=detected_format)
            >>> suggestions = defaults.get_suggestions()
            >>> for suggestion in suggestions:
            ...     print(f"  - {suggestion}")
        """
        if not self.dataset_format:
            return []
        return self.dataset_format.suggestions

    @staticmethod
    def _nearest_power_of_2(n: int) -> int:
        """Round to nearest power of 2.

        Args:
            n: Input number

        Returns:
            Nearest power of 2

        Examples:
            >>> SmartDefaults._nearest_power_of_2(15)
            16
            >>> SmartDefaults._nearest_power_of_2(17)
            16
        """
        if n <= 0:
            return 1

        # Find closest power of 2
        import math

        power = round(math.log2(n))
        return 2**power

    def print_recommendations(
        self, dataset_size: Optional[int] = None, image_size: int = 224
    ) -> None:
        """Print recommended configuration to console.

        Args:
            dataset_size: Number of samples in dataset
            image_size: Input image size
        """
        print("\n=== Smart Configuration Recommendations ===")
        print(f"Device: {self.device_type.upper()}")
        print(f"System Memory: {self.system_memory_gb:.1f} GB")
        print()

        # Print dataset info if available
        if self.dataset_format:
            print("Dataset Information:")
            print(f"  Type: {self.dataset_format.dataset_type}")
            print(f"  Format: {self.dataset_format.image_format}")
            print(f"  Images: {self.dataset_format.image_count}")
            if self.dataset_format.warnings:
                print(f"  Warnings: {len(self.dataset_format.warnings)}")
                for warning in self.dataset_format.warnings:
                    print(f"    - {warning}")
            print()

        train_config = self.get_training_defaults(
            dataset_size=dataset_size, image_size=image_size
        )
        print("Training Configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")

        # Print suggestions if available
        if self.dataset_format and self.dataset_format.suggestions:
            print()
            print("Preprocessing Suggestions:")
            for suggestion in self.dataset_format.suggestions:
                print(f"  - {suggestion}")

        print()
        inference_config = self.get_inference_defaults(
            dataset_size=dataset_size, image_size=image_size
        )
        print("Inference Configuration:")
        for key, value in inference_config.items():
            print(f"  {key}: {value}")


def get_smart_defaults(
    dataset_size: Optional[int] = None, image_size: int = 224
) -> Dict[str, Any]:
    """Convenience function to get smart training defaults.

    Args:
        dataset_size: Number of samples in dataset
        image_size: Input image size

    Returns:
        Dictionary with recommended training configuration

    Examples:
        >>> defaults = get_smart_defaults(dataset_size=1000)
        >>> defaults['batch_size']
        16
    """
    sd = SmartDefaults()
    return sd.get_training_defaults(dataset_size=dataset_size, image_size=image_size)
