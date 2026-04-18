"""
Utilities for ResNet50_test
Educational Research Project - NOT for clinical use

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from .device_detection import (
    DeviceDetector,
    get_device_config,
    get_optimal_device,
    print_device_status,
)
from .bool_flags import normalize_bool_flags, parse_bool_literal
from .cli_args import add_tracking_args, serialize_tracking_args
from .config_loader import (
    DEFAULT_CONFIGS,
    coerce_cli_args,
    dict_to_cli_args,
    load_config_args,
    read_config,
)
from .export_formats import (
    export_figure_multi_format,
    parse_export_formats,
    plot_history_format,
    save_metrics_figure_format,
)
from .embeddings import extract_embedding_matrix

__all__ = [
    "DEFAULT_CONFIGS",
    "DeviceDetector",
    "add_tracking_args",
    "coerce_cli_args",
    "dict_to_cli_args",
    "export_figure_multi_format",
    "extract_embedding_matrix",
    "get_device_config",
    "get_optimal_device",
    "load_config_args",
    "normalize_bool_flags",
    "parse_bool_literal",
    "parse_export_formats",
    "plot_history_format",
    "print_device_status",
    "read_config",
    "save_metrics_figure_format",
    "serialize_tracking_args",
]
