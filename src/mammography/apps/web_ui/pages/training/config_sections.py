"""Configuration sections for the Streamlit Training page."""

from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    if st is None:
        raise ImportError("Streamlit is required to run the web UI dashboard.") from _STREAMLIT_IMPORT_ERROR


def render_data_section() -> Dict[str, Any]:
    """Render data configuration section and return parameters."""
    _require_streamlit()
    st.subheader("📂 Data Configuration")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config["dataset"] = st.selectbox(
            "Dataset Preset",
            options=["", "archive", "mamografias", "patches_completo"],
            help="Preset dataset configuration (archive/mamografias/patches_completo)",
        )

        config["csv"] = st.text_input(
            "CSV Path",
            value="classificacao.csv",
            help="Path to CSV file, directory with featureS.txt, or manual path",
        )

        config["dicom_root"] = st.text_input(
            "DICOM Root Path",
            value="archive",
            help="Root directory for DICOM files (used with classificacao.csv)",
        )

        config["outdir"] = st.text_input(
            "Output Directory",
            value="outputs/run",
            help="Directory for saving training outputs",
        )

    with col2:
        config["cache_mode"] = st.selectbox(
            "Cache Mode",
            options=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"],
            help="Data caching strategy for faster loading",
        )

        config["cache_dir"] = st.text_input(
            "Cache Directory",
            value="outputs/cache",
            help="Directory for cached data (used when cache mode is disk or tensor-disk)",
        )

        config["embeddings_dir"] = st.text_input(
            "Embeddings Directory",
            value="",
            help="Directory with features.npy + metadata.csv (optional)",
        )

        config["log_level"] = st.selectbox(
            "Log Level",
            options=["critical", "error", "warning", "info", "debug"],
            index=3,  # default to "info"
            help="Logging verbosity level",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        config["subset"] = st.number_input(
            "Subset Size",
            min_value=0,
            value=0,
            step=1,
            help="Limit number of samples (0 = use all)",
        )

    with col2:
        config["include_class_5"] = st.checkbox(
            "Include Class 5",
            value=False,
            help="Keep samples with classification 5 when loading classificacao.csv",
        )

    with col3:
        config["val_frac"] = st.number_input(
            "Validation Fraction",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.05,
            help="Fraction of data to use for validation",
        )

    return config

def render_model_section() -> Dict[str, Any]:
    """Render model configuration section and return parameters."""
    _require_streamlit()
    st.subheader("🏗️ Model Configuration")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["arch"] = st.selectbox(
            "Architecture",
            options=["efficientnet_b0", "resnet50"],
            help="Model architecture to use",
        )

    with col2:
        config["classes"] = st.selectbox(
            "Task",
            options=["multiclass", "binary"],
            help="Classification task: multiclass (BI-RADS 1-4) or binary (A/B vs C/D)",
        )

    with col3:
        config["pretrained"] = st.checkbox(
            "Use Pretrained Weights",
            value=True,
            help="Initialize with ImageNet pretrained weights when available",
        )

    # View-specific training
    with st.expander("🔍 View-Specific Training"):
        config["view_specific_training"] = st.checkbox(
            "Enable View-Specific Training",
            value=False,
            help="Train separate models for each view (CC/MLO)",
        )

        if config["view_specific_training"]:
            config["views"] = st.text_input(
                "Views to Train",
                value="CC,MLO",
                help="Comma-separated list of views to train",
            )

            config["ensemble_method"] = st.selectbox(
                "Ensemble Method",
                options=["none", "average", "weighted", "max"],
                help="Method for combining predictions from different views",
            )
        else:
            config["views"] = ""
            config["ensemble_method"] = "none"

    return config

def render_training_section() -> Dict[str, Any]:
    """Render training hyperparameters section and return parameters."""
    _require_streamlit()
    st.subheader("🎯 Training Hyperparameters")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["epochs"] = st.number_input(
            "Epochs",
            min_value=1,
            value=100,
            step=1,
            help="Number of training epochs",
        )

        config["batch_size"] = st.number_input(
            "Batch Size",
            min_value=1,
            value=16,
            step=1,
            help="Training batch size",
        )

        config["lr"] = st.number_input(
            "Learning Rate",
            min_value=1e-7,
            max_value=1.0,
            value=1e-4,
            format="%.1e",
            help="Learning rate for the classification head",
        )

    with col2:
        config["backbone_lr"] = st.number_input(
            "Backbone Learning Rate",
            min_value=1e-7,
            max_value=1.0,
            value=1e-5,
            format="%.1e",
            help="Learning rate for the backbone (if trainable)",
        )

        config["weight_decay"] = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=1.0,
            value=1e-4,
            format="%.1e",
            help="L2 regularization weight decay",
        )

        config["img_size"] = st.number_input(
            "Image Size",
            min_value=32,
            max_value=1024,
            value=512,
            step=32,
            help="Input image size (will be resized to this)",
        )

    with col3:
        config["seed"] = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1,
            help="Random seed for reproducibility",
        )

        config["device"] = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda", "mps"],
            help="Compute device (auto will select the best available)",
        )

        config["amp"] = st.checkbox(
            "Enable AMP",
            value=False,
            help="Enable Automatic Mixed Precision for faster training on CUDA/MPS",
        )

    return config

def render_optimization_section() -> Dict[str, Any]:
    """Render optimization settings section and return parameters."""
    _require_streamlit()
    st.subheader("⚡ Optimization Settings")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backbone Training**")

        config["train_backbone"] = st.checkbox(
            "Train Backbone",
            value=False,
            help="Train the backbone feature extractor",
        )

        config["unfreeze_last_block"] = st.checkbox(
            "Unfreeze Last Block",
            value=True,
            help="Unfreeze the last block of the backbone",
        )

        config["warmup_epochs"] = st.number_input(
            "Warmup Epochs",
            min_value=0,
            value=0,
            step=1,
            help="Number of warmup epochs before full training",
        )

    with col2:
        st.markdown("**Performance Options**")

        config["deterministic"] = st.checkbox(
            "Deterministic Mode",
            value=False,
            help="Enable deterministic algorithms (may be slower)",
        )

        config["allow_tf32"] = st.checkbox(
            "Allow TF32",
            value=True,
            help="Allow TensorFloat-32 for faster computation on Ampere GPUs",
        )

        config["fused_optim"] = st.checkbox(
            "Fused Optimizer",
            value=False,
            help="Use fused AdamW optimizer on CUDA (faster but more memory)",
        )

        config["torch_compile"] = st.checkbox(
            "Torch Compile",
            value=False,
            help="Optimize model with torch.compile (PyTorch 2.0+)",
        )

    # Learning rate scheduling
    with st.expander("📉 Learning Rate Scheduling"):
        config["scheduler"] = st.selectbox(
            "Scheduler Type",
            options=["auto", "none", "plateau", "cosine", "step"],
            help="Learning rate scheduler strategy",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            config["lr_reduce_patience"] = st.number_input(
                "LR Reduce Patience",
                min_value=0,
                value=0,
                step=1,
                help="Epochs to wait before reducing LR on plateau (0 = disabled)",
            )

            config["lr_reduce_factor"] = st.number_input(
                "LR Reduce Factor",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help="Factor by which to reduce learning rate",
            )

        with col2:
            config["lr_reduce_min_lr"] = st.number_input(
                "Min Learning Rate",
                min_value=1e-10,
                max_value=1e-3,
                value=1e-7,
                format="%.1e",
                help="Minimum learning rate threshold",
            )

            config["lr_reduce_cooldown"] = st.number_input(
                "LR Reduce Cooldown",
                min_value=0,
                value=0,
                step=1,
                help="Epochs to wait before resuming normal operation after LR reduction",
            )

        with col3:
            config["scheduler_step_size"] = st.number_input(
                "Step Size",
                min_value=1,
                value=5,
                step=1,
                help="Step size for step scheduler",
            )

            config["scheduler_gamma"] = st.number_input(
                "Gamma",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help="Multiplicative factor for step scheduler",
            )

    # Early stopping
    with st.expander("⏹️ Early Stopping"):
        config["early_stop_patience"] = st.number_input(
            "Early Stop Patience",
            min_value=0,
            value=0,
            step=1,
            help="Epochs to wait for improvement before stopping (0 = disabled)",
        )

        config["early_stop_min_delta"] = st.number_input(
            "Min Delta",
            min_value=0.0,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help="Minimum change to qualify as improvement",
        )

    return config
