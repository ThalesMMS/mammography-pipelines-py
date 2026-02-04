#
# 2_🔍_Explainability.py
# mammography-pipelines
#
# Streamlit page for generating explainability visualizations (GradCAM) for model predictions.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Explainability page for visualizing GradCAM heatmaps on mammography predictions."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from mammography.models.nets import build_model
from mammography.utils.common import resolve_device, configure_runtime
from mammography.io.dicom import is_dicom_path, dicom_to_pil_rgb
from mammography.vis.explainability import GradCAMExplainer
from mammography.apps.web_ui.utils import ensure_shared_session_state
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    if "explainability_image" not in st.session_state:
        st.session_state.explainability_image = None
    if "explainability_model" not in st.session_state:
        st.session_state.explainability_model = None
    if "explainability_checkpoint_path" not in st.session_state:
        st.session_state.explainability_checkpoint_path = None
    if "explainability_heatmap" not in st.session_state:
        st.session_state.explainability_heatmap = None
    if "explainability_prediction" not in st.session_state:
        st.session_state.explainability_prediction = None


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from state dict keys if present.

    This is needed when loading models that were trained with DataParallel.
    """
    if not state_dict:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(
    checkpoint_path: str,
    arch: str,
    num_classes: int,
    device: str,
) -> tuple[torch.nn.Module, str]:
    """Load a model checkpoint and return the initialized model.

    Args:
        checkpoint_path: Path to the checkpoint file
        arch: Model architecture ('resnet50' or 'efficientnet_b0')
        num_classes: Number of output classes
        device: Device to load model on ('auto', 'cuda', 'cpu', or 'mps')

    Returns:
        Tuple of (loaded model, device string)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Resolve device
    device_obj = resolve_device(device)
    configure_runtime(device_obj, deterministic=False, allow_tf32=True)

    # Build model architecture
    model = build_model(
        arch=arch,
        num_classes=num_classes,
        train_backbone=False,
        unfreeze_last_block=False,
        pretrained=False,
    )

    # Load checkpoint
    try:
        state = torch.load(checkpoint_path, map_location=device_obj)

        # Extract state_dict if checkpoint is a dict with metadata
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip 'module.' prefix if present (from DataParallel training)
        if isinstance(state, dict):
            state = _strip_module_prefix(state)

        # Load state dict into model
        model.load_state_dict(state, strict=False)

    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    # Move model to device and set to eval mode
    model.to(device_obj)
    model.eval()

    return model, str(device_obj)


def _load_and_preprocess_image(
    file_path: str,
    img_size: int,
) -> tuple[Image.Image, torch.Tensor]:
    """Load and preprocess an image for inference.

    Args:
        file_path: Path to the image file
        img_size: Target image size

    Returns:
        Tuple of (PIL Image for display, preprocessed tensor for model)
    """
    # Load image
    if is_dicom_path(file_path):
        pil_img = dicom_to_pil_rgb(file_path)
    else:
        pil_img = Image.open(file_path).convert("RGB")

    # Create transform pipeline
    transform = Compose([
        ToImage(),
        Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToDtype(torch.float32, scale=True),
    ])

    # Apply transform
    tensor = transform(pil_img)

    # Normalize (ImageNet stats - standard for medical imaging)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return pil_img, tensor


def _get_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """Get the target layer for GradCAM based on model architecture.

    Args:
        model: PyTorch model
        arch: Model architecture name

    Returns:
        Target layer for GradCAM
    """
    if arch == "resnet50":
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        else:
            raise ValueError("ResNet model doesn't have expected 'layer4' attribute")
    elif arch == "efficientnet_b0":
        if hasattr(model, "features"):
            return model.features[-1]
        else:
            raise ValueError("EfficientNet model doesn't have expected 'features' attribute")
    else:
        raise ValueError(f"Unsupported architecture for GradCAM: {arch}")


def _generate_gradcam_overlay(
    explainer: GradCAMExplainer,
    image_tensor: torch.Tensor,
    target_class: Optional[int],
    original_image: Image.Image,
    alpha: float,
    colormap: str,
) -> tuple[Image.Image, int, np.ndarray]:
    """Generate GradCAM heatmap overlay.

    Args:
        explainer: GradCAMExplainer instance
        image_tensor: Preprocessed image tensor
        target_class: Target class for GradCAM (None for predicted class)
        original_image: Original PIL image
        alpha: Overlay alpha value
        colormap: Colormap name

    Returns:
        Tuple of (overlay image, predicted class, probabilities)
    """
    # Generate heatmap
    heatmap = explainer.generate_heatmap(image_tensor, target_class=target_class)

    # Get prediction
    with torch.no_grad():
        logits = explainer.model(image_tensor.unsqueeze(0).to(next(explainer.model.parameters()).device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))

    # Convert heatmap to numpy
    heatmap_np = heatmap.cpu().numpy()

    # Create overlay using the explainer's save_overlay method
    # We'll use a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        explainer.save_overlay(
            image_tensor,
            heatmap,
            tmp_path,
            alpha=alpha,
            colormap=colormap,
        )
        overlay_img = Image.open(tmp_path)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return overlay_img, pred_class, probs


def main() -> None:
    """Render the explainability page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Explainability - Mammography Pipelines",
        page_icon="🔍",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("🔍 Model Explainability")

    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
    <h3 style="color: #856404; margin-top: 0;">⚠️ EDUCATIONAL RESEARCH USE ONLY</h3>
    <p style="color: #856404; margin-bottom: 0;">
    This tool is for <strong>educational and research purposes only</strong>. It is <strong>NOT</strong>
    intended for clinical diagnosis or treatment. All results should be validated by qualified
    medical professionals before any clinical decision-making.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.header("GradCAM Visualization")

    st.markdown("""
    Generate Gradient-weighted Class Activation Mapping (GradCAM) heatmaps to visualize
    which regions of the mammography image contributed most to the model's prediction.

    GradCAM highlights important features by showing areas of high activation in the
    final convolutional layers of the neural network.
    """)

    st.info(
        "💡 **Quick Start:** (1) Enter your model checkpoint path, "
        "(2) Upload an image or use one from the Inference page, "
        "(3) Configure GradCAM options, and (4) Click 'Generate GradCAM'."
    )

    # Model selection
    st.subheader("1. Select Model Checkpoint")

    col1, col2 = st.columns([2, 1])

    with col1:
        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            placeholder="path/to/checkpoint.pt",
            help="Path to the trained model checkpoint file (.pt or .pth)",
        )

    with col2:
        arch = st.selectbox(
            "Model Architecture",
            options=["resnet50", "efficientnet_b0"],
            help="Architecture used for training the model",
        )

    # Number of classes
    num_classes = st.radio(
        "Number of Classes",
        options=[2, 4],
        horizontal=True,
        help="2 for binary classification (non-dense/dense), 4 for density levels (BI-RADS A-D)",
    )

    # File upload
    st.subheader("2. Upload Image")

    # Check if there's a shared image from inference page
    if st.session_state.shared_uploaded_image is not None:
        st.info(
            f"💡 Image from Inference page available: {st.session_state.shared_uploaded_image_name}. "
            "You can use this image or upload a new one below."
        )
        use_shared_image = st.checkbox(
            f"Use image from Inference: {st.session_state.shared_uploaded_image_name}",
            value=True,
            help="Use the image that was uploaded on the Inference page"
        )
    else:
        use_shared_image = False

    uploaded_file = st.file_uploader(
        "Choose a mammography image",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        help="Upload a single mammography image in PNG, JPEG, or DICOM format",
        disabled=use_shared_image,
    )

    # GradCAM options
    st.subheader("3. GradCAM Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha = st.slider(
            "Overlay Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Transparency of the heatmap overlay (0=transparent, 1=opaque)",
        )

    with col2:
        colormap = st.selectbox(
            "Colormap",
            options=["jet", "viridis", "plasma", "inferno", "turbo"],
            help="Color scheme for the heatmap visualization",
        )

    with col3:
        target_class_option = st.selectbox(
            "Target Class",
            options=["Predicted"] + [f"Class {i+1}" for i in range(num_classes)],
            help="Class to generate GradCAM for (Predicted = use model's prediction)",
        )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            img_size = st.number_input(
                "Image Size",
                min_value=128,
                max_value=1024,
                value=224,
                step=32,
                help="Input image size for the model",
            )

        with col2:
            device = st.selectbox(
                "Device",
                options=["auto", "cuda", "cpu", "mps"],
                help="Compute device for inference",
            )

        with col3:
            target_layer_mode = st.selectbox(
                "Target Layer",
                options=["Auto-detect", "Custom"],
                help="Layer to use for GradCAM visualization",
            )

        # Custom target layer input (only show if "Custom" is selected)
        if target_layer_mode == "Custom":
            custom_target_layer = st.text_input(
                "Custom Target Layer Name",
                placeholder="e.g., layer4[-1] for ResNet, features[-1] for EfficientNet",
                help="Enter the exact layer name/path in the model",
            )
        else:
            custom_target_layer = None

    # Model status indicator
    if st.session_state.explainability_model is not None:
        st.success(
            f"✅ Model loaded: {st.session_state.explainability_checkpoint_path} "
            f"(Architecture: {arch}, Classes: {num_classes})"
        )
        if st.button("🗑️ Clear Loaded Model"):
            st.session_state.explainability_model = None
            st.session_state.explainability_checkpoint_path = None
            st.rerun()

    # Generate GradCAM button
    st.subheader("4. Generate GradCAM")

    # Determine if we have an image (either uploaded or from shared state)
    has_image = uploaded_file is not None or (use_shared_image and st.session_state.shared_uploaded_image is not None)

    if st.button("🔥 Generate GradCAM", type="primary", disabled=not (checkpoint_path and has_image)):
        if not checkpoint_path:
            st.error("❌ Please provide a model checkpoint path.")
            st.info("💡 Enter the file path to your trained model (e.g., 'outputs/run/model_best.pt')")
        elif not os.path.exists(checkpoint_path):
            st.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            st.info("💡 Please verify the file path exists and try again.")
        elif not has_image:
            st.error("❌ Please upload an image or use the shared image from Inference page.")
            st.info("💡 Either upload a new image or check the box to use an image from the Inference page.")
        else:
            # Load checkpoint if not already loaded or if path changed
            if (
                st.session_state.explainability_model is None
                or st.session_state.explainability_checkpoint_path != checkpoint_path
            ):
                with st.spinner(f"Loading model checkpoint from {checkpoint_path}..."):
                    try:
                        model, device_str = _load_checkpoint(
                            checkpoint_path=checkpoint_path,
                            arch=arch,
                            num_classes=num_classes,
                            device=device,
                        )
                        st.session_state.explainability_model = model
                        st.session_state.explainability_checkpoint_path = checkpoint_path
                        st.success(
                            f"✅ Model loaded successfully! "
                            f"({arch}, {num_classes} classes, device: {device_str})"
                        )
                    except FileNotFoundError as exc:
                        st.error(f"❌ Checkpoint file not found: {exc}")
                        st.info("💡 Verify the checkpoint path is correct and the file exists.")
                        st.stop()
                    except RuntimeError as exc:
                        st.error(f"❌ Failed to load checkpoint: {exc}")
                        st.info(
                            "💡 This may happen if:\n"
                            "- The checkpoint was trained with a different architecture\n"
                            "- The checkpoint file is corrupted\n"
                            "- The number of classes doesn't match the trained model"
                        )
                        st.stop()
                    except Exception as exc:
                        st.error(f"❌ Unexpected error loading checkpoint: {exc}")
                        st.info("💡 Check that PyTorch is installed and the checkpoint file is valid.")
                        st.stop()
            else:
                st.info(f"Using previously loaded model from {checkpoint_path}")

            # Get image path (either from upload or shared state)
            if use_shared_image and st.session_state.shared_uploaded_image_path:
                # Use the shared image from inference page
                file_path = st.session_state.shared_uploaded_image_path
                temp_dir = None
                st.success(f"✅ Using shared image: {st.session_state.shared_uploaded_image_name}")
            else:
                # Save uploaded file to temporary location
                with st.spinner("Loading image..."):
                    temp_dir = tempfile.mkdtemp(prefix="mammography_explain_")
                    file_path = os.path.join(temp_dir, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"✅ Image saved: {uploaded_file.name}")

            # Load and preprocess image
            with st.spinner("Preprocessing image..."):
                try:
                    original_img, image_tensor = _load_and_preprocess_image(
                        file_path=file_path,
                        img_size=img_size,
                    )
                    st.session_state.explainability_image = original_img
                    st.success("✅ Image preprocessed successfully")
                except Exception as exc:
                    st.error(f"❌ Failed to preprocess image: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The image file is corrupted or in an unsupported format\n"
                        "- DICOM files are missing required metadata\n"
                        "- The file is not a valid image"
                    )
                    st.stop()

            # Create GradCAM explainer
            with st.spinner("Initializing GradCAM explainer..."):
                try:
                    model = st.session_state.explainability_model

                    # Determine target layer
                    if target_layer_mode == "Custom" and custom_target_layer:
                        # Parse custom target layer path (e.g., "layer4[-1]")
                        try:
                            layer_parts = custom_target_layer.strip().split(".")
                            target_layer = model
                            for part in layer_parts:
                                # Handle array indexing like "layer4[-1]"
                                if "[" in part and "]" in part:
                                    attr_name, idx = part.split("[")
                                    idx = int(idx.replace("]", ""))
                                    target_layer = getattr(target_layer, attr_name)[idx]
                                else:
                                    target_layer = getattr(target_layer, part)
                            st.info(f"Using custom target layer: {custom_target_layer}")
                        except Exception as exc:
                            st.error(f"❌ Failed to parse custom target layer '{custom_target_layer}': {exc}")
                            st.stop()
                    else:
                        # Auto-detect target layer
                        target_layer = _get_target_layer(model, arch)
                        st.info(f"Using auto-detected target layer for {arch}")

                    explainer = GradCAMExplainer(
                        model=model,
                        target_layer=target_layer,
                        device=next(model.parameters()).device,
                    )
                    st.success("✅ GradCAM explainer initialized")
                except Exception as exc:
                    st.error(f"❌ Failed to initialize GradCAM: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The model architecture doesn't have the expected layers\n"
                        "- The custom target layer path is incorrect\n"
                        "- The model is not compatible with GradCAM"
                    )
                    st.stop()

            # Generate GradCAM
            with st.spinner("Generating GradCAM heatmap..."):
                try:
                    # Parse target class
                    target_class = None
                    if target_class_option != "Predicted":
                        target_class = int(target_class_option.split()[-1]) - 1

                    # Generate overlay
                    overlay_img, pred_class, probs = _generate_gradcam_overlay(
                        explainer=explainer,
                        image_tensor=image_tensor,
                        target_class=target_class,
                        original_image=original_img,
                        alpha=alpha,
                        colormap=colormap,
                    )

                    # Store results in session state
                    st.session_state.explainability_heatmap = overlay_img
                    st.session_state.explainability_prediction = {
                        "class": pred_class,
                        "probs": probs,
                    }

                    st.success("✅ GradCAM heatmap generated successfully!")

                except Exception as exc:
                    st.error(f"❌ Failed to generate GradCAM: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The image is incompatible with the model\n"
                        "- There's a device/memory error\n"
                        "- The target class is out of range\n"
                        "- The model architecture is incompatible with GradCAM"
                    )
                    with st.expander("🔍 View Full Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.stop()
                finally:
                    # Clean up temporary files (only if we created a temp dir)
                    if temp_dir is not None:
                        try:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        except Exception:
                            pass  # Best effort cleanup

    # Display results
    if st.session_state.explainability_heatmap is not None:
        st.header("Results")

        # Display prediction
        pred_info = st.session_state.explainability_prediction
        pred_class = pred_info["class"]
        probs = pred_info["probs"]

        # BI-RADS labels
        if num_classes == 4:
            birads_map = {
                0: "Fatty (A)",
                1: "Mostly Fatty (B)",
                2: "Mostly Dense (C)",
                3: "Dense (D)",
            }
        else:
            birads_map = {
                0: "Non-Dense",
                1: "Dense",
            }

        prediction_label = birads_map.get(pred_class, f"Class {pred_class + 1}")
        confidence = probs[pred_class] * 100

        st.subheader(f"Prediction: {prediction_label} (Confidence: {confidence:.2f}%)")

        # Display probabilities
        prob_data = {
            "Class": [birads_map.get(i, f"Class {i+1}") for i in range(len(probs))],
            "Probability": [f"{p*100:.2f}%" for p in probs],
        }
        st.table(prob_data)

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.explainability_image, use_container_width=True)

        with col2:
            st.subheader("GradCAM Overlay")
            st.image(st.session_state.explainability_heatmap, use_container_width=True)

        # Download button
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            st.session_state.explainability_heatmap.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                img_bytes = f.read()
            st.download_button(
                label="📥 Download GradCAM Overlay",
                data=img_bytes,
                file_name="gradcam_overlay.png",
                mime="image/png",
            )
            # Clean up
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use This Page

        1. **Select a Model**: Enter the path to your trained model checkpoint file
        2. **Choose Architecture**: Select the architecture that matches your checkpoint
        3. **Set Number of Classes**: Choose 2 (binary) or 4 (density levels)
        4. **Upload Image**: Select a single mammography image to analyze
        5. **Configure GradCAM**: Adjust alpha, colormap, and target class
        6. **Generate GradCAM**: Click the button to create the visualization

        ### Understanding GradCAM

        **GradCAM (Gradient-weighted Class Activation Mapping)** visualizes which regions
        of the input image were most important for the model's prediction.

        - **Red/Yellow regions**: High importance - these areas strongly influenced the prediction
        - **Blue/Purple regions**: Low importance - minimal contribution to the prediction
        - **Alpha**: Controls transparency (lower = see more of original image)

        ### Target Class Options

        - **Predicted**: Generate GradCAM for the class the model predicted
        - **Class N**: Generate GradCAM for a specific class (useful for comparing activations)

        ### Colormaps

        - **jet**: Classic red-yellow-blue gradient (default)
        - **viridis**: Perceptually uniform purple-green-yellow
        - **plasma**: Perceptually uniform purple-orange-yellow
        - **inferno**: Perceptually uniform black-red-yellow
        - **turbo**: Enhanced rainbow colormap

        ### Target Layer (Advanced)

        GradCAM visualizes activations from a specific convolutional layer:

        - **Auto-detect**: Automatically selects the last convolutional layer
          - ResNet50: `layer4[-1]` (final residual block)
          - EfficientNet B0: `features[-1]` (final MBConv block)
        - **Custom**: Specify a different layer to visualize earlier or intermediate activations
          - Example for ResNet: `layer4.0`, `layer3[-1]`, `layer2[1]`
          - Example for EfficientNet: `features.6`, `features[4]`

        Visualizing earlier layers shows lower-level features (edges, textures), while later
        layers show higher-level semantic features relevant to the classification.

        ### Troubleshooting

        - **No heatmap visible**: Try increasing alpha value
        - **Model error**: Verify checkpoint matches the selected architecture and num_classes
        - **Out of Memory**: Try reducing image size in advanced options

        ### References

        - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
          Gradient-based Localization", ICCV 2017
        """)


if __name__ == "__main__":
    main()
