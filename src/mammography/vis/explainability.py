#
# explainability.py
# mammography-pipelines
#
# Explainability utilities for model interpretability: GradCAM, attention maps, and other visualization techniques.
#
# Thales Matheus Mendonça Santos - January 2026
#
"""
Explainability utilities for model interpretability.

Provides:
- GradCAM (Gradient-weighted Class Activation Mapping) for CNN models
- Attention map visualization for Vision Transformers
- Batch processing for multiple images
- Export utilities for heatmap overlays
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple, Union

logger = logging.getLogger("mammography")


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN model interpretability.

    Generates heatmaps showing which regions of an input image are most important for
    the model's prediction. Supports ResNet (layer4) and EfficientNet (features) architectures.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

    Example:
        >>> model = load_model("resnet50.pth")
        >>> explainer = GradCAMExplainer(model, device="cuda")
        >>> heatmap = explainer.generate_heatmap(image_tensor, target_class=1)
        >>> explainer.save_overlay(image_tensor, heatmap, "output.png")
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize GradCAM explainer.

        Args:
            model: PyTorch model to explain
            target_layer: Layer to compute gradients for. If None, automatically
                         selects layer4[-1] for ResNet or features[-1] for EfficientNet
            device: Device to run computations on. If None, uses model's current device
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

        # Auto-detect target layer if not provided
        if target_layer is None:
            if hasattr(model, "layer4"):
                self.target_layer = model.layer4[-1]
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError(
                    "Cannot auto-detect target layer. Model must have 'layer4' (ResNet) "
                    "or 'features' (EfficientNet) attribute, or target_layer must be provided."
                )
        else:
            self.target_layer = target_layer

        self.activations: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []
        self._hooks_registered = False

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        if self._hooks_registered:
            return

        def fwd_hook(module, input, output):
            self.activations.append(output.detach())

        def bwd_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0].detach())

        self.handle_fwd = self.target_layer.register_forward_hook(fwd_hook)
        self.handle_bwd = self.target_layer.register_full_backward_hook(bwd_hook)
        self._hooks_registered = True

    def _remove_hooks(self):
        """Remove registered hooks."""
        if self._hooks_registered:
            self.handle_fwd.remove()
            self.handle_bwd.remove()
            self._hooks_registered = False

    def _clear_buffers(self):
        """Clear activation and gradient buffers."""
        self.activations.clear()
        self.gradients.clear()

    def generate_heatmap(
        self,
        x: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for input image(s).

        Args:
            x: Input tensor of shape (B, C, H, W) or (C, H, W)
            target_class: Target class index for backpropagation. If None, uses
                         the predicted class. Can be int (same class for all images)
                         or tensor of shape (B,) for batch processing

        Returns:
            Heatmap tensor of shape (B, H, W) or (H, W), normalized to [0, 1]
        """
        self.model.eval()
        single_image = x.ndim == 3
        if single_image:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        x.requires_grad_(True)

        self._clear_buffers()
        self._register_hooks()

        try:
            # Forward pass
            output = self.model(x)

            # Determine target classes
            if target_class is None:
                target_class = output.argmax(dim=1)
            elif isinstance(target_class, int):
                target_class = torch.full((x.shape[0],), target_class, device=self.device)
            else:
                target_class = target_class.to(self.device)

            # Backward pass on target class scores
            selected = output.gather(1, target_class.unsqueeze(1)).sum()
            self.model.zero_grad()
            selected.backward()

            # Compute Grad-CAM
            if not self.activations or not self.gradients:
                logger.warning("No activations or gradients captured. Returning zeros.")
                return torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=self.device)

            act = self.activations[0]  # (B, C, H', W')
            grad = self.gradients[0]   # (B, C, H', W')

            # Global average pooling of gradients
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

            # Weighted combination of activation maps
            cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, H', W')

            # ReLU to keep only positive influences
            cam = torch.relu(cam)

            # Upsample to input size
            cam = torch.nn.functional.interpolate(
                cam,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            cam = cam.squeeze(1)  # (B, H, W)

            # Normalize per image to [0, 1]
            cam_min = cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max + 1e-6)

            return cam[0] if single_image else cam

        finally:
            self._remove_hooks()
            self._clear_buffers()

    def save_overlay(
        self,
        x: torch.Tensor,
        heatmap: torch.Tensor,
        output_path: Union[str, Path],
        alpha: float = 0.35,
        colormap: str = "jet"
    ):
        """
        Save GradCAM heatmap overlaid on original image.

        Args:
            x: Input image tensor of shape (C, H, W), normalized with ImageNet stats
            heatmap: Heatmap tensor of shape (H, W), values in [0, 1]
            output_path: Path to save the blended image
            alpha: Blending factor for overlay (0=original, 1=heatmap)
            colormap: Matplotlib colormap name for heatmap visualization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Denormalize image (ImageNet stats)
        img = x.detach().cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

        # Convert heatmap to RGB using colormap
        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_uint8 = np.uint8(255 * heatmap_np)

        # Convert to PIL and apply colormap
        from matplotlib import cm as mpl_cm
        cmap = mpl_cm.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap_uint8)[:, :, :3]  # Drop alpha channel

        # Create PIL images
        base = Image.fromarray(np.uint8(img * 255)).convert("RGB")
        heatmap_img = Image.fromarray(np.uint8(heatmap_rgb * 255)).convert("RGB")

        # Blend and save
        blended = Image.blend(base, heatmap_img, alpha=alpha)
        blended.save(output_path)

    def save_batch_overlays(
        self,
        x: torch.Tensor,
        heatmaps: torch.Tensor,
        output_dir: Union[str, Path],
        metas: Optional[List[Dict[str, Any]]] = None,
        prefix: str = "gradcam",
        alpha: float = 0.35,
        colormap: str = "jet"
    ) -> int:
        """
        Save GradCAM overlays for a batch of images.

        Args:
            x: Batch of input images, shape (B, C, H, W)
            heatmaps: Batch of heatmaps, shape (B, H, W)
            output_dir: Directory to save overlays
            metas: Optional list of metadata dicts with 'accession' or other identifiers
            prefix: Filename prefix for saved images
            alpha: Blending factor for overlay
            colormap: Matplotlib colormap name

        Returns:
            Number of images successfully saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i in range(x.shape[0]):
            try:
                identifier = f"sample_{i}"
                if metas and i < len(metas):
                    identifier = metas[i].get("accession", identifier)

                fname = output_dir / f"{prefix}_{i}_{identifier}.png"
                self.save_overlay(x[i], heatmaps[i], fname, alpha=alpha, colormap=colormap)
                saved += 1
            except Exception as exc:
                logger.warning(f"Failed to save overlay {i}: {exc}")

        return saved

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


class ViTAttentionVisualizer:
    """
    Vision Transformer attention map visualizer for model interpretability.

    Extracts and visualizes attention weights from Vision Transformer models to show
    which image regions the model focuses on. Supports ViT architectures from torchvision.

    Educational Note:
    - ViT models split images into patches and compute self-attention between patches
    - Attention maps show how strongly each patch attends to every other patch
    - The [CLS] token attention indicates which patches are most important for classification
    - Averaging across attention heads provides a robust visualization

    Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale", ICLR 2021.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True)
        >>> visualizer = ViTAttentionVisualizer(model, device="cuda")
        >>> attn_map = visualizer.generate_attention_map(image_tensor)
        >>> visualizer.save_overlay(image_tensor, attn_map, "attention.png")
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_idx: int = -1,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize ViT attention visualizer.

        Args:
            model: Vision Transformer model (expects torchvision ViT architecture)
            attention_layer_idx: Which encoder layer to visualize (-1 for last layer)
            device: Device to run computations on. If None, uses model's current device

        Raises:
            ValueError: If model is not a Vision Transformer architecture
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.attention_layer_idx = attention_layer_idx

        # Validate ViT architecture
        if not hasattr(model, "encoder"):
            raise ValueError(
                "Model does not appear to be a Vision Transformer. "
                "Expected 'encoder' attribute (torchvision ViT architecture)."
            )

        # Locate target attention layer
        self.encoder_layers = model.encoder.layers
        if abs(attention_layer_idx) > len(self.encoder_layers):
            raise ValueError(
                f"attention_layer_idx {attention_layer_idx} out of range. "
                f"Model has {len(self.encoder_layers)} encoder layers."
            )

        self.target_layer = self.encoder_layers[attention_layer_idx]
        self.attention_weights: List[torch.Tensor] = []
        self._hooks_registered = False

    def _register_hooks(self):
        """Register forward hook on attention layer to capture attention weights."""
        if self._hooks_registered:
            return

        def attention_hook(module, input, output):
            # ViT encoder layer outputs (hidden_state, optional attention_weights)
            # We need to hook into the self_attention module directly
            pass

        # Hook into the self-attention module's forward to capture attention weights
        def self_attn_hook(module, input, output):
            # torchvision ViT self_attention returns (output, attention_weights)
            # attention_weights shape: (B, num_heads, N, N) where N = num_patches + 1
            if isinstance(output, tuple) and len(output) == 2:
                self.attention_weights.append(output[1].detach())

        # Access self_attention module in the encoder layer
        if hasattr(self.target_layer, "self_attention"):
            self.handle = self.target_layer.self_attention.register_forward_hook(self_attn_hook)
            self._hooks_registered = True
        else:
            raise ValueError("Cannot locate self_attention module in encoder layer")

    def _remove_hooks(self):
        """Remove registered hooks."""
        if self._hooks_registered:
            self.handle.remove()
            self._hooks_registered = False

    def _clear_buffers(self):
        """Clear attention weight buffers."""
        self.attention_weights.clear()

    def generate_attention_map(
        self,
        x: torch.Tensor,
        use_cls_token: bool = True,
        head_reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Generate attention map from Vision Transformer.

        Args:
            x: Input tensor of shape (B, C, H, W) or (C, H, W)
            use_cls_token: If True, visualize [CLS] token attention to patches.
                          If False, average attention across all patches
            head_reduction: How to aggregate multi-head attention ("mean" or "max")

        Returns:
            Attention map tensor of shape (B, H, W) or (H, W), normalized to [0, 1]

        Educational Note:
        - [CLS] token attention shows which patches are most important for prediction
        - Averaging across all token attention provides a global importance map
        - Mean reduction averages attention heads; max takes the maximum
        """
        self.model.eval()
        single_image = x.ndim == 3
        if single_image:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        self._clear_buffers()
        self._register_hooks()

        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(x)

            # Extract attention weights
            if not self.attention_weights:
                logger.warning("No attention weights captured. Returning zeros.")
                return torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=self.device)

            attn = self.attention_weights[0]  # (B, num_heads, N, N)
            B, num_heads, N, _ = attn.shape

            # Reduce across attention heads
            if head_reduction == "mean":
                attn = attn.mean(dim=1)  # (B, N, N)
            elif head_reduction == "max":
                attn = attn.max(dim=1)[0]  # (B, N, N)
            else:
                raise ValueError(f"Unknown head_reduction: {head_reduction}")

            # Extract relevant attention scores
            if use_cls_token:
                # [CLS] token is typically the first token (index 0)
                # Get attention from [CLS] to all patches
                attn_map = attn[:, 0, 1:]  # (B, N-1) - exclude [CLS] to [CLS]
            else:
                # Average attention across all tokens (excluding self-attention)
                attn_map = attn[:, 1:, 1:].mean(dim=1)  # (B, N-1)

            # Determine patch grid size
            # ViT-B/16 uses 16x16 patches, so for 224x224 image: 14x14 patches
            num_patches = attn_map.shape[1]
            grid_size = int(np.sqrt(num_patches))

            # Reshape to 2D grid
            attn_map = attn_map.reshape(B, grid_size, grid_size)  # (B, h, w)

            # Upsample to input image size using bilinear interpolation
            attn_map = attn_map.unsqueeze(1)  # (B, 1, h, w)
            attn_map = torch.nn.functional.interpolate(
                attn_map,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            attn_map = attn_map.squeeze(1)  # (B, H, W)

            # Normalize to [0, 1] per image
            attn_min = attn_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            attn_max = attn_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)

            return attn_map[0] if single_image else attn_map

        finally:
            self._remove_hooks()
            self._clear_buffers()

    def save_overlay(
        self,
        x: torch.Tensor,
        attention_map: torch.Tensor,
        output_path: Union[str, Path],
        alpha: float = 0.35,
        colormap: str = "viridis"
    ):
        """
        Save attention map overlaid on original image.

        Args:
            x: Input image tensor of shape (C, H, W), normalized with ImageNet stats
            attention_map: Attention map tensor of shape (H, W), values in [0, 1]
            output_path: Path to save the blended image
            alpha: Blending factor for overlay (0=original, 1=attention map)
            colormap: Matplotlib colormap name for attention visualization

        Educational Note: Viridis colormap is perceptually uniform and colorblind-friendly
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Denormalize image (ImageNet stats)
        img = x.detach().cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

        # Convert attention map to RGB using colormap
        attn_np = attention_map.detach().cpu().numpy()
        attn_uint8 = np.uint8(255 * attn_np)

        # Apply colormap
        from matplotlib import cm as mpl_cm
        cmap = mpl_cm.get_cmap(colormap)
        attn_rgb = cmap(attn_uint8)[:, :, :3]  # Drop alpha channel

        # Create PIL images
        base = Image.fromarray(np.uint8(img * 255)).convert("RGB")
        attn_img = Image.fromarray(np.uint8(attn_rgb * 255)).convert("RGB")

        # Blend and save
        blended = Image.blend(base, attn_img, alpha=alpha)
        blended.save(output_path)

    def save_batch_overlays(
        self,
        x: torch.Tensor,
        attention_maps: torch.Tensor,
        output_dir: Union[str, Path],
        metas: Optional[List[Dict[str, Any]]] = None,
        prefix: str = "vit_attention",
        alpha: float = 0.35,
        colormap: str = "viridis"
    ) -> int:
        """
        Save attention map overlays for a batch of images.

        Args:
            x: Batch of input images, shape (B, C, H, W)
            attention_maps: Batch of attention maps, shape (B, H, W)
            output_dir: Directory to save overlays
            metas: Optional list of metadata dicts with 'accession' or other identifiers
            prefix: Filename prefix for saved images
            alpha: Blending factor for overlay
            colormap: Matplotlib colormap name

        Returns:
            Number of images successfully saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i in range(x.shape[0]):
            try:
                identifier = f"sample_{i}"
                if metas and i < len(metas):
                    identifier = metas[i].get("accession", identifier)

                fname = output_dir / f"{prefix}_{i}_{identifier}.png"
                self.save_overlay(x[i], attention_maps[i], fname, alpha=alpha, colormap=colormap)
                saved += 1
            except Exception as exc:
                logger.warning(f"Failed to save attention overlay {i}: {exc}")

        return saved

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


def generate_explanations_batch(
    images: List[torch.Tensor],
    model: nn.Module,
    explainer_type: str = "gradcam",
    target_classes: Optional[List[Optional[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 8,
    **explainer_kwargs
) -> List[Optional[torch.Tensor]]:
    """
    Generate explanations for a batch of images.

    Educational Note: Batch processing significantly improves efficiency
    when generating explanations for large numbers of images.

    Args:
        images: List of image tensors, each of shape (C, H, W)
        model: PyTorch model to explain
        explainer_type: Type of explainer ("gradcam" or "vit_attention")
        target_classes: Optional list of target classes for each image
        device: Device to run computations on
        batch_size: Number of images to process at once
        **explainer_kwargs: Additional arguments passed to explainer constructor

    Returns:
        List[Optional[torch.Tensor]]: List of heatmap tensors (H, W) or None on failure

    Example:
        >>> images = [load_image("img1.png"), load_image("img2.png")]
        >>> heatmaps = generate_explanations_batch(images, model)
        >>> # Process successful heatmaps
        >>> for i, heatmap in enumerate(heatmaps):
        ...     if heatmap is not None:
        ...         save_heatmap(heatmap, f"output_{i}.png")
    """
    if not images:
        return []

    # Initialize explainer
    if explainer_type == "gradcam":
        explainer = GradCAMExplainer(model, device=device, **explainer_kwargs)
    elif explainer_type == "vit_attention":
        explainer = ViTAttentionVisualizer(model, device=device, **explainer_kwargs)
    else:
        raise ValueError(f"Unknown explainer_type: {explainer_type}. Use 'gradcam' or 'vit_attention'")

    # Prepare target classes
    if target_classes is None:
        target_classes = [None] * len(images)

    # Validate inputs
    if len(target_classes) != len(images):
        raise ValueError(f"target_classes length ({len(target_classes)}) must match images length ({len(images)})")

    all_heatmaps = []

    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_targets = target_classes[i:i + batch_size]

        try:
            # Stack images into batch tensor
            batch_tensor = torch.stack(batch_images)

            # Generate explanations
            if explainer_type == "gradcam":
                # Convert target classes to tensor if needed
                if all(t is None for t in batch_targets):
                    batch_heatmaps = explainer.generate_heatmap(batch_tensor, target_class=None)
                else:
                    # Use first non-None target or predicted class
                    target_tensor = torch.tensor(
                        [t if t is not None else -1 for t in batch_targets],
                        device=explainer.device
                    )
                    batch_heatmaps = explainer.generate_heatmap(
                        batch_tensor,
                        target_class=target_tensor if not (target_tensor == -1).all() else None
                    )
            else:  # vit_attention
                batch_heatmaps = explainer.generate_attention_map(batch_tensor)

            # Split batch results
            for j in range(len(batch_images)):
                all_heatmaps.append(batch_heatmaps[j])

        except Exception as exc:
            logger.warning(f"Failed to generate explanations for batch {i//batch_size}: {exc}")
            # Append None for failed batch
            for _ in range(len(batch_images)):
                all_heatmaps.append(None)

    successful_count = sum(1 for hm in all_heatmaps if hm is not None)
    logger.info(f"Successfully generated {successful_count}/{len(images)} explanations")

    return all_heatmaps


def export_explanations_report(
    images: List[torch.Tensor],
    model: nn.Module,
    output_dir: Union[str, Path] = "explanations_report",
    metas: Optional[List[Dict[str, Any]]] = None,
    target_classes: Optional[List[Optional[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 8,
    alpha: float = 0.35
) -> Dict[str, Any]:
    """
    Generate a comprehensive explainability report for a set of images.

    Creates both GradCAM and attention-based explanations (if model supports)
    and saves them to the output directory with metadata.

    Educational Note: Comparing multiple explainability techniques provides
    more robust insights into model behavior than relying on a single method.

    Args:
        images: List of image tensors, each of shape (C, H, W)
        model: PyTorch model to explain
        output_dir: Output directory for report
        metas: Optional list of metadata dicts with identifiers
        target_classes: Optional list of target classes for each image
        device: Device to run computations on
        batch_size: Number of images to process at once
        alpha: Blending factor for overlays

    Returns:
        Dict containing:
            - 'output_dir': Path to report directory
            - 'num_images': Total number of images processed
            - 'num_successful': Number of successful explanations
            - 'gradcam_dir': Path to GradCAM outputs
            - 'attention_dir': Path to attention outputs (if applicable)

    Example:
        >>> images = [load_image(f"img{i}.png") for i in range(10)]
        >>> metas = [{"accession": f"ACC{i}"} for i in range(10)]
        >>> report = export_explanations_report(images, model, metas=metas)
        >>> print(f"Report saved to {report['output_dir']}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        logger.warning("No images provided for explanation report")
        return {
            'output_dir': str(output_dir),
            'num_images': 0,
            'num_successful': 0
        }

    report = {
        'output_dir': str(output_dir),
        'num_images': len(images),
        'num_successful': 0
    }

    # 1. Generate GradCAM explanations
    logger.info("Generating GradCAM explanations...")
    try:
        gradcam_explainer = GradCAMExplainer(model, device=device)
        gradcam_heatmaps = generate_explanations_batch(
            images,
            model,
            explainer_type="gradcam",
            target_classes=target_classes,
            device=device,
            batch_size=batch_size
        )

        # Save GradCAM overlays
        gradcam_dir = output_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)

        batch_tensor = torch.stack(images)
        valid_heatmaps = torch.stack([hm if hm is not None else torch.zeros_like(images[0][0])
                                      for hm in gradcam_heatmaps])

        saved = gradcam_explainer.save_batch_overlays(
            batch_tensor,
            valid_heatmaps,
            gradcam_dir,
            metas=metas,
            alpha=alpha
        )

        report['gradcam_dir'] = str(gradcam_dir)
        report['num_successful'] += saved
        logger.info(f"Saved {saved} GradCAM overlays to {gradcam_dir}")

    except Exception as exc:
        logger.warning(f"Failed to generate GradCAM explanations: {exc}")

    # 2. Attempt ViT attention visualization (if model supports)
    try:
        if hasattr(model, "encoder"):
            logger.info("Generating ViT attention visualizations...")
            vit_explainer = ViTAttentionVisualizer(model, device=device)
            attention_maps = generate_explanations_batch(
                images,
                model,
                explainer_type="vit_attention",
                device=device,
                batch_size=batch_size
            )

            # Save attention overlays
            attention_dir = output_dir / "vit_attention"
            attention_dir.mkdir(exist_ok=True)

            batch_tensor = torch.stack(images)
            valid_maps = torch.stack([am if am is not None else torch.zeros_like(images[0][0])
                                     for am in attention_maps])

            saved = vit_explainer.save_batch_overlays(
                batch_tensor,
                valid_maps,
                attention_dir,
                metas=metas,
                alpha=alpha
            )

            report['attention_dir'] = str(attention_dir)
            logger.info(f"Saved {saved} attention overlays to {attention_dir}")

    except Exception as exc:
        logger.info(f"ViT attention visualization not available: {exc}")

    # 3. Generate summary metadata
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Explainability Report Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images: {report['num_images']}\n")
        f.write(f"Successful GradCAM: {report['num_successful']}\n")
        if 'attention_dir' in report:
            f.write(f"ViT attention maps generated: Yes\n")
        else:
            f.write(f"ViT attention maps generated: No\n")
        f.write(f"\nOutput directory: {output_dir}\n")

        if metas:
            f.write("\nProcessed images:\n")
            for i, meta in enumerate(metas):
                accession = meta.get("accession", f"sample_{i}")
                f.write(f"  {i+1}. {accession}\n")

    report['summary_path'] = str(summary_path)
    logger.info(f"Explainability report saved to {output_dir}")

    return report
