from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from torch import nn

from mammography.vis.explainability import (
    GradCAMExplainer,
    ViTAttentionVisualizer,
    export_explanations_report,
    generate_explanations_batch,
)


class MockResNetModel(nn.Module):
    """Mock ResNet-like model for GradCAM testing."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MockEfficientNetModel(nn.Module):
    """Mock EfficientNet-like model for GradCAM testing."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MockViTModel(nn.Module):
    """Mock Vision Transformer model for attention testing."""

    def __init__(self, num_classes: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.class_token = nn.Parameter(torch.randn(1, 1, 768))

        # Create mock encoder with attention layers
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([
            self._create_mock_encoder_layer() for _ in range(num_layers)
        ])
        self.encoder.pos_embedding = nn.Parameter(torch.randn(1, 197, 768))

        self.heads = nn.Module()
        self.heads.head = nn.Linear(768, num_classes)

    def _create_mock_encoder_layer(self) -> nn.Module:
        layer = nn.Module()
        layer.self_attention = MockSelfAttention()
        layer.mlp = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768)
        )
        layer.ln_1 = nn.LayerNorm(768)
        layer.ln_2 = nn.LayerNorm(768)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embedding
        x = x + self.encoder.pos_embedding[:, :x.shape[1], :]

        # Pass through encoder layers
        for layer in self.encoder.layers:
            # Self-attention with residual
            attn_out = layer.self_attention(layer.ln_1(x))
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = x + attn_out

            # MLP with residual
            x = x + layer.mlp(layer.ln_2(x))

        # Classification head (use class token)
        cls_token = x[:, 0]
        return self.heads.head(cls_token)


class MockSelfAttention(nn.Module):
    """Mock self-attention module that returns attention weights."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        # Generate mock attention weights
        # Shape: (B, num_heads, N, N)
        attn_weights = torch.softmax(
            torch.randn(B, self.num_heads, N, N, device=x.device),
            dim=-1
        )

        # Simple linear projection for output
        out = self.proj(x)

        return out, attn_weights


def test_gradcam_explainer_resnet_single_image() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(3, 224, 224)
    heatmap = explainer.generate_heatmap(x)

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_explainer_resnet_batch() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(4, 3, 224, 224)
    heatmap = explainer.generate_heatmap(x)

    assert heatmap.shape == (4, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_explainer_efficientnet() -> None:
    model = MockEfficientNetModel(num_classes=2)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224)
    heatmap = explainer.generate_heatmap(x)

    assert heatmap.shape == (2, 224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_explainer_with_target_class() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224)

    # Test with integer target class
    heatmap1 = explainer.generate_heatmap(x, target_class=1)
    assert heatmap1.shape == (2, 224, 224)

    # Test with tensor target classes
    target_classes = torch.tensor([0, 2])
    heatmap2 = explainer.generate_heatmap(x, target_class=target_classes)
    assert heatmap2.shape == (2, 224, 224)


def test_gradcam_explainer_invalid_model() -> None:
    # Model without layer4 or features should raise error
    class InvalidModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
            self.fc = nn.Linear(64, 4)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    model = InvalidModel()

    with pytest.raises(ValueError, match="Cannot auto-detect target layer"):
        GradCAMExplainer(model, device="cpu")


def test_gradcam_save_overlay(tmp_path) -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    # ImageNet normalized image
    x = torch.randn(3, 224, 224)
    heatmap = explainer.generate_heatmap(x)

    output_path = tmp_path / "gradcam_overlay.png"
    explainer.save_overlay(x, heatmap, output_path, alpha=0.35, colormap="jet")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_gradcam_save_batch_overlays(tmp_path) -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(3, 3, 224, 224)
    heatmaps = explainer.generate_heatmap(x)

    output_dir = tmp_path / "gradcam_batch"
    metas = [{"accession": f"ACC{i}"} for i in range(3)]

    saved = explainer.save_batch_overlays(
        x, heatmaps, output_dir, metas=metas, prefix="gradcam", alpha=0.35
    )

    assert saved == 3
    assert output_dir.exists()
    assert len(list(output_dir.glob("gradcam_*.png"))) == 3


def test_vit_attention_visualizer_single_image() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, attention_layer_idx=-1, device="cpu")

    x = torch.randn(3, 224, 224)
    attn_map = visualizer.generate_attention_map(x, use_cls_token=True)

    assert attn_map.shape == (224, 224)
    assert attn_map.min() >= 0.0
    assert attn_map.max() <= 1.0


def test_vit_attention_visualizer_batch() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, attention_layer_idx=-1, device="cpu")

    x = torch.randn(4, 3, 224, 224)
    attn_map = visualizer.generate_attention_map(x, use_cls_token=True)

    assert attn_map.shape == (4, 224, 224)
    assert attn_map.min() >= 0.0
    assert attn_map.max() <= 1.0


def test_vit_attention_visualizer_head_reduction() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224)

    # Test mean reduction
    attn_mean = visualizer.generate_attention_map(x, head_reduction="mean")
    assert attn_mean.shape == (2, 224, 224)

    # Test max reduction
    attn_max = visualizer.generate_attention_map(x, head_reduction="max")
    assert attn_max.shape == (2, 224, 224)


def test_vit_attention_visualizer_cls_vs_avg() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224)

    # Test CLS token attention
    attn_cls = visualizer.generate_attention_map(x, use_cls_token=True)
    assert attn_cls.shape == (2, 224, 224)

    # Test average attention
    attn_avg = visualizer.generate_attention_map(x, use_cls_token=False)
    assert attn_avg.shape == (2, 224, 224)


def test_vit_attention_visualizer_invalid_model() -> None:
    # Model without encoder should raise error
    model = MockResNetModel(num_classes=4)

    with pytest.raises(ValueError, match="does not appear to be a Vision Transformer"):
        ViTAttentionVisualizer(model, device="cpu")


def test_vit_attention_visualizer_invalid_layer_idx() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)

    with pytest.raises(ValueError, match="attention_layer_idx .* out of range"):
        ViTAttentionVisualizer(model, attention_layer_idx=10, device="cpu")


def test_vit_attention_save_overlay(tmp_path) -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, device="cpu")

    x = torch.randn(3, 224, 224)
    attn_map = visualizer.generate_attention_map(x)

    output_path = tmp_path / "attention_overlay.png"
    visualizer.save_overlay(x, attn_map, output_path, alpha=0.35, colormap="viridis")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_vit_attention_save_batch_overlays(tmp_path) -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, device="cpu")

    x = torch.randn(3, 3, 224, 224)
    attn_maps = visualizer.generate_attention_map(x)

    output_dir = tmp_path / "attention_batch"
    metas = [{"accession": f"ACC{i}"} for i in range(3)]

    saved = visualizer.save_batch_overlays(
        x, attn_maps, output_dir, metas=metas, prefix="vit_attention", alpha=0.35
    )

    assert saved == 3
    assert output_dir.exists()
    assert len(list(output_dir.glob("vit_attention_*.png"))) == 3


def test_generate_explanations_batch_gradcam() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(5)]

    heatmaps = generate_explanations_batch(
        images,
        model,
        explainer_type="gradcam",
        device="cpu",
        batch_size=2
    )

    assert len(heatmaps) == 5
    assert all(hm is not None for hm in heatmaps)
    assert all(hm.shape == (224, 224) for hm in heatmaps)


def test_generate_explanations_batch_vit_attention() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(5)]

    attn_maps = generate_explanations_batch(
        images,
        model,
        explainer_type="vit_attention",
        device="cpu",
        batch_size=2
    )

    assert len(attn_maps) == 5
    assert all(am is not None for am in attn_maps)
    assert all(am.shape == (224, 224) for am in attn_maps)


def test_generate_explanations_batch_with_target_classes() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(3)]
    target_classes = [0, 1, 2]

    heatmaps = generate_explanations_batch(
        images,
        model,
        explainer_type="gradcam",
        target_classes=target_classes,
        device="cpu",
        batch_size=2
    )

    assert len(heatmaps) == 3
    assert all(hm is not None for hm in heatmaps)


def test_generate_explanations_batch_empty_list() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    heatmaps = generate_explanations_batch(
        [],
        model,
        explainer_type="gradcam",
        device="cpu"
    )

    assert heatmaps == []


def test_generate_explanations_batch_invalid_explainer_type() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    images = [torch.randn(3, 224, 224)]

    with pytest.raises(ValueError, match="Unknown explainer_type"):
        generate_explanations_batch(
            images,
            model,
            explainer_type="invalid_type",
            device="cpu"
        )


def test_generate_explanations_batch_target_classes_mismatch() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(3)]
    target_classes = [0, 1]  # Wrong length

    with pytest.raises(ValueError, match="target_classes length .* must match images length"):
        generate_explanations_batch(
            images,
            model,
            explainer_type="gradcam",
            target_classes=target_classes,
            device="cpu"
        )


def test_export_explanations_report_empty_images(tmp_path) -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    output_dir = tmp_path / "empty_report"

    report = export_explanations_report(
        [],
        model,
        output_dir=output_dir,
        device="cpu"
    )

    assert report['num_images'] == 0
    assert report['num_successful'] == 0
    assert output_dir.exists()


def test_export_explanations_report_resnet(tmp_path) -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(3)]
    metas = [{"accession": f"ACC{i}"} for i in range(3)]

    output_dir = tmp_path / "resnet_report"

    report = export_explanations_report(
        images,
        model,
        output_dir=output_dir,
        metas=metas,
        device="cpu",
        batch_size=2,
        alpha=0.35
    )

    assert report['num_images'] == 3
    assert report['num_successful'] > 0
    assert output_dir.exists()
    assert 'gradcam_dir' in report
    assert (output_dir / "gradcam").exists()
    assert (output_dir / "summary.txt").exists()


def test_export_explanations_report_vit(tmp_path) -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    images = [torch.randn(3, 224, 224) for _ in range(3)]
    metas = [{"accession": f"ACC{i}"} for i in range(3)]

    output_dir = tmp_path / "vit_report"

    report = export_explanations_report(
        images,
        model,
        output_dir=output_dir,
        metas=metas,
        device="cpu",
        batch_size=2,
        alpha=0.35
    )

    assert report['num_images'] == 3
    assert output_dir.exists()

    # Should have both GradCAM and attention visualizations
    # Note: GradCAM might fail on ViT but attention should work
    assert (output_dir / "summary.txt").exists()

    # Check summary file content
    summary_content = (output_dir / "summary.txt").read_text()
    assert "Explainability Report Summary" in summary_content
    assert "Total images: 3" in summary_content


def test_gradcam_gradient_flow() -> None:
    model = MockResNetModel(num_classes=4)
    model.eval()

    explainer = GradCAMExplainer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224, requires_grad=True)

    # Generate heatmap and verify gradients can flow
    heatmap = explainer.generate_heatmap(x, target_class=1)

    assert heatmap.shape == (2, 224, 224)
    # Gradients should not accumulate on input since we use eval mode
    # This tests that the explainer doesn't break gradient computation


def test_vit_attention_no_gradient_accumulation() -> None:
    model = MockViTModel(num_classes=4, num_layers=2)
    model.eval()

    visualizer = ViTAttentionVisualizer(model, device="cpu")

    x = torch.randn(2, 3, 224, 224, requires_grad=True)

    # Generate attention map (should use no_grad internally)
    attn_map = visualizer.generate_attention_map(x)

    assert attn_map.shape == (2, 224, 224)
    # No gradients should be tracked for attention visualization
    assert not attn_map.requires_grad
