#
# nets.py
# mammography-pipelines
#
# Defines EfficientNetB0/ResNet50/ViT classifiers with optional tabular fusion and fine-tuning controls.
#
# Thales Matheus Mendonça Santos - November 2025
#
import torch
import torch.nn as nn
import warnings
from typing import Optional
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None  # type: ignore
    # Note: DeiT models will raise RuntimeError if attempted without timm installed


def _load_with_fallback(factory, weights, arch_name: str) -> nn.Module:
    """Load a torchvision model, falling back to random weights if pretrained download fails."""
    if weights is None:
        return factory(weights=None)
    try:
        return factory(weights=weights)
    except Exception as exc:
        warnings.warn(
            f"Failed to load pretrained weights for {arch_name}; using random init. Error: {exc}",
            RuntimeWarning,
        )
        return factory(weights=None)

class EfficientNetWithFusion(nn.Module):
    """Wrapper that optionally concatenates tabular embeddings before classification."""

    def __init__(self, base_model: nn.Module, num_classes: int, extra_feature_dim: int = 0):
        super().__init__()
        self.backbone = base_model
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.extra_feature_dim = int(extra_feature_dim or 0)

        # Safely extract classifier input features
        # EfficientNet typically has Sequential(Dropout, Linear) but check to be safe
        if isinstance(base_model.classifier, nn.Sequential) and len(base_model.classifier) > 1:
            # Standard structure: Sequential(Dropout, Linear)
            for module in base_model.classifier:
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    break
            else:
                raise ValueError("Could not find Linear layer in EfficientNet classifier")
        elif isinstance(base_model.classifier, nn.Linear):
            # Simple Linear classifier
            in_features = base_model.classifier.in_features
        else:
            raise ValueError(f"Unexpected EfficientNet classifier structure: {type(base_model.classifier)}")

        # Replace the default classifier with identity so we can attach our own head.
        base_model.classifier = nn.Identity()

        fusion_in = in_features + self.extra_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(fusion_in, num_classes),
        )

    def forward(self, x: torch.Tensor, extra_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if self.extra_feature_dim > 0:
            if extra_features is None:
                raise ValueError("Embeddings habilitados, mas extra_features=None foi recebido.")
            if extra_features.shape[0] != x.shape[0]:
                raise ValueError(
                    f"extra_features batch mismatch: CNN batch={x.shape[0]} vs embeddings batch={extra_features.shape[0]}"
                )
            # Validate feature dimension matches expected size
            if extra_features.shape[1] != self.extra_feature_dim:
                raise ValueError(
                    f"extra_features dimension mismatch: expected {self.extra_feature_dim}, got {extra_features.shape[1]}"
                )
            if extra_features.dtype != x.dtype:
                extra_features = extra_features.to(dtype=x.dtype)
            x = torch.cat([x, extra_features], dim=1)
        return self.classifier(x)

class ResNetWithFusion(nn.Module):
    """Wrapper that optionally concatenates tabular embeddings before classification."""

    def __init__(self, base_model: nn.Module, num_classes: int, extra_feature_dim: int = 0):
        super().__init__()
        self.backbone = base_model
        self.extra_feature_dim = int(extra_feature_dim or 0)

        in_features = base_model.fc.in_features
        # Replace the default fc with identity so we can attach our own head.
        base_model.fc = nn.Identity()

        fusion_in = in_features + self.extra_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(fusion_in, num_classes),
        )

    def forward(self, x: torch.Tensor, extra_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if self.extra_feature_dim > 0:
            if extra_features is None:
                raise ValueError("Embeddings habilitados, mas extra_features=None foi recebido.")
            if extra_features.shape[0] != x.shape[0]:
                raise ValueError(
                    f"extra_features batch mismatch: CNN batch={x.shape[0]} vs embeddings batch={extra_features.shape[0]}"
                )
            # Validate feature dimension matches expected size
            if extra_features.shape[1] != self.extra_feature_dim:
                raise ValueError(
                    f"extra_features dimension mismatch: expected {self.extra_feature_dim}, got {extra_features.shape[1]}"
                )
            if extra_features.dtype != x.dtype:
                extra_features = extra_features.to(dtype=x.dtype)
            x = torch.cat([x, extra_features], dim=1)
        return self.classifier(x)

class ViTWithFusion(nn.Module):
    """Wrapper that optionally concatenates tabular embeddings before classification."""

    def __init__(self, base_model: nn.Module, num_classes: int, extra_feature_dim: int = 0):
        super().__init__()
        self.backbone = base_model
        self.extra_feature_dim = int(extra_feature_dim or 0)

        in_features = base_model.heads.head.in_features
        # Replace the default head with identity so we can attach our own head.
        base_model.heads = nn.Identity()

        fusion_in = in_features + self.extra_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(fusion_in, num_classes),
        )

    def forward(self, x: torch.Tensor, extra_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features from ViT backbone
        x = self.backbone(x)
        if self.extra_feature_dim > 0:
            if extra_features is None:
                raise ValueError("Embeddings habilitados, mas extra_features=None foi recebido.")
            if extra_features.shape[0] != x.shape[0]:
                raise ValueError(
                    f"extra_features batch mismatch: ViT batch={x.shape[0]} vs embeddings batch={extra_features.shape[0]}"
                )
            # Validate feature dimension matches expected size
            if extra_features.shape[1] != self.extra_feature_dim:
                raise ValueError(
                    f"extra_features dimension mismatch: expected {self.extra_feature_dim}, got {extra_features.shape[1]}"
                )
            if extra_features.dtype != x.dtype:
                extra_features = extra_features.to(dtype=x.dtype)
            x = torch.cat([x, extra_features], dim=1)
        return self.classifier(x)

class DeiTWithFusion(nn.Module):
    """Wrapper that optionally concatenates tabular embeddings before classification (for timm DeiT models)."""

    def __init__(self, base_model: nn.Module, num_classes: int, extra_feature_dim: int = 0):
        super().__init__()
        self.backbone = base_model
        self.extra_feature_dim = int(extra_feature_dim or 0)

        # timm DeiT models have a 'head' attribute
        in_features = base_model.head.in_features if hasattr(base_model, 'head') else base_model.num_features
        # Replace the default head with identity so we can attach our own head.
        base_model.head = nn.Identity()

        fusion_in = in_features + self.extra_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(fusion_in, num_classes),
        )

    def forward(self, x: torch.Tensor, extra_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features from DeiT backbone using forward_features
        x = self.backbone.forward_features(x)
        # timm models return patch embeddings; take the class token (first token)
        if x.dim() == 3:  # (batch, num_patches, embed_dim)
            x = x[:, 0]  # Take class token

        if self.extra_feature_dim > 0:
            if extra_features is None:
                raise ValueError("Embeddings habilitados, mas extra_features=None foi recebido.")
            if extra_features.shape[0] != x.shape[0]:
                raise ValueError(
                    f"extra_features batch mismatch: DeiT batch={x.shape[0]} vs embeddings batch={extra_features.shape[0]}"
                )
            # Validate feature dimension matches expected size
            if extra_features.shape[1] != self.extra_feature_dim:
                raise ValueError(
                    f"extra_features dimension mismatch: expected {self.extra_feature_dim}, got {extra_features.shape[1]}"
                )
            if extra_features.dtype != x.dtype:
                extra_features = extra_features.to(dtype=x.dtype)
            x = torch.cat([x, extra_features], dim=1)
        return self.classifier(x)

def build_model(
    arch: str = "efficientnet_b0",
    num_classes: int = 4,
    train_backbone: bool = False,
    unfreeze_last_block: bool = True,
    extra_feature_dim: int = 0,
    pretrained: bool = True,
) -> nn.Module:
    """Build EfficientNetB0/ResNet50/ViT with a customizable head and optional feature fusion."""
    
    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = _load_with_fallback(efficientnet_b0, weights, "efficientnet_b0")
        m = EfficientNetWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)
        
        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True
            
        if unfreeze_last_block:
            # EfficientNetB0: last 2 blocks
            num_blocks = len(m.features)
            for i in range(max(0, num_blocks - 2), num_blocks):
                for p in m.features[i].parameters():
                    p.requires_grad = True
                    
        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True
                
        return m

    elif arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = _load_with_fallback(resnet50, weights, "resnet50")
        m = ResNetWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            for p in m.backbone.layer4.parameters():
                p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    elif arch == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = _load_with_fallback(vit_b_16, weights, "vit_b_16")
        m = ViTWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            # ViT: unfreeze last encoder block
            if hasattr(m.backbone, 'encoder') and hasattr(m.backbone.encoder, 'layers'):
                for p in m.backbone.encoder.layers[-1].parameters():
                    p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    elif arch == "vit_b_32":
        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        base = _load_with_fallback(vit_b_32, weights, "vit_b_32")
        m = ViTWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            # ViT: unfreeze last encoder block
            if hasattr(m.backbone, 'encoder') and hasattr(m.backbone.encoder, 'layers'):
                for p in m.backbone.encoder.layers[-1].parameters():
                    p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    elif arch == "vit_l_16":
        weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = _load_with_fallback(vit_l_16, weights, "vit_l_16")
        m = ViTWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            # ViT: unfreeze last encoder block
            if hasattr(m.backbone, 'encoder') and hasattr(m.backbone.encoder, 'layers'):
                for p in m.backbone.encoder.layers[-1].parameters():
                    p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    elif arch == "deit_small":
        if not TIMM_AVAILABLE:
            raise RuntimeError(
                "timm is required for DeiT models but is not installed. "
                "Install with: pip install timm"
            )
        try:
            base = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
        except Exception as exc:
            if pretrained:
                warnings.warn(
                    f"Failed to load pretrained weights for deit_small; using random init. Error: {exc}",
                    RuntimeWarning,
                )
                base = timm.create_model('deit_small_patch16_224', pretrained=False)
            else:
                raise
        m = DeiTWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            # DeiT: unfreeze last transformer block
            if hasattr(m.backbone, 'blocks') and len(m.backbone.blocks) > 0:
                for p in m.backbone.blocks[-1].parameters():
                    p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    elif arch == "deit_base":
        if not TIMM_AVAILABLE:
            raise RuntimeError(
                "timm is required for DeiT models but is not installed. "
                "Install with: pip install timm"
            )
        try:
            base = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        except Exception as exc:
            if pretrained:
                warnings.warn(
                    f"Failed to load pretrained weights for deit_base; using random init. Error: {exc}",
                    RuntimeWarning,
                )
                base = timm.create_model('deit_base_patch16_224', pretrained=False)
            else:
                raise
        m = DeiTWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)

        # Configure initial freezing
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True

        if unfreeze_last_block:
            # DeiT: unfreeze last transformer block
            if hasattr(m.backbone, 'blocks') and len(m.backbone.blocks) > 0:
                for p in m.backbone.blocks[-1].parameters():
                    p.requires_grad = True

        if train_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = True

        return m

    else:
        raise ValueError(f"Architecture {arch} not supported.")

def create_model(
    arch: str = "efficientnet_b0",
    num_classes: int = 4,
    train_backbone: bool = False,
    unfreeze_last_block: bool = True,
    extra_feature_dim: int = 0,
    pretrained: bool = True,
) -> nn.Module:
    """Alias for build_model for backwards compatibility and testing."""
    return build_model(
        arch=arch,
        num_classes=num_classes,
        train_backbone=train_backbone,
        unfreeze_last_block=unfreeze_last_block,
        extra_feature_dim=extra_feature_dim,
        pretrained=pretrained,
    )
