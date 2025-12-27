#
# nets.py
# mammography-pipelines
#
# Defines EfficientNetB0/ResNet50 classifiers with optional tabular fusion and fine-tuning controls.
#
# Thales Matheus Mendonça Santos - November 2025
#
import torch
import torch.nn as nn
import warnings
from typing import Optional
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights


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

        in_features = base_model.classifier[1].in_features  # Dropout + Linear
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
    """Build EfficientNetB0/ResNet50 with a customizable head and optional feature fusion."""
    
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
        if extra_feature_dim > 0:
            # TODO: Implement fusion for ResNet if needed
            raise NotImplementedError("Fusion not implemented for ResNet50 yet.")
            
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = _load_with_fallback(resnet50, weights, "resnet50")
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
            
        if unfreeze_last_block:
            for p in m.layer4.parameters():
                p.requires_grad = True
                
        if train_backbone:
            for p in m.parameters():
                p.requires_grad = True
        return m
        
    else:
        raise ValueError(f"Architecture {arch} not supported.")
