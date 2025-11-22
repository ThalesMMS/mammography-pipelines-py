#
# nets.py
# mammography-pipelines-py
#
# Defines EfficientNetB0/ResNet50 classifiers with optional tabular fusion and fine-tuning controls.
#
# Thales Matheus Mendonça Santos - November 2025
#
import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights

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
) -> nn.Module:
    """Build EfficientNetB0/ResNet50 with a customizable head and optional feature fusion."""
    
    if arch == "efficientnet_b0":
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m = EfficientNetWithFusion(base, num_classes=num_classes, extra_feature_dim=extra_feature_dim)
        
        # Configura congelamento inicial
        for p in m.backbone.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True
            
        if unfreeze_last_block:
            # EfficientNetB0: últimos 2 blocos
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
            # TODO: Implementar fusão para ResNet se necessário
            raise NotImplementedError("Fusion not implemented for ResNet50 yet.")
            
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
