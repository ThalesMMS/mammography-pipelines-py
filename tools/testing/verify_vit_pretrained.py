#!/usr/bin/env python3
"""Verify ViT model can be instantiated with pretrained weights."""

from mammography.models.nets import build_model
import torch

# Build ViT model with pretrained weights
m = build_model('vit_b_16', num_classes=4, pretrained=True)

# Test forward pass
x = torch.randn(2, 3, 224, 224)
y = m(x)

print(f'Output shape: {y.shape}')
assert y.shape == (2, 4), f"Expected shape (2, 4), got {y.shape}"
print('SUCCESS: ViT model with pretrained weights working')
