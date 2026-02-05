#!/usr/bin/env python
"""Test script for model instantiation."""
from mammography.models.nets import create_model
import torch

# Test EfficientNet
model_eff = create_model('efficientnet_b0', num_classes=4)
print(f"EfficientNet: {type(model_eff).__name__}")

# Test ResNet
model_res = create_model('resnet50', num_classes=4)
print(f"ResNet50: {type(model_res).__name__}")

# Test ViT
model_vit = create_model('vit_b_16', num_classes=4)
print(f"ViT B/16: {type(model_vit).__name__}")

print("Model OK")
