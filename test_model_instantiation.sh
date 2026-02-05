#!/usr/bin/env bash
# Test script for model instantiation verification

set -e

export PYTHONPATH=./src

echo "Testing EfficientNet model instantiation..."
python -c "from mammography.models.nets import create_model; import torch; model = create_model('efficientnet_b0', num_classes=4); print('EfficientNet OK')"

echo "Testing ResNet50 model instantiation..."
python -c "from mammography.models.nets import create_model; import torch; model = create_model('resnet50', num_classes=4); print('ResNet50 OK')"

echo "Testing ViT model instantiation..."
python -c "from mammography.models.nets import create_model; import torch; model = create_model('vit_b_16', num_classes=4); print('ViT OK')"

echo "Model OK"
