#!/usr/bin/env bash
#
# verify_inference.sh
# Verification script for subtask-4-3: Execute inference command on checkpoint
#
# This script verifies that the inference command can run successfully
# with a trained checkpoint on sample data.

set -e

echo "=== Subtask 4-3: Inference Command Verification ==="
echo ""

# Check if checkpoint exists
if [ ! -f "outputs/best_model.pt" ]; then
    echo "ERROR: Checkpoint not found at outputs/best_model.pt"
    exit 1
fi
echo "✓ Checkpoint found: outputs/best_model.pt"

# Check if input data exists
if [ ! -d "mamografias/DleftCC" ]; then
    echo "ERROR: Input directory not found: mamografias/DleftCC"
    exit 1
fi
echo "✓ Input directory found: mamografias/DleftCC"

# Run inference command
echo ""
echo "Running inference command..."
echo "Command: python -m mammography.cli inference --checkpoint outputs/best_model.pt --input mamografias/DleftCC --batch-size 4"
echo ""

python -m mammography.cli inference \
    --checkpoint outputs/best_model.pt \
    --input mamografias/DleftCC \
    --batch-size 4

echo ""
echo "=== Inference completed successfully ==="
echo ""
echo "Note: The inference.py file is already correctly implemented with:"
echo "  - Checkpoint loading with model state_dict"
echo "  - Support for both image and DICOM inputs"
echo "  - Batch processing with DataLoader"
echo "  - Automatic mixed precision (AMP) support"
echo "  - CSV output with predictions and probabilities"
echo ""
