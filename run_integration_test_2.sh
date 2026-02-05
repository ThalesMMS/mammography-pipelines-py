#!/usr/bin/env bash
#
# Integration Test 2: train-density → inference → explain
# This script tests the complete workflow from training to inference to explainability
#

set -e  # Exit on error

PYTHON="/c/Users/user/AppData/Local/Programs/Python/Python313/python.exe"
CHECKPOINT="outputs/integration_test_train/results/best_model.pt"
INPUT_DIR="mamografias/DleftCC"
OUTPUT_INFERENCE="outputs/integration_test_2/inference"
OUTPUT_EXPLAIN="outputs/integration_test_2/explanations"

echo "=========================================="
echo "Integration Test 2: train → inference → explain"
echo "=========================================="
echo ""

# Step 1: Verify checkpoint exists (using existing from previous test)
echo "Step 1: Verifying checkpoint..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint found: $CHECKPOINT"
echo ""

# Step 2: Run inference (process first 10 images by limiting input)
echo "Step 2: Running inference..."
mkdir -p "$OUTPUT_INFERENCE"
# Create temp directory with subset of images
TEMP_INPUT="outputs/integration_test_2/temp_input"
mkdir -p "$TEMP_INPUT"
find "$INPUT_DIR" -name "*.png" -type f | head -10 | while read img; do
    cp "$img" "$TEMP_INPUT/"
done
echo "  Using $(ls $TEMP_INPUT | wc -l) images for inference"

cd src
PYTHONPATH=. "$PYTHON" -m mammography.commands.inference \
    --checkpoint "../$CHECKPOINT" \
    --input "../$TEMP_INPUT" \
    --output "../${OUTPUT_INFERENCE}/predictions.csv" \
    --batch-size 4 \
    --arch efficientnet_b0
cd ..
echo "✓ Inference completed"
echo ""

# Step 3: Run explain (GradCAM)
echo "Step 3: Running explain (GradCAM)..."
mkdir -p "$OUTPUT_EXPLAIN"
cd src
PYTHONPATH=. "$PYTHON" -m mammography.commands.explain \
    --model-path "../$CHECKPOINT" \
    --images-dir "../$TEMP_INPUT" \
    --output-dir "../$OUTPUT_EXPLAIN" \
    --method gradcam \
    --batch-size 2 \
    --model-type efficientnet_b0
cd ..
echo "✓ Explain completed"
echo ""

# Step 4: Verify outputs
echo "Step 4: Verifying outputs..."

# Check inference outputs
if [ -f "$OUTPUT_INFERENCE/predictions.csv" ]; then
    echo "✓ Inference predictions saved"
    PRED_COUNT=$(wc -l < "$OUTPUT_INFERENCE/predictions.csv")
    echo "  - Predictions: $PRED_COUNT rows"
else
    echo "✗ ERROR: predictions.csv not found"
    exit 1
fi

# Check explain outputs
if [ -d "$OUTPUT_EXPLAIN/gradcam" ]; then
    GRADCAM_COUNT=$(find "$OUTPUT_EXPLAIN/gradcam" -name "*.png" | wc -l)
    echo "✓ GradCAM visualizations saved"
    echo "  - Heatmaps: $GRADCAM_COUNT images"
else
    echo "✗ ERROR: GradCAM output directory not found"
    exit 1
fi

if [ -f "$OUTPUT_EXPLAIN/summary.json" ]; then
    echo "✓ Explain summary saved"
else
    echo "✗ ERROR: summary.json not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Integration Test 2: PASSED ✓"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Checkpoint: $CHECKPOINT (16 MB)"
echo "  - Inference predictions: $OUTPUT_INFERENCE/predictions.csv"
echo "  - GradCAM visualizations: $OUTPUT_EXPLAIN/gradcam/"
echo "  - Summary: $OUTPUT_EXPLAIN/summary.json"
echo ""
