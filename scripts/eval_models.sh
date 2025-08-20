#!/bin/bash

# ============================================================================
# CG-HCAN Model Evaluation Script
# ============================================================================

echo "Starting CG-HCAN model evaluation..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Evaluation parameters
BATCH_SIZE=8
USE_TTA=true
OUTPUT_DIR="results/evaluation"

echo "Evaluation Configuration:"
echo "- Batch Size: $BATCH_SIZE"
echo "- Test-Time Augmentation: $USE_TTA"
echo "- Output Directory: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/visualizations
mkdir -p $OUTPUT_DIR/predictions
mkdir -p logs/evaluation

# Model paths to evaluate
declare -a MODELS=(
    "model/gid_experiments/fold_0/best_model.pth"
    "model/gid_experiments/fold_1/best_model.pth"
    "model/gid_experiments/fold_2/best_model.pth"
    "model/gid_experiments/fold_3/best_model.pth"
    "model/gid_experiments/fold_4/best_model.pth"
)

# Test data paths
TEST_IMAGE_DIR="data/test/image"
TEST_LABEL1_DIR="data/test/label1"

echo "Found ${#MODELS[@]} models to evaluate"
echo ""

# Evaluate each model
for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    fold_num=$i
    
    echo "=================================="
    echo "Evaluating Model: Fold $fold_num"
    echo "Model Path: $model_path"
    echo "=================================="
    
    if [ ! -f "$model_path" ]; then
        echo "WARNING: Model file not found: $model_path"
        echo "Skipping fold $fold_num"
        continue
    fi
    
    # Run inference
    python infer.py \
        --model_path "$model_path" \
        --test_image_dir "$TEST_IMAGE_DIR" \
        --test_label1_dir "$TEST_LABEL1_DIR" \
        --output_dir "$OUTPUT_DIR/fold_$fold_num" \
        --batch_size $BATCH_SIZE \
        --use_tta $USE_TTA \
        --save_visualizations \
        2>&1 | tee logs/evaluation/fold_${fold_num}_eval.log
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for fold $fold_num"
        continue
    fi
    
    echo "Fold $fold_num evaluation completed!"
    echo ""
done

# Calculate ensemble predictions
echo "=================================="
echo "Calculating Ensemble Predictions"
echo "=================================="

python scripts/ensemble_predictions.py \
    --input_dirs $OUTPUT_DIR/fold_* \
    --output_dir $OUTPUT_DIR/ensemble \
    --method "average" \
    2>&1 | tee logs/evaluation/ensemble_eval.log

# Calculate final metrics
echo "=================================="
echo "Calculating Final Metrics"
echo "=================================="

python scripts/calculate_metrics.py \
    --predictions_dir $OUTPUT_DIR \
    --ground_truth_dir "$TEST_LABEL1_DIR" \
    --output_file $OUTPUT_DIR/final_metrics.json \
    --include_ensemble \
    2>&1 | tee logs/evaluation/metrics_calculation.log

echo ""
echo "Evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: logs/evaluation/"
echo "Final metrics: $OUTPUT_DIR/final_metrics.json"

# Display summary
if [ -f "$OUTPUT_DIR/final_metrics.json" ]; then
    echo ""
    echo "=== EVALUATION SUMMARY ==="
    python -c "
import json
with open('$OUTPUT_DIR/final_metrics.json', 'r') as f:
    metrics = json.load(f)
    
print('Individual Fold Results:')
for fold in range(5):
    if f'fold_{fold}' in metrics:
        fold_metrics = metrics[f'fold_{fold}']
        print(f'  Fold {fold}: Level-2 mIoU = {fold_metrics.get(\"level2_miou\", \"N/A\"):.3f}, Level-3 mIoU = {fold_metrics.get(\"level3_miou\", \"N/A\"):.3f}')

if 'ensemble' in metrics:
    ens_metrics = metrics['ensemble']
    print(f'\\nEnsemble Results:')
    print(f'  Level-2 mIoU = {ens_metrics.get(\"level2_miou\", \"N/A\"):.3f}')
    print(f'  Level-3 mIoU = {ens_metrics.get(\"level3_miou\", \"N/A\"):.3f}')
    print(f'  Overall mIoU = {ens_metrics.get(\"overall_miou\", \"N/A\"):.3f}')
"
fi
