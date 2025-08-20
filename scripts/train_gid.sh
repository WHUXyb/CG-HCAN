#!/bin/bash

# ============================================================================
# CG-HCAN Training Script for GID Dataset
# ============================================================================

echo "Starting CG-HCAN training on GID dataset..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
BATCH_SIZE=14
EPOCHS=85
LEARNING_RATE=2e-4
BICAF_REDUCTION=8
USE_CLIP_GNN=true
N_SPLITS=5

echo "Training Configuration:"
echo "- Batch Size: $BATCH_SIZE"
echo "- Epochs: $EPOCHS"
echo "- Learning Rate: $LEARNING_RATE"
echo "- BiCAF Reduction: $BICAF_REDUCTION"
echo "- CLIP-GNN: $USE_CLIP_GNN"
echo "- Cross-validation splits: $N_SPLITS"
echo ""

# Create output directories
mkdir -p model/gid_experiments
mkdir -p logs/gid_training

# Start training with 5-fold cross-validation
for fold in {0..4}; do
    echo "=================================="
    echo "Training Fold $fold / 4"
    echo "=================================="
    
    python train.py \
        --config configs/gid_config.py \
        --fold $fold \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --bicaf_reduction $BICAF_REDUCTION \
        --use_clip_gnn $USE_CLIP_GNN \
        --output_dir model/gid_experiments/fold_$fold \
        --log_dir logs/gid_training/fold_$fold \
        2>&1 | tee logs/gid_training/fold_${fold}_training.log
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for fold $fold"
        exit 1
    fi
    
    echo "Fold $fold completed successfully!"
    echo ""
done

echo "All folds completed successfully!"
echo "Results saved in: model/gid_experiments/"
echo "Logs saved in: logs/gid_training/"

# Calculate average performance across folds
echo "Calculating average performance across folds..."
python scripts/calculate_cv_results.py \
    --results_dir model/gid_experiments \
    --output_file results/gid_cv_summary.json

echo "Training completed! Check results/gid_cv_summary.json for final metrics."
