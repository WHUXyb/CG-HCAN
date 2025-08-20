@echo off
REM ============================================================================
REM CG-HCAN Training Script for GID Dataset (Windows)
REM ============================================================================

echo Starting CG-HCAN training on GID dataset...

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0

REM Training parameters
set BATCH_SIZE=14
set EPOCHS=85
set LEARNING_RATE=2e-4
set BICAF_REDUCTION=8
set USE_CLIP_GNN=true
set N_SPLITS=5

echo Training Configuration:
echo - Batch Size: %BATCH_SIZE%
echo - Epochs: %EPOCHS%
echo - Learning Rate: %LEARNING_RATE%
echo - BiCAF Reduction: %BICAF_REDUCTION%
echo - CLIP-GNN: %USE_CLIP_GNN%
echo - Cross-validation splits: %N_SPLITS%
echo.

REM Create output directories
if not exist "model\gid_experiments" mkdir "model\gid_experiments"
if not exist "logs\gid_training" mkdir "logs\gid_training"

REM Start training with 5-fold cross-validation
for /L %%i in (0,1,4) do (
    echo ==================================
    echo Training Fold %%i / 4
    echo ==================================
    
    python train.py ^
        --config configs/gid_config.py ^
        --fold %%i ^
        --batch_size %BATCH_SIZE% ^
        --epochs %EPOCHS% ^
        --lr %LEARNING_RATE% ^
        --bicaf_reduction %BICAF_REDUCTION% ^
        --use_clip_gnn %USE_CLIP_GNN% ^
        --output_dir model/gid_experiments/fold_%%i ^
        --log_dir logs/gid_training/fold_%%i ^
        > logs/gid_training/fold_%%i_training.log 2>&1
    
    if errorlevel 1 (
        echo ERROR: Training failed for fold %%i
        exit /b 1
    )
    
    echo Fold %%i completed successfully!
    echo.
)

echo All folds completed successfully!
echo Results saved in: model/gid_experiments/
echo Logs saved in: logs/gid_training/

REM Calculate average performance across folds
echo Calculating average performance across folds...
python scripts/calculate_cv_results.py ^
    --results_dir model/gid_experiments ^
    --output_file results/gid_cv_summary.json

echo Training completed! Check results/gid_cv_summary.json for final metrics.
pause
