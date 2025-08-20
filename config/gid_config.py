# -*- coding: utf-8 -*-
"""
gid_config.py

Configuration file for GID dataset training and inference.
This file contains all hyperparameters and paths for the CG-HCAN model.
"""

import os

# ============================================================================
# Dataset Configuration
# ============================================================================

# Dataset paths (modify according to your setup)
DATASET_ROOT = "/path/to/your/data"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train/image")
TRAIN_LABEL1_DIR = os.path.join(DATASET_ROOT, "train/label1")  # Primary masks
TRAIN_LABEL2_DIR = os.path.join(DATASET_ROOT, "train/label2")  # Level-2 labels
TRAIN_LABEL3_DIR = os.path.join(DATASET_ROOT, "train/label3")  # Level-3 labels

TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test/image")
TEST_LABEL1_DIR = os.path.join(DATASET_ROOT, "test/label1")

# Model and output directories
MODEL_OUTPUT_DIR = "./model"
RESULTS_OUTPUT_DIR = "./results"

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# Encoder settings
ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3

# UNet++ Decoder settings
DECODER_CHANNELS = [256, 128, 64, 32, 16]
DECODER_USE_BATCHNORM = True
DECODER_ATTENTION_TYPE = None

# BiCAF Fusion settings
USE_BICAF_FUSION = True
BICAF_REDUCTION_FACTOR = 8  # Channel reduction factor (8, 16, or 32)
USE_MULTI_SCALE_FUSION = True

# GNN Enhancement settings
USE_ADAPTIVE_GNN = False
USE_CLIP_GNN = True
CLIP_MODEL_NAME = "ViT-B/32"
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 2

# Output classes
NUM_CLASSES_LEVEL2 = 6   # Level-2 categories
NUM_CLASSES_LEVEL3 = 16  # Level-3 categories

# ============================================================================
# Training Configuration
# ============================================================================

# Basic training parameters
BATCH_SIZE = 14
NUM_EPOCHS = 85
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-5

# Cross-validation
N_SPLITS = 5  # 5-fold cross-validation
RANDOM_SEED = 42

# Loss function weights
LOSS_WEIGHT_LEVEL2 = 1.0
LOSS_WEIGHT_LEVEL3 = 1.0
HIERARCHY_CONSISTENCY_WEIGHT = 0.5

# Optimization settings
USE_AMP = True  # Mixed precision training
GRADIENT_CLIP_VALUE = 1.0

# Learning rate scheduler
LR_SCHEDULER = "OneCycleLR"
LR_MAX = 2e-4
LR_PCT_START = 0.3
LR_DIV_FACTOR = 25.0
LR_FINAL_DIV_FACTOR = 1e4

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# ============================================================================
# Data Processing Configuration
# ============================================================================

# Image preprocessing
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.5,
    "rotation": 15,
    "scale": (0.8, 1.2),
    "brightness": 0.2,
    "contrast": 0.2,
}

# ============================================================================
# Inference Configuration
# ============================================================================

# Test-time augmentation
USE_TTA = True
TTA_TRANSFORMS = ["original", "horizontal_flip", "vertical_flip", "both_flip"]

# Output settings
SAVE_PREDICTIONS = True
SAVE_VISUALIZATIONS = True
VISUALIZATION_OPACITY = 0.7

# ============================================================================
# Hardware Configuration
# ============================================================================

# Device settings
DEVICE = "cuda:0"  # Set to "cpu" if CUDA is not available
NUM_WORKERS = 8
PIN_MEMORY = True

# ============================================================================
# Logging and Monitoring
# ============================================================================

# Experiment tracking
USE_TENSORBOARD = True
USE_WANDB = False
WANDB_PROJECT = "CG-HCAN"
WANDB_ENTITY = "your-username"

# Logging settings
LOG_LEVEL = "INFO"
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 5  # Save model every N epochs

# Checkpoint settings
SAVE_BEST_ONLY = True
MONITOR_METRIC = "val_level3_miou"
MONITOR_MODE = "max"
