# -*- coding: utf-8 -*-
"""
forest_config.py

Configuration file for forest/ecological monitoring dataset.
Specialized configuration for fine-grained forest type classification.
"""

import os

# ============================================================================
# Dataset Configuration
# ============================================================================

# Dataset paths (modify according to your setup)
DATASET_ROOT = "/path/to/your/forest_data"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train/image")
TRAIN_LABEL1_DIR = os.path.join(DATASET_ROOT, "train/label1")  # Forest/non-forest
TRAIN_LABEL2_DIR = os.path.join(DATASET_ROOT, "train/label2")  # Forest types
TRAIN_LABEL3_DIR = os.path.join(DATASET_ROOT, "train/label3")  # Species-level

TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test/image")
TEST_LABEL1_DIR = os.path.join(DATASET_ROOT, "test/label1")

# Model and output directories
MODEL_OUTPUT_DIR = "./model_forest"
RESULTS_OUTPUT_DIR = "./results_forest"

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# Encoder settings (ResNet for forest imagery)
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3

# UNet++ Decoder settings
DECODER_CHANNELS = [512, 256, 128, 64, 32]
DECODER_USE_BATCHNORM = True
DECODER_ATTENTION_TYPE = "scse"  # Spatial and channel squeeze & excitation

# BiCAF Fusion settings
USE_BICAF_FUSION = True
BICAF_REDUCTION_FACTOR = 16  # Higher reduction for ResNet50
USE_MULTI_SCALE_FUSION = True

# GNN Enhancement settings (important for fine-grained forest classification)
USE_ADAPTIVE_GNN = True
USE_CLIP_GNN = True
CLIP_MODEL_NAME = "ViT-L/14"  # Larger CLIP model for better forest understanding
GNN_HIDDEN_DIM = 512
GNN_NUM_LAYERS = 3

# Output classes (forest-specific)
NUM_CLASSES_LEVEL2 = 4   # Major forest types (deciduous, coniferous, mixed, other)
NUM_CLASSES_LEVEL3 = 12  # Species-level classification

# ============================================================================
# Training Configuration
# ============================================================================

# Basic training parameters
BATCH_SIZE = 8  # Smaller batch for ResNet50
NUM_EPOCHS = 120
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
WEIGHT_DECAY = 1e-4

# Cross-validation
N_SPLITS = 5
RANDOM_SEED = 42

# Loss function weights (emphasize fine-grained classification)
LOSS_WEIGHT_LEVEL2 = 0.8
LOSS_WEIGHT_LEVEL3 = 1.2
HIERARCHY_CONSISTENCY_WEIGHT = 0.7

# Optimization settings
USE_AMP = True
GRADIENT_CLIP_VALUE = 0.5

# Learning rate scheduler
LR_SCHEDULER = "CosineAnnealingWarmRestarts"
LR_T_0 = 20
LR_T_MULT = 2
LR_ETA_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0005

# ============================================================================
# Data Processing Configuration
# ============================================================================

# Image preprocessing (higher resolution for forest details)
IMAGE_SIZE = (768, 768)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data augmentation (forest-specific)
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.3,  # Less vertical flip for forest
    "rotation": 30,
    "scale": (0.7, 1.3),
    "brightness": 0.3,
    "contrast": 0.3,
    "hue_saturation": 0.1,  # Important for forest color variations
    "gaussian_noise": 0.02,
}

# ============================================================================
# Inference Configuration
# ============================================================================

# Test-time augmentation
USE_TTA = True
TTA_TRANSFORMS = [
    "original", 
    "horizontal_flip", 
    "rotation_90", 
    "rotation_270"
]

# Output settings
SAVE_PREDICTIONS = True
SAVE_VISUALIZATIONS = True
VISUALIZATION_OPACITY = 0.6

# ============================================================================
# Hardware Configuration
# ============================================================================

# Device settings
DEVICE = "cuda:0"
NUM_WORKERS = 6  # Fewer workers for higher resolution
PIN_MEMORY = True

# ============================================================================
# Forest-Specific Settings
# ============================================================================

# Seasonal considerations
SEASONAL_AUGMENTATION = True
LEAF_COLOR_VARIATIONS = True

# Multi-spectral support (if available)
USE_MULTISPECTRAL = False
SPECTRAL_BANDS = ["red", "green", "blue", "nir"]  # Add NIR if available

# ============================================================================
# Logging and Monitoring
# ============================================================================

# Experiment tracking
USE_TENSORBOARD = True
USE_WANDB = True
WANDB_PROJECT = "CG-HCAN-Forest"
WANDB_ENTITY = "forest-monitoring"

# Logging settings
LOG_LEVEL = "INFO"
LOG_INTERVAL = 5
SAVE_INTERVAL = 10

# Checkpoint settings
SAVE_BEST_ONLY = True
MONITOR_METRIC = "val_level3_f1"  # F1 score important for imbalanced forest classes
MONITOR_MODE = "max"
