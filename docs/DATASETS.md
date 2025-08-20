# Dataset Documentation

This document provides comprehensive information about datasets used with the CG-HCAN framework for hierarchical remote sensing segmentation.

## Supported Datasets

### 1. GID Dataset (Gaofen Image Dataset)

The GID dataset is our primary benchmark for land use classification.

#### Dataset Overview

- **Source**: Gaofen-2 satellite imagery
- **Resolution**: 4m per pixel
- **Coverage**: Multiple regions in China
- **Classes**: Hierarchical land use categories
- **Total Images**: 150 high-resolution images
- **Image Size**: Variable (typically 7200×6800 pixels)

#### Class Hierarchy

```
Level 1 (Binary): Built-up vs. Non-built-up
├── Level 2 (6 classes):
│   ├── Built-up area
│   ├── Farmland
│   ├── Forest
│   ├── Meadow
│   ├── Water
│   └── Other
└── Level 3 (16 classes):
    ├── Industrial land
    ├── Urban residential
    ├── Rural residential
    ├── Traffic land
    ├── Paddy field
    ├── Irrigated land
    ├── Dry cropland
    ├── Garden plot
    ├── Arbor woodland
    ├── Shrub land
    ├── Natural grassland
    ├── Artificial grassland
    ├── River
    ├── Lake
    ├── Pond
    └── Other land
```

#### Directory Structure

```
GID/
├── train/
│   ├── image/          # RGB satellite images (.tif)
│   ├── label1/         # Level-1 binary masks (.tif)
│   ├── label2/         # Level-2 category masks (.tif)
│   └── label3/         # Level-3 fine-grained masks (.tif)
├── test/
│   ├── image/          # Test images (.tif)
│   └── label1/         # Test binary masks (.tif)
└── metadata/
    ├── class_mapping.json
    ├── hierarchy.json
    └── statistics.json
```

#### Download Instructions

1. **Official Source**: Contact the GID dataset authors
2. **Preprocessed Version**: Available upon request (userxyb@whu.edu.cn)
3. **Sample Data**: [Download link](https://pan.baidu.com/s/example)

### 2. Custom Forest Dataset

A specialized dataset for forest type classification.

#### Dataset Overview

- **Source**: Landsat-8 and Sentinel-2 imagery
- **Resolution**: 10-30m per pixel
- **Coverage**: Forest regions worldwide
- **Classes**: Hierarchical forest types
- **Total Images**: 200+ images
- **Image Size**: 1024×1024 pixels

#### Class Hierarchy

```
Level 1: Forest vs. Non-forest
├── Level 2 (4 classes):
│   ├── Deciduous forest
│   ├── Coniferous forest
│   ├── Mixed forest
│   └── Other vegetation
└── Level 3 (12 classes):
    ├── Broadleaf deciduous
    ├── Needle-leaf deciduous
    ├── Broadleaf evergreen
    ├── Needle-leaf evergreen
    ├── Mixed broadleaf
    ├── Mixed needle-leaf
    ├── Grassland
    ├── Shrubland
    ├── Wetland vegetation
    ├── Agricultural crops
    ├── Urban vegetation
    └── Bare land
```

## Data Preparation

### 1. Download and Extract

```bash
# Create data directory
mkdir -p data/raw

# Download GID dataset (example)
wget -O data/raw/gid_dataset.zip "https://example.com/gid_dataset.zip"
unzip data/raw/gid_dataset.zip -d data/raw/

# Organize directory structure
python scripts/organize_dataset.py --input data/raw/GID --output data/GID
```

### 2. Data Preprocessing

```bash
# Run preprocessing script
python dataProcess.py \
    --dataset_path data/GID \
    --output_path data/GID_processed \
    --resize_to 512 \
    --normalize \
    --split_ratio 0.8
```

### 3. Quality Check

```bash
# Verify data integrity
python scripts/verify_dataset.py --dataset_path data/GID_processed
```

## Dataset Statistics

### GID Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | 150 |
| Training Images | 120 |
| Test Images | 30 |
| Average Image Size | 7200×6800 |
| Total Pixels | 7.7B |
| Class Distribution | Balanced |

### Class Distribution (Level 2)

| Class | Percentage | Pixel Count |
|-------|------------|-------------|
| Built-up area | 18.5% | 1.42B |
| Farmland | 35.2% | 2.71B |
| Forest | 22.8% | 1.75B |
| Meadow | 12.1% | 0.93B |
| Water | 8.4% | 0.65B |
| Other | 3.0% | 0.23B |

## Creating Custom Datasets

### 1. Data Format Requirements

Your dataset should follow this structure:

```
your_dataset/
├── train/
│   ├── image/          # RGB images (.tif, .jpg, .png)
│   ├── label1/         # Primary binary masks
│   ├── label2/         # Level-2 category masks
│   └── label3/         # Level-3 fine-grained masks
├── test/
│   ├── image/
│   └── label1/
└── metadata/
    └── class_mapping.json
```

### 2. Class Mapping File

Create a `class_mapping.json` file:

```json
{
  "level1_to_level2": {
    "0": [0],
    "1": [1, 2, 3, 4, 5]
  },
  "level2_to_level3": {
    "0": [0],
    "1": [1, 2, 3, 4],
    "2": [5, 6, 7, 8],
    "3": [9, 10],
    "4": [11, 12, 13],
    "5": [14, 15]
  },
  "class_names": {
    "level2": ["background", "built-up", "farmland", "forest", "meadow", "water"],
    "level3": ["background", "industrial", "urban", "rural", "traffic", 
               "paddy", "irrigated", "dry", "garden", "arbor", "shrub",
               "natural_grass", "artificial_grass", "river", "lake", "pond"]
  }
}
```

### 3. Validation Script

Use our validation script to check your dataset:

```bash
python scripts/validate_custom_dataset.py \
    --dataset_path your_dataset/ \
    --config_path configs/your_config.py
```

## Data Augmentation

The framework includes comprehensive data augmentation:

### Spatial Augmentations
- Random rotation (±30°)
- Random scaling (0.7-1.3×)
- Random horizontal/vertical flipping
- Random cropping and padding

### Color Augmentations
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)
- Hue/saturation shifts (±10%)
- Gaussian noise addition

### Advanced Augmentations
- Mixup and CutMix
- Mosaic augmentation
- Multi-scale training
- Test-time augmentation (TTA)

## Performance Benchmarks

### GID Dataset Benchmarks

| Method | Level-2 mIoU | Level-3 mIoU | Overall mIoU | Training Time |
|--------|--------------|--------------|--------------|---------------|
| UNet | 72.3% | 65.1% | 68.7% | 4h |
| UNet++ | 74.8% | 67.9% | 71.4% | 5h |
| DeepLabV3+ | 75.2% | 68.4% | 71.8% | 6h |
| **CG-HCAN** | **78.5%** | **71.2%** | **74.9%** | **7h** |

### Forest Dataset Benchmarks

| Method | Level-2 mIoU | Level-3 mIoU | Overall mIoU |
|--------|--------------|--------------|--------------|
| UNet | 68.9% | 61.3% | 65.1% |
| **CG-HCAN** | **74.2%** | **66.8%** | **70.5%** |

## Data Loading Optimization

### Recommended Settings

```python
# For training
BATCH_SIZE = 14  # Adjust based on GPU memory
NUM_WORKERS = 8  # Number of CPU cores
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# For inference
BATCH_SIZE = 8
NUM_WORKERS = 4
```

### Memory Usage Guidelines

| Image Size | Batch Size | GPU Memory | Recommended GPU |
|------------|------------|------------|-----------------|
| 512×512 | 16 | 8GB | RTX 3070 |
| 512×512 | 32 | 12GB | RTX 3080 |
| 768×768 | 8 | 12GB | RTX 3080 |
| 1024×1024 | 4 | 16GB | RTX 4080 |

## Troubleshooting

### Common Issues

1. **Mismatched dimensions**: Ensure all masks have the same spatial dimensions as images
2. **Invalid class indices**: Check that class indices are within valid ranges
3. **Memory errors**: Reduce batch size or image resolution
4. **Slow loading**: Optimize data storage format (use HDF5 or LMDB)

### Data Quality Checks

```bash
# Check for corrupted images
python scripts/check_data_quality.py --dataset_path data/GID_processed

# Verify class distributions
python scripts/analyze_class_distribution.py --dataset_path data/GID_processed

# Check hierarchy consistency
python scripts/verify_hierarchy.py --dataset_path data/GID_processed
```

## Contributing New Datasets

We welcome contributions of new datasets! Please:

1. Follow the standard directory structure
2. Include comprehensive metadata
3. Provide data quality reports
4. Submit a pull request with documentation

For more information, contact us at userxyb@whu.edu.cn
