# Dataset Download Instructions

## Dataset Access

The datasets used in this project can be downloaded from the following link:

**Download Link**: https://pan.baidu.com/s/1f2Mv9YnrB05PHz6UMCR1Kg?pwd=0onh

**Access Code**: 0onh

## Dataset Structure

After downloading and extracting the dataset, organize the files according to the following structure:

```
data/
├── train/
│   ├── image/          # RGB remote sensing images (.tif)
│   ├── label1/         # Level-1 labels (binary masks)
│   ├── label2/         # Level-2 labels (multi-class)
│   └── label3/         # Level-3 labels (fine-grained)
├── test/
│   ├── image/          # Test RGB images
│   └── label1/         # Test Level-1 labels
└── model/              # Pre-trained model checkpoints
```

## Dataset Information

### GID Dataset
- **Source**: Gaofen Image Dataset
- **Resolution**: High-resolution Gaofen-2 satellite images
- **Image Size**: 6,800×7,200 pixels
- **Total Images**: 150 images
- **Hierarchical Structure**: 
  - Level-1: 15 classes (primary land use categories)
  - Level-2: 15 classes (secondary land use categories) 
  - Level-3: 37 classes (fine-grained land use categories)

### Forest Dataset
- **Source**: Custom forest classification dataset
- **Purpose**: Multi-temporal forest monitoring
- **Focus**: Forest type and health assessment
- **Structure**: Hierarchical forest classification labels

## Usage Notes

1. Ensure all image files are in `.tif` format
2. Maintain the hierarchical label structure for proper training
3. Check that image and label file names correspond correctly
4. Verify data integrity after download and extraction

## Contact

For any issues with dataset access or structure, please contact:
- **Email**: userxyb@whu.edu.cn
