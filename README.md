# Hierarchical Category Refinement for Remote Sensing Segmentation A New Task, Benchmark Datasets, and the CG-HCAN Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains the official implementation of **"Hierarchical Category Refinement for Remote Sensing Segmentation: A New Task, Benchmark Datasets, and the CG-HCAN Framework"** submitted to IEEE Transactions on Geoscience and Remote Sensing.

## 🌟 Highlights

- **📊 New Task**: Introduces hierarchical category refinement for remote sensing segmentation
- **🏗️ Novel Architecture**: CG-HCAN with dual-encoder framework and BiCAF fusion module
- **🔗 CLIP Integration**: CLIP-guided Graph Neural Network for semantic-aware classification
- **📈 Superior Performance**: Comprehensive evaluation on GID dataset and custom forest dataset
- **🔄 Hierarchical Consistency**: Multi-level prediction ensuring logical hierarchy constraints

## 🚀 Key Features

### 🏗️ Architecture Overview

```
CG-HCAN Framework:
├── Dual-Branch Encoder
│   ├── RGB Image Branch (MiT-B2)
│   └── Mask Branch (Simplified Encoder)
├── BiCAF Fusion Module
│   ├── Bidirectional Cross-Attention Mechanism
│   ├── Channel Reduction and Feature Enhancement
│   └── Spatial Attention Integration
├── UNet++ Decoder
│   ├── Multi-scale Feature Aggregation
│   └── Dense Skip Connections
├── CLIP-Guided Graph Neural Network
│   ├── Category Embedding with CLIP ViT-B/32
│   └── Graph-based Semantic Enhancement
└── Hierarchical Prediction Heads
    ├── Level-2 Land Use Classification
    └── Level-3 Fine-grained Classification
```

### 🔥 Core Innovations

1. **Hierarchical Category Refinement**: Novel task formulation addressing multi-level land use classification
2. **BiCAF Fusion**: Bidirectional Cross-Attention Fusion for effective dual-branch feature integration  
3. **CLIP-GNN Enhancement**: CLIP-guided Graph Neural Network for semantic-aware classification refinement
4. **Hierarchy Consistency Loss**: Mathematical constraint ensuring logical relationships between hierarchical levels

## 🛠️ Installation

### Environment Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (Recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/WHUXyb/CG-HCAN.git
cd CG-HCAN

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install CLIP (optional, for CLIP-GNN)
pip install git+https://github.com/openai/CLIP.git
```

## 📁 Dataset Preparation

### Directory Structure

```
data/
├── train/
│   ├── image/          # RGB remote sensing images (.tif)
│   ├── label1/         # Level-1 labels (binary masks)
│   ├── label2/         # Level-2 labels (multi-class)
│   └── label3/         # Level-3 labels (fine-grained)
├── test/
│   ├── image/
│   └── label1/
└── model/              # Saved model checkpoints
```

### Benchmark Datasets

- **GID Dataset**: Gaofen Image Dataset with hierarchical land use annotation
  - 150 high-resolution Gaofen-2 satellite images (6,800×7,200 pixels)
  - Hierarchical labels: Level-1 (15 classes) → Level-2 (15 classes) → Level-3 (37 classes)
- **Forest Dataset**: Custom forest classification dataset
  - Multi-temporal forest monitoring with hierarchical structure
  - Specialized for forest type and health assessment

## 🏃‍♂️ Quick Start

### Training

```bash
# 5-fold cross-validation training (paper setup)
python train.py

# Training with CLIP-GNN enhancement
python train.py --use_clip_gnn True --clip_model ViT-B/32

# Custom configuration for different datasets
python train.py \
    --epochs 85 \
    --batch_size 14 \
    --lr 2e-4 \
    --weight_decay 5e-5 \
    --bicaf_reduction 8 \
    --use_amp True
```

### Inference

```bash
# Hierarchical inference (Level-2 and Level-3)
python infer.py \
    --model_path ./model/fold0_best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results

# Inference with Test-Time Augmentation
python infer.py \
    --model_path ./model/fold0_best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results \
    --use_tta True \
    --model_version detailed
```

## ⚙️ Configuration

### BiCAF Fusion Configuration

```python
# Bidirectional Cross-Attention Fusion Parameters
USE_BICAF_FUSION = True          # Enable BiCAF fusion
BICAF_REDUCTION_FACTOR = 8       # Channel reduction factor (8/16/32)
USE_MULTI_SCALE_FUSION = True    # Multi-scale fusion
```

### GNN Semantic Enhancement

```python
# Graph Neural Network Configuration
USE_ADAPTIVE_GNN = False         # Basic adaptive GNN
USE_CLIP_GNN = True             # CLIP-guided GNN
CLIP_MODEL_NAME = 'ViT-B/32'    # CLIP model version
```

### Training Configuration

```python
# Experimental Setup (following paper)
EPOCHS = 85                      # Training epochs
BATCH_SIZE = 14                  # Batch size per GPU
LEARNING_RATE = 2e-4            # Initial learning rate
WEIGHT_DECAY = 5e-5             # L2 regularization
OPTIMIZER = 'AdamW'             # AdamW optimizer
LR_SCHEDULER = 'OneCycleLR'     # Learning rate scheduling
CROSS_VALIDATION = 5            # 5-fold cross-validation
MIXED_PRECISION = True          # FP16 mixed precision training
```

## 📈 Performance Optimization

### 1. Memory Optimization

- **Mixed Precision Training**: Uses AMP to reduce GPU memory usage by 50%
- **Gradient Accumulation**: Simulate large batch sizes on small GPUs
- **Local Attention**: BiCAF uses 8×8 block partitioning, reducing complexity from O(N²) to O(N)

### 2. Training Stability

- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: OneCycleLR strategy
- **Early Stopping**: Prevents overfitting

### 3. Inference Acceleration

- **Model Quantization**: Support for INT8 quantized inference
- **TensorRT Optimization**: GPU-accelerated inference
- **Batch Inference**: Support for large-scale data processing

## 🎯 Applications

### Land Use Classification

- Urban Planning
- Environmental Monitoring
- Agricultural Management
- Natural Resource Survey

### Ecological Environment Monitoring

- Forest Cover Analysis
- Wetland Protection
- Desertification Monitoring
- Biodiversity Assessment

## 📂 Project Structure

```
CG-HCAN/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── model.py                  # Core network architecture
├── train.py                  # Training script
├── infer.py                  # Inference script
├── dataProcess.py           # Data processing utilities
├── DataAugmentation.py      # Data augmentation module
├── hierarchy_dict.py        # Category hierarchy mapping
├── cal_acc2.py             # Accuracy calculation tools
├── configs/                 # Configuration files
│   ├── gid_config.py       # GID dataset config
│   └── forest_config.py    # Forest dataset config
├── scripts/                 # Run scripts
│   ├── train_gid.sh        # GID training script
│   └── eval_models.sh      # Model evaluation script
└── docs/                   # Detailed documentation
    ├── INSTALL.md          # Installation guide
    ├── DATASETS.md         # Dataset description
    └── MODEL_DETAILS.md    # Model details
```

## 📋 TODO List

- [ ] Support for more pre-trained backbone networks
- [ ] Real-time inference demo
- [ ] More benchmark datasets
- [ ] Pre-trained model downloads
- [ ] Distributed training support
- [ ] Web visualization interface

## 📄 Citation

If this work is helpful for your research, please consider citing our paper:

```bibtex
@article{xiong2024hierarchical,
  title={Hierarchical Category Refinement for Remote Sensing Segmentation: A New Task, Benchmark Datasets, and the CG-HCAN Framework},
  author={Xiong, Yuanbo and Zhang, Bowen and Li, Songtao and Wang, Jianwei and Liu, Mingming},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  note={Under Review}
}
```

## 📞 Contact

- **Corresponding Author**: Yuanbo Xiong
- **Email**: userxyb@whu.edu.cn
- **Institution**: School of Resource and Environmental Sciences, Wuhan University
- **Address**: 129 Luoyu Road, Wuhan 430079, China

## 📜 License

This project is released under the [MIT License](LICENSE).

## 🤝 Contributing

We welcome Issues and Pull Requests! Please refer to the contributing guidelines for details.

## 🔗 Related Links

- [Paper Preprint](https://arxiv.org/abs/xxxx.xxxxx)
- [Project Homepage](https://whuxxyb.github.io/CG-HCAN)
- [Dataset Download](https://pan.baidu.com/s/xxxxxxxxx)
- [Pre-trained Models](https://github.com/WHUXyb/CG-HCAN/releases)

---

⭐ If this project helps you, please give us a star!

## Acknowledgments

We thank the contributors to the open-source libraries that made this work possible:
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
