# Hierarchical Category Refinement for Remote Sensing Segmentation A New Task, Benchmark Datasets, and the CG-HCAN Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains the official implementation of **"Hierarchical Category Refinement for Remote Sensing Segmentation: A New Task, Benchmark Datasets, and the CG-HCAN Framework"** published in IEEE Transactions on Geoscience and Remote Sensing.

## 🌟 Highlights

- **📊 New Task**: Introduces hierarchical category refinement for remote sensing segmentation
- **🏗️ Novel Architecture**: CG-HCAN with dual-encoder MiT-B2 and BiCAF fusion
- **🔗 CLIP Integration**: CLIP-guided Graph Neural Network for semantic enhancement
- **📈 SOTA Performance**: Achieves state-of-the-art results on GID and custom datasets
- **🔄 Hierarchical Consistency**: Multi-level prediction with hierarchy consistency loss

## 🚀 Key Features

### 🏗️ Architecture Overview

```
CG-HCAN Framework:
├── Dual-Branch Encoder
│   ├── RGB Image Branch (MiT-B2)
│   └── Mask Branch (Simplified Encoder)
├── BiCAF Fusion Module
│   ├── Bidirectional Cross-Attention
│   ├── Depthwise Separable Convolution
│   └── Local Attention Mechanism
├── UNet++ Decoder
│   ├── Multi-scale Feature Fusion
│   └── Dense Skip Connections
├── Semantic Enhancement Module
│   ├── CLIP-Guided GNN
│   └── Adaptive Semantic GNN
└── Multi-level Prediction Heads
    ├── Level-2 Classification Head
    └── Level-3 Classification Head
```

### 🔥 Core Innovations

1. **BiCAF Fusion**: Bidirectional Cross-Attention Fusion for effective multi-modal feature integration
2. **CLIP-GNN**: CLIP-guided Graph Neural Network for semantic-aware fine-grained classification
3. **Hierarchy Consistency**: Novel loss function ensuring logical hierarchy between prediction levels
4. **Test-Time Augmentation (TTA)**: Multi-directional flip augmentation for improved inference accuracy

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

### Supported Datasets

- **GID Dataset**: Gaofen Image Dataset for land use classification
- **Custom Datasets**: Support for custom hierarchical segmentation datasets

## 🏃‍♂️ Quick Start

### Training

```bash
# Basic training with BiCAF fusion
python train.py

# Enable CLIP-GNN enhancement
python train.py --use_clip_gnn True

# Custom configuration training
python train.py \
    --batch_size 16 \
    --lr 2e-4 \
    --epochs 100 \
    --bicaf_reduction 8 \
    --use_multi_scale True
```

### Inference

```bash
# Basic inference
python infer.py \
    --model_path ./model/best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results

# Enable Test-Time Augmentation (TTA)
python infer.py \
    --model_path ./model/best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results \
    --use_tta True
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

### Training Hyperparameters

```python
# Core Training Parameters
EPOCHS = 85                      # Training epochs
BATCH_SIZE = 14                  # Batch size
LR = 2e-4                       # Learning rate
WEIGHT_DECAY = 5e-5             # Weight decay
USE_AMP = True                  # Mixed precision training
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
  author={Xiong, Y. and Zhang, B. and Li, S.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## 📞 Contact

- **Authors**: Xiong et al.
- **Email**: userxyb@whu.edu.cn
- **Institution**: Wuhan University

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
