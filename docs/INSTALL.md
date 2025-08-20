# Installation Guide

This document provides detailed installation instructions for the CG-HCAN framework.

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3080 or better)
- **Memory**: At least 16GB RAM (32GB recommended for large datasets)
- **Storage**: At least 50GB free space for datasets and model checkpoints

### Software Requirements

- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, or macOS 10.15+
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (if using GPU)

## Step-by-Step Installation

### 1. Create Python Environment

We recommend using conda or virtualenv to create an isolated environment:

```bash
# Using conda
conda create -n cg-hcan python=3.8
conda activate cg-hcan

# Or using virtualenv
python -m venv cg-hcan
source cg-hcan/bin/activate  # On Windows: cg-hcan\Scripts\activate
```

### 2. Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Clone Repository

```bash
git clone https://github.com/WHUXyb/CG-HCAN.git
cd CG-HCAN
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install CLIP (Optional)

For CLIP-GNN enhancement:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 6. Verify Installation

Run the verification script to check if everything is installed correctly:

```python
python -c "
import torch
import torchvision
import segmentation_models_pytorch as smp
print('PyTorch version:', torch.__version__)
print('Torchvision version:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU names:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('SMP version:', smp.__version__)
"
```

## Alternative Installation Methods

### Using Docker

A Docker image is available for easy deployment:

```bash
docker pull whuxxyb/cg-hcan:latest
docker run --gpus all -it --rm -v $(pwd):/workspace whuxxyb/cg-hcan:latest
```

### Using pip (Package Installation)

Install directly from PyPI (when available):

```bash
pip install cg-hcan
```

## Common Installation Issues

### Issue 1: CUDA Version Mismatch

**Problem**: PyTorch was installed with a different CUDA version than your system.

**Solution**: 
1. Check your CUDA version: `nvcc --version`
2. Install PyTorch with matching CUDA version
3. Visit [PyTorch installation page](https://pytorch.org/get-started/locally/) for exact commands

### Issue 2: Segmentation Models PyTorch Import Error

**Problem**: `ImportError: No module named 'segmentation_models_pytorch'`

**Solution**:
```bash
pip install segmentation-models-pytorch
```

### Issue 3: CLIP Installation Error

**Problem**: Error installing CLIP from GitHub.

**Solution**:
```bash
# Alternative installation
pip install clip-by-openai
```

### Issue 4: Memory Error During Training

**Problem**: Out of memory error during training.

**Solution**:
1. Reduce batch size in configuration
2. Enable mixed precision training (set `USE_AMP = True`)
3. Use gradient accumulation

### Issue 5: Permission Error on Windows

**Problem**: Permission denied when creating directories.

**Solution**:
1. Run command prompt as administrator
2. Or install in user directory: `pip install --user`

## Performance Optimization

### 1. CUDA Optimization

```bash
# Enable optimized CUDA kernel selection
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=True
```

### 2. Data Loading Optimization

```python
# In your configuration
NUM_WORKERS = min(8, os.cpu_count())
PIN_MEMORY = True
```

### 3. Mixed Precision Training

```python
# Enable in configuration
USE_AMP = True
```

## Development Installation

For development and contributions:

```bash
# Clone with development dependencies
git clone https://github.com/WHUXyb/CG-HCAN.git
cd CG-HCAN

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Uninstallation

To remove CG-HCAN and clean up:

```bash
# Remove conda environment
conda env remove -n cg-hcan

# Or remove virtualenv
rm -rf cg-hcan/

# Remove CUDA cache
rm -rf ~/.cache/torch
```

## Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/WHUXyb/CG-HCAN/issues)
2. Create a new issue with:
   - Your OS and Python version
   - Complete error messages
   - Steps you've already tried
3. Contact us at userxyb@whu.edu.cn

## Next Steps

After successful installation:

1. Download and prepare your dataset (see [DATASETS.md](DATASETS.md))
2. Run the quick start example (see main [README.md](../README.md))
3. Explore the configuration options (see [configs/](../configs/))
4. Check out the example scripts (see [scripts/](../scripts/))
