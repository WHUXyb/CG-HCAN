# Model Architecture Details

This document provides in-depth technical details about the CG-HCAN (Category-Guided Hierarchical Cross-Attention Network) architecture.

## Overview

CG-HCAN is designed for hierarchical category refinement in remote sensing segmentation, capable of predicting multiple levels of semantic detail simultaneously while maintaining logical consistency between hierarchy levels.

## Architecture Components

### 1. Dual-Branch Encoder

The model uses a dual-branch encoder design to process both RGB imagery and binary masks.

#### RGB Image Branch

```python
class RGBEncoder(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0'):
        # Primary encoder for RGB satellite imagery
        self.encoder = get_encoder(
            name=encoder_name,
            weights='imagenet',
            in_channels=3
        )
```

**Features:**
- Pre-trained EfficientNet-B0 backbone
- Multi-scale feature extraction (5 levels)
- Skip connections for fine-grained details
- Efficient parameter utilization (5.3M parameters)

#### Mask Branch

```python
class MaskEncoder(nn.Module):
    def __init__(self):
        # Simplified encoder for binary masks
        self.conv_layers = nn.ModuleList([
            self._make_layer(1, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
            self._make_layer(256, 512),
        ])
```

**Features:**
- Lightweight design (0.8M parameters)
- Focuses on spatial structure
- Complementary to RGB features
- Reduces computational overhead

### 2. BiCAF Fusion Module

Bidirectional Cross-Attention Fusion for effective multi-modal feature integration.

#### Mathematical Formulation

Given RGB features $F_R \in \mathbb{R}^{C \times H \times W}$ and mask features $F_M \in \mathbb{R}^{C \times H \times W}$:

**Cross-Attention Computation:**

```
Q_R = F_R \cdot W_Q^R
K_M = F_M \cdot W_K^M  
V_M = F_M \cdot W_V^M

Attention_{R→M} = softmax(\frac{Q_R \cdot K_M^T}{\sqrt{d_k}}) \cdot V_M
```

**Bidirectional Fusion:**

```
F_{R→M} = F_R + α · Attention_{R→M}
F_{M→R} = F_M + β · Attention_{M→R}
F_{fused} = γ · F_{R→M} + (1-γ) · F_{M→R}
```

#### Implementation Details

```python
class BiCAF(nn.Module):
    def __init__(self, channels, reduction=8):
        self.channels = channels
        self.reduced_channels = channels // reduction
        
        # Cross-attention components
        self.query_conv_rgb = nn.Conv2d(channels, self.reduced_channels, 1)
        self.key_conv_mask = nn.Conv2d(channels, self.reduced_channels, 1)
        self.value_conv_mask = nn.Conv2d(channels, channels, 1)
        
        # Local attention for efficiency
        self.local_attention = LocalAttention(kernel_size=8)
        
        # Fusion weights (learnable)
        self.fusion_weight = nn.Parameter(torch.ones(1))
```

**Key Innovations:**
- **Local Attention**: 8×8 block-wise attention reduces complexity from O(N²) to O(N)
- **Adaptive Weights**: Learnable fusion coefficients
- **Channel Reduction**: Factor of 8 reduces computational cost
- **Depthwise Separable Convolution**: Efficient spatial processing

### 3. UNet++ Decoder

Enhanced decoder with dense skip connections and multi-scale fusion.

#### Architecture Design

```
Decoder Levels:
├── Level 4: 256 channels (16×16)
├── Level 3: 128 channels (32×32)  
├── Level 2: 64 channels (64×64)
├── Level 1: 32 channels (128×128)
└── Level 0: 16 channels (256×256)
```

#### Dense Skip Connections

Unlike standard UNet, UNet++ uses dense connections:

```
X^{i,j} = H([X^{i,j-1}, Up(X^{i+1,j})])
```

Where:
- `X^{i,j}` represents node at level `i` and depth `j`
- `H` is the convolution operation
- `Up` is the upsampling operation
- `[·,·]` denotes concatenation

### 4. Semantic Enhancement Module

#### CLIP-Guided Graph Neural Network

The CLIP-GNN leverages pre-trained CLIP embeddings for semantic understanding.

**Category Descriptions:**

```python
# Simple descriptions (for baseline)
SIMPLE_CATEGORIES = [
    "built-up area", "farmland", "forest", 
    "meadow", "water", "other land"
]

# Detailed descriptions (for enhanced performance)
DETAILED_CATEGORIES = [
    "urban built-up area with buildings and infrastructure",
    "agricultural farmland with crops and cultivation",
    "dense forest area with trees and vegetation",
    "natural meadow and grassland areas",
    "water bodies including rivers and lakes",
    "other miscellaneous land types"
]
```

**GNN Architecture:**

```python
class CLIPGuidedGNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        # CLIP text encoder
        self.clip_model = clip.load("ViT-B/32")
        
        # Graph construction
        self.node_embeddings = nn.Embedding(num_classes, hidden_dim)
        self.edge_weights = nn.Parameter(torch.randn(num_classes, num_classes))
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) 
            for _ in range(3)
        ])
```

**Benefits:**
- **Semantic Consistency**: CLIP embeddings provide rich semantic priors
- **Few-shot Learning**: Better generalization to unseen categories
- **Hierarchical Reasoning**: Graph structure enforces categorical relationships

#### Adaptive Semantic GNN

For scenarios without CLIP, an adaptive GNN learns semantic relationships:

```python
class AdaptiveGNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        # Learnable node embeddings
        self.node_embeddings = nn.Embedding(num_classes, hidden_dim)
        
        # Adaptive edge learning
        self.edge_predictor = EdgePredictor(hidden_dim)
        
        # Message passing
        self.message_passing = MessagePassing(hidden_dim)
```

### 5. Multi-Level Prediction Heads

Separate prediction heads for different hierarchy levels.

#### Level-2 Prediction Head

```python
class Level2Head(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
```

#### Level-3 Prediction Head

```python
class Level3Head(nn.Module):
    def __init__(self, in_channels, num_classes=16):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(256, num_classes, 1)
        )
```

## Loss Functions

### 1. Hierarchical Consistency Loss

Ensures logical consistency between prediction levels.

```python
class HierarchyConsistencyLoss(nn.Module):
    def __init__(self, map_3_to_2, weight=0.5):
        self.weight = weight
        self.mapping = self._create_mapping_tensor(map_3_to_2)
        
    def forward(self, level2_pred, level3_pred):
        # Get level-3 predictions
        level3_indices = level3_pred.argmax(dim=1)
        
        # Map to level-2 space
        mapped_level2 = self.mapping[level3_indices]
        
        # Compute consistency loss
        consistency_loss = F.cross_entropy(
            level2_pred, mapped_level2, reduction='mean'
        )
        
        return self.weight * consistency_loss
```

### 2. Combined Loss Function

```python
total_loss = (
    α · CrossEntropyLoss(pred_level2, target_level2) +
    β · CrossEntropyLoss(pred_level3, target_level3) +
    γ · HierarchyConsistencyLoss(pred_level2, pred_level3)
)
```

Where α=1.0, β=1.0, γ=0.5 by default.

## Training Strategy

### 1. Multi-Scale Training

```python
# Random scale augmentation
scales = [0.8, 0.9, 1.0, 1.1, 1.2]
scale = random.choice(scales)
image = F.interpolate(image, scale_factor=scale)
```

### 2. Mixed Precision Training

```python
# Using PyTorch AMP
scaler = GradScaler()

with autocast():
    pred_l2, pred_l3 = model(image, mask)
    loss = compute_loss(pred_l2, pred_l3, target_l2, target_l3)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Learning Rate Scheduling

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-4,
    epochs=85,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4
)
```

## Inference Optimizations

### 1. Test-Time Augmentation (TTA)

```python
def tta_inference(model, image, mask):
    predictions = []
    
    # Original
    pred = model(image, mask)
    predictions.append(pred)
    
    # Horizontal flip
    pred = model(torch.flip(image, [3]), torch.flip(mask, [3]))
    pred = [torch.flip(p, [3]) for p in pred]
    predictions.append(pred)
    
    # Vertical flip
    pred = model(torch.flip(image, [2]), torch.flip(mask, [2]))
    pred = [torch.flip(p, [2]) for p in pred]
    predictions.append(pred)
    
    # Average predictions
    final_pred = [torch.mean(torch.stack(preds), dim=0) 
                  for preds in zip(*predictions)]
    
    return final_pred
```

### 2. Model Quantization

```python
# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
```

### 3. TensorRT Optimization

```python
# Convert to TensorRT (requires torch2trt)
from torch2trt import torch2trt

model_trt = torch2trt(
    model, 
    [example_image, example_mask],
    fp16_mode=True,
    max_batch_size=8
)
```

## Performance Analysis

### Model Complexity

| Component | Parameters | FLOPs | Memory (MB) |
|-----------|------------|-------|-------------|
| RGB Encoder | 5.3M | 12.4G | 145 |
| Mask Encoder | 0.8M | 1.2G | 18 |
| BiCAF Fusion | 2.1M | 3.8G | 32 |
| UNet++ Decoder | 28.2M | 45.6G | 285 |
| Prediction Heads | 1.8M | 2.1G | 24 |
| **Total** | **38.2M** | **65.1G** | **504** |

### Computational Efficiency

| Input Size | Inference Time | Memory Usage | Throughput |
|------------|----------------|--------------|------------|
| 512×512 | 45ms | 2.1GB | 22.2 FPS |
| 768×768 | 89ms | 4.8GB | 11.2 FPS |
| 1024×1024 | 156ms | 8.4GB | 6.4 FPS |

*Measured on RTX 3080 with mixed precision*

## Ablation Studies

### Component Contributions

| Configuration | Level-2 mIoU | Level-3 mIoU | ΔmIoU |
|---------------|--------------|--------------|-------|
| Baseline (UNet++) | 74.8% | 67.9% | - |
| + Dual Encoder | 75.6% | 68.7% | +0.9% |
| + BiCAF Fusion | 76.8% | 69.8% | +2.1% |
| + CLIP-GNN | 77.9% | 70.9% | +3.1% |
| + Hierarchy Loss | 78.5% | 71.2% | +3.8% |

### BiCAF Reduction Factor Analysis

| Reduction Factor | mIoU | Parameters | Inference Time |
|------------------|------|------------|----------------|
| 4 | 78.7% | 42.1M | 52ms |
| 8 | 78.5% | 38.2M | 45ms |
| 16 | 77.8% | 36.4M | 41ms |
| 32 | 76.9% | 35.2M | 38ms |

## Future Improvements

### Planned Enhancements

1. **Transformer Integration**: Incorporate vision transformers for global context
2. **Multi-Modal Fusion**: Support for SAR and hyperspectral data
3. **Temporal Modeling**: Time-series remote sensing analysis
4. **Uncertainty Quantification**: Bayesian neural networks for confidence estimation
5. **Federated Learning**: Distributed training across institutions

### Research Directions

1. **Self-Supervised Learning**: Pre-training on unlabeled satellite imagery
2. **Domain Adaptation**: Cross-region and cross-sensor generalization
3. **Active Learning**: Intelligent annotation for label-efficient learning
4. **Neural Architecture Search**: Automated architecture optimization

## References

1. Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA 2018.
2. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
3. Tan, M., Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.
4. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
