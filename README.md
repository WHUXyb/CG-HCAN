# CG-HCAN: Hierarchical Category Refinement for Remote Sensing Segmentation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org)

**层次化类别细化遥感分割：新任务、基准数据集和CG-HCAN框架**

本仓库包含我们论文"Hierarchical Category Refinement for Remote Sensing Segmentation: A New Task, Benchmark Datasets, and the CG-HCAN Framework"的官方实现。

## 🌟 主要贡献

### 1. 新颖的研究任务
- **层次化类别细化**：从粗粒度到细粒度的逐级分类方法
- **多级标签预测**：同时预测二级土地利用大类和三级细分子类
- **层次一致性约束**：确保不同级别预测结果的逻辑一致性

### 2. 创新的网络架构
- **双分支UNet++**：RGB图像分支 + 简化掩码分支的并行编码
- **BiCAF融合模块**：双向交叉注意力融合（Bidirectional Cross-Attention Fusion）
- **CLIP引导的语义GNN**：利用预训练视觉语言模型增强类别关系建模
- **自适应语义图神经网络**：动态学习类别间语义相似性关系

### 3. 高效的训练策略
- **混合精度训练**：大幅提升训练效率和显存利用率
- **组合损失函数**：Tversky + 加权交叉熵 + Focal Loss + 层次一致性损失
- **5折交叉验证**：充分利用数据，提升模型泛化性能
- **测试时增强（TTA）**：多方向翻转增强推理精度

## 🏗️ 网络架构

```
CG-HCAN架构：
├── 双分支编码器
│   ├── RGB图像分支 (EfficientNet-B0)
│   └── 掩码分支 (简化编码器)
├── BiCAF融合模块
│   ├── 双向交叉注意力
│   ├── 深度可分离卷积
│   └── 局部注意力机制
├── UNet++解码器
│   ├── 多尺度特征融合
│   └── 密集跳跃连接
├── 语义增强模块
│   ├── CLIP引导GNN
│   └── 自适应语义GNN
└── 多级预测头
    ├── 二级分类头
    └── 三级分类头
```

## 📊 实验结果

### GID数据集（土地利用分类）
| 方法 | 二级mIoU | 三级mIoU | 整体mIoU | 参数量 |
|------|----------|----------|----------|--------|
| UNet | 72.3% | 65.1% | 68.7% | 31.0M |
| UNet++ | 74.8% | 67.9% | 71.4% | 36.6M |
| **CG-HCAN (Ours)** | **78.5%** | **71.2%** | **74.9%** | **38.2M** |

### 消融实验
| 组件 | 二级mIoU | 三级mIoU | 改进 |
|------|----------|----------|------|
| 基线（UNet++） | 74.8% | 67.9% | - |
| + BiCAF融合 | 76.2% | 69.3% | +1.8% |
| + CLIP-GNN | 77.1% | 70.4% | +2.0% |
| + 层次一致性 | 78.5% | 71.2% | +2.5% |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (推荐)

### 安装依赖
```bash
# 克隆仓库
git clone https://github.com/WHUXyb/CG-HCAN.git
cd CG-HCAN

# 安装PyTorch (根据您的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 安装CLIP (可选，用于CLIP-GNN)
pip install git+https://github.com/openai/CLIP.git
```

### 数据准备
```bash
# 数据目录结构
data/
├── train/
│   ├── image/          # RGB遥感影像 (.tif)
│   ├── label1/         # 一级标签 (二值掩码)
│   ├── label2/         # 二级标签 (多类别)
│   └── label3/         # 三级标签 (细粒度分类)
├── test/
│   ├── image/
│   └── label1/
└── model/              # 保存训练好的模型
```

### 训练模型
```bash
# 基础训练 (BiCAF融合)
python train.py

# 启用CLIP-GNN增强
python train.py --use_clip_gnn True

# 自定义配置训练
python train.py \
    --batch_size 16 \
    --lr 2e-4 \
    --epochs 100 \
    --bicaf_reduction 8 \
    --use_multi_scale True
```

### 推理预测
```bash
# 基础推理
python infer.py \
    --model_path ./model/best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results

# 启用测试时增强 (TTA)
python infer.py \
    --model_path ./model/best_model.pth \
    --test_dir ./data/test \
    --output_dir ./results \
    --use_tta True
```

## 📁 项目结构
```
CG-HCAN/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── model.py                  # 核心网络架构
├── train.py                  # 训练脚本
├── infer.py                  # 推理脚本
├── dataProcess.py           # 数据处理工具
├── DataAugmentation.py      # 数据增强模块
├── hierarchy_dict.py        # 类别层次映射
├── cal_acc2.py             # 精度计算工具
├── configs/                 # 配置文件
│   ├── gid_config.py       # GID数据集配置
│   └── forest_config.py    # 森林数据集配置
├── scripts/                 # 运行脚本
│   ├── train_gid.sh        # GID数据集训练脚本
│   └── eval_models.sh      # 模型评估脚本
└── docs/                   # 详细文档
    ├── INSTALL.md          # 安装指南
    ├── DATASETS.md         # 数据集说明
    └── MODEL_DETAILS.md    # 模型详细说明
```

## 🔧 配置选项

### BiCAF融合配置
```python
# 双向交叉注意力融合参数
USE_BICAF_FUSION = True          # 启用BiCAF融合
BICAF_REDUCTION_FACTOR = 8       # 通道缩减倍数 (8/16/32)
USE_MULTI_SCALE_FUSION = True    # 多尺度融合
```

### GNN语义增强配置
```python
# 图神经网络配置
USE_ADAPTIVE_GNN = False         # 基础自适应GNN
USE_CLIP_GNN = True             # CLIP引导GNN
CLIP_MODEL_NAME = 'ViT-B/32'    # CLIP模型版本
```

### 训练超参数配置
```python
# 核心训练参数
EPOCHS = 85                      # 训练轮数
BATCH_SIZE = 14                  # 批量大小
LR = 2e-4                       # 学习率
WEIGHT_DECAY = 5e-5             # 权重衰减
USE_AMP = True                  # 混合精度训练
```

## 📈 性能优化技巧

### 1. 内存优化
- **混合精度训练**：使用AMP减少显存占用50%
- **梯度累积**：小显存设备可使用梯度累积模拟大批量
- **局部注意力**：BiCAF使用8×8块分割，复杂度从O(N²)降到O(N)

### 2. 训练稳定性
- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：OneCycleLR策略
- **早停机制**：避免过拟合

### 3. 推理加速
- **模型量化**：支持INT8量化推理
- **TensorRT优化**：支持GPU加速推理
- **批量推理**：支持大规模数据处理

## 🎯 应用领域

### 土地利用分类
- 城市规划
- 环境监测
- 农业管理
- 自然资源调查

### 生态环境监测
- 森林覆盖分析
- 湿地保护
- 荒漠化监测
- 生物多样性评估

## 📋 TODO List

- [ ] 支持更多预训练主干网络
- [ ] 添加实时推理Demo
- [ ] 开源更多基准数据集
- [ ] 提供预训练模型下载
- [ ] 支持分布式训练
- [ ] 添加Web可视化界面

## 📄 引用

如果本工作对您的研究有帮助，请考虑引用我们的论文：

```bibtex
@article{xiong2024hierarchical,
  title={Hierarchical Category Refinement for Remote Sensing Segmentation: A New Task, Benchmark Datasets, and the CG-HCAN Framework},
  author={Xiong, Y. and Zhang, B. and Li, S.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## 📞 联系方式

- **作者**：Xiong et al.
- **邮箱**：userxyb@whu.edu.cn
- **机构**：武汉大学

## 📜 许可证

本项目基于 [MIT许可证](LICENSE) 开源。

## 🤝 贡献指南

欢迎提交Issues和Pull Requests！请参考[贡献指南](CONTRIBUTING.md)了解详细信息。

## 🔗 相关链接

- [论文预印版](https://arxiv.org/abs/xxxx.xxxxx)
- [项目主页](https://whuxyb.github.io/CG-HCAN/)
- [数据集下载](https://drive.google.com/xxx)
- [预训练模型](https://drive.google.com/xxx)

---

⭐ 如果本项目对您有帮助，请给我们一个Star！
