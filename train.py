# -*- coding: utf-8 -*-
"""
train.py

训练脚本：5折交叉验证 + DualEncoderUNet++ + BiCAF融合 + 自定义损失

主要特性：
- 双分支UNet++：RGB图像分支 + 简化掩码分支
- BiCAF融合：Bidirectional Cross-Attention Fusion 双向交叉注意力融合
- 多级预测：同时预测二级和三级土地利用类别
- CLIP增强：CLIP引导的语义图神经网络增强细粒度分类
- 层次一致性：确保三级预测与二级预测的层次关系
- 混合精度训练：提升训练效率和显存利用率
- 5折交叉验证：充分利用数据，提升模型泛化性能

使用 dataProcess.py 和 model.py
"""
import os
import glob
import time
import random
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度训练相关模块

from dataProcess import split_train_val, get_dataloader, cal_val_iou, cal_val_iou_multi_level
from model import DualEncoderUNetPP, HierarchyConsistencyLoss

# -------------------- 配置 --------------------
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True
# torch.backends.cudnn.enabled = True      # 默认即为 True，可显式开启
# torch.backends.cudnn.deterministic = False

# 检查CUDA环境并输出调试信息
print(f"调试信息 - torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"调试信息 - torch.version.cuda: {torch.version.cuda}")
print(f"调试信息 - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
if torch.cuda.is_available():
    print(f"调试信息 - torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"调试信息 - GPU {i}: {torch.cuda.get_device_name(i)}")

# 超参数
EPOCHS       = 85         # 减少总训练轮次，专注于50个epoch内达到最优
BATCH_SIZE   = 14         # 增大批量大小，提高训练效率
LR           = 2e-4       # 降低学习率，防止梯度爆炸
WEIGHT_DECAY = 5e-5       # 增强正则化，防止过拟合
FOLDS        = 5
EARLY_STOP   = 15          # 增加早停轮次，给模型更多恢复机会
PCT_START     = 0.3       # 增加OneCycleLR预热比例，延长预热阶段
CLASS_WEIGHTS_LEVEL2 = [1.0] * 6  # 二级类别权重（GID数据集：0背景 + 5土地利用大类 = 6类）
CLASS_WEIGHTS_LEVEL3 = [1.0] * 16 # 三级类别权重（GID数据集：0背景 + 15土地利用子类 = 16类）
USE_AMP      = True      # 启用混合精度训练，提升训练效率
MAX_GRAD_NORM = 1.0       # 添加梯度裁剪参数

# 损失权重 - 降低权重避免损失过大
LEVEL2_WEIGHT = 0.7      # 降低二级标签损失权重
LEVEL3_WEIGHT = 0.7      # 降低三级标签损失权重
HIERARCHY_WEIGHT = 0.3   # 降低层次一致性损失权重0.3

# ================== BiCAF融合配置 ==================
# BiCAF (Bidirectional Cross-Attention Fusion) 双向交叉注意力融合配置
USE_BICAF_FUSION = True           # 是否使用BiCAF融合（False则使用简单相加）
BICAF_REDUCTION_FACTOR = 8        # BiCAF通道缩减倍数（8/16/32，越大参数越少）
USE_MULTI_SCALE_FUSION = True    # 是否在深层启用多尺度融合（增加少量参数但可能提升精度）

# GNN配置
USE_ADAPTIVE_GNN = False          # 使用基础自适应图神经网络
USE_CLIP_GNN = False              # 使用CLIP引导的语义图神经网络

# 融合策略说明：
# - BiCAF融合：双向交叉注意力，参数少但效果好，推荐使用
# - 简单相加：最基础的融合方式，无额外参数
# - reduction_factor控制参数量：8(更多参数) < 16(平衡) < 32(更少参数)
print(f"融合策略: {'BiCAF双向交叉注意力融合' if USE_BICAF_FUSION else '简单相加融合'}")
if USE_BICAF_FUSION:
    print(f"BiCAF配置: reduction_factor={BICAF_REDUCTION_FACTOR}, multi_scale={USE_MULTI_SCALE_FUSION}")
print(f"GNN配置: adaptive_gnn={USE_ADAPTIVE_GNN}, clip_gnn={USE_CLIP_GNN}")
# ===================================================

# 数据路径，请根据实际情况修改
IMAGE_DIR     = r"./data/train/image"
PRIMARY_DIR   = r"./data/train/label1"
SECONDARY_DIR = r"./data/train/label2"
TERTIARY_DIR  = r"./data/train/label3"  # 三级标签路径
MODEL_DIR     = r"./model"
LOG_DIR      = r"./logs"  # 日志保存路径
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)  # 创建日志目录

# 配置日志记录器
def setup_logger(fold_idx):
    """
    配置训练日志记录器
    
    参数:
        fold_idx: 当前训练的折数索引
        
    返回:
        配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger(f'fold{fold_idx}')
    logger.setLevel(logging.INFO)
    
    # 清除之前的handlers，避免重复记录
    if logger.handlers:
        logger.handlers = []
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f'train_fold{fold_idx}_{timestamp}.log')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 收集文件列表并排序
image_paths     = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
primary_paths   = sorted(glob.glob(os.path.join(PRIMARY_DIR, "*.tif")))
secondary_paths = sorted(glob.glob(os.path.join(SECONDARY_DIR, "*.tif")))
tertiary_paths  = sorted(glob.glob(os.path.join(TERTIARY_DIR, "*.tif")))
assert len(image_paths)==len(primary_paths)==len(secondary_paths)==len(tertiary_paths), "路径长度不一致"

# # 每10个样本取1个，以保证样本分布的合理性
# image_paths = image_paths[::10]
# primary_paths = primary_paths[::10]
# secondary_paths = secondary_paths[::10]
# tertiary_paths = tertiary_paths[::10]

# num_samples_loaded = len(image_paths)

# logger_main = logging.getLogger('main_script') # Use a general logger for this info if setup_logger is fold-specific
# # If no general logger, this specific log might need to be inside train_fold or a new general logger setup
# # For now, assuming a way to log this information globally or we can adapt it.
# # A simple print might also work if logging setup is complex here.
# print(f"调试信息：每10个样本取1个，共加载 {num_samples_loaded} 个样本进行训练和验证。")

# 自定义损失：Tversky + 加权交叉熵 + Focal
def check_model_stability(model, logger):
    """
    检查模型参数的数值稳定性
    
    参数:
        model: 要检查的模型
        logger: 日志记录器
    
    返回:
        bool: True表示模型稳定，False表示存在异常
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
                logger.warning(f"参数 {name} 梯度异常: {grad_norm}")
                return False
        
        param_norm = param.data.norm()
        if torch.isnan(param_norm) or torch.isinf(param_norm):
            logger.warning(f"参数 {name} 数值异常: {param_norm}")
            return False
    
    return True

class CombinedLoss(nn.Module):
    """
    组合损失函数：Tversky + 加权交叉熵 + Focal
    支持同时计算二级和三级标签的损失
    """
    def __init__(self,
                 weight_tversky=1.0,
                 weight_ce=1.0,
                 weight_focal=0.5,
                 class_weights_level2=None,
                 class_weights_level3=None,
                 level2_weight=1.0,
                 level3_weight=1.0):
        super().__init__()
        from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
        
        # 二级标签损失
        self.tversky_level2 = TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
        cw2 = torch.tensor(class_weights_level2, dtype=torch.float, device=DEVICE)
        self.ce_level2 = nn.CrossEntropyLoss(weight=cw2, ignore_index=255)
        self.focal_level2 = FocalLoss(mode='multiclass', gamma=2.0)
        
        # 三级标签损失
        self.tversky_level3 = TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
        cw3 = torch.tensor(class_weights_level3, dtype=torch.float, device=DEVICE)
        self.ce_level3 = nn.CrossEntropyLoss(weight=cw3, ignore_index=255)
        self.focal_level3 = FocalLoss(mode='multiclass', gamma=2.0)
        
        # 损失权重
        self.w_t = weight_tversky
        self.w_ce = weight_ce
        self.w_f = weight_focal
        self.level2_weight = level2_weight
        self.level3_weight = level3_weight

    def forward(self, inputs, targets):
        """
        计算组合损失
        
        参数:
            inputs: 模型输出的预测结果元组 (level2_pred, level3_pred)
            targets: 目标标签元组 (level2_target, level3_target)
        
        返回:
            组合损失值
        """
        # 解包输入和目标
        level2_pred, level3_pred = inputs
        level2_target, level3_target = targets
        
        # 二级标签损失
        loss_t2 = self.tversky_level2(level2_pred, level2_target)
        loss_ce2 = self.ce_level2(level2_pred, level2_target)
        loss_f2 = self.focal_level2(level2_pred, level2_target)
        loss_level2 = self.w_t*loss_t2 + self.w_ce*loss_ce2 + self.w_f*loss_f2
        
        # 三级标签损失
        loss_t3 = self.tversky_level3(level3_pred, level3_target)
        loss_ce3 = self.ce_level3(level3_pred, level3_target)
        loss_f3 = self.focal_level3(level3_pred, level3_target)
        loss_level3 = self.w_t*loss_t3 + self.w_ce*loss_ce3 + self.w_f*loss_f3
        
        # 总损失
        return self.level2_weight * loss_level2 + self.level3_weight * loss_level3


def train_fold(fold_idx):
    # 设置日志记录器
    logger = setup_logger(fold_idx)
    logger.info(f"===== Training fold {fold_idx} =====")
    logger.info(f"混合精度训练状态: {'启用' if USE_AMP else '禁用'}")
    
    # 划分
    (train_imgs, train_pm, train_sm, train_tm,
     val_imgs,   val_pm,   val_sm,   val_tm) = split_train_val(
        image_paths, primary_paths, secondary_paths, tertiary_paths,
        val_index=fold_idx, n_splits=FOLDS
    )
    # DataLoader
    train_loader = get_dataloader(
        train_imgs, train_pm, train_sm, train_tm,
        mode='train', batch_size=BATCH_SIZE,
        shuffle=True, num_workers=16
    )
    # 记录训练/验证集样本数量
    logger.info(f"Train samples: {len(train_imgs)},  Val samples: {len(val_imgs)}")
    val_loader   = get_dataloader(
        val_imgs, val_pm, val_sm, val_tm,
        mode='val', batch_size=BATCH_SIZE,
        shuffle=False, num_workers=16
    )

    # 模型
    model = DualEncoderUNetPP(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        num_classes_level2=6,  # GID数据集：6类（5土地利用大类+背景）
        num_classes_level3=16, # GID数据集：16类（15土地利用子类+背景）
        # BiCAF融合配置
        use_cross_attention_fusion=USE_BICAF_FUSION,        # 使用BiCAF融合
        cross_attention_reduction=BICAF_REDUCTION_FACTOR,   # 通道缩减倍数
        use_multi_scale_fusion=USE_MULTI_SCALE_FUSION,      # 多尺度融合
        # GNN配置
        use_adaptive_gnn=USE_ADAPTIVE_GNN,                  # 基础自适应GNN
        use_clip_gnn=USE_CLIP_GNN                          # CLIP引导GNN
    )
    
    # 确保CLIP模型使用正确的设备
    if hasattr(model, 'clip_gnn') and model.clip_gnn is not None:
        model.clip_gnn._device_override = torch.device(DEVICE)
        # 重新初始化CLIP模型以使用正确的设备
        if hasattr(model.clip_gnn, 'clip_model') and model.clip_gnn.clip_model is not None:
            try:
                import clip
                model.clip_gnn.clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
                model.clip_gnn.clip_model.eval()
                for param in model.clip_gnn.clip_model.parameters():
                    param.requires_grad = False
                model.clip_gnn.device = torch.device(DEVICE)
                logger.info(f"CLIP模型已重新初始化到设备: {DEVICE}")
            except Exception as e:
                logger.warning(f"重新初始化CLIP模型失败: {e}")
    
    model = model.to(DEVICE)
    
    # 检查模型参数是否正常初始化
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.error(f"模型参数 {name} 包含NaN或Inf值！")
            raise ValueError(f"模型初始化异常: {name}")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计BiCAF融合模块参数量
    bicaf_params = 0
    if USE_BICAF_FUSION and hasattr(model, 'fusion_modules'):
        for module in model.fusion_modules:
            if not isinstance(module, torch.nn.Identity):
                bicaf_params += sum(p.numel() for p in module.parameters())
    
    logger.info(f"=== 模型配置信息 ===")
    logger.info(f"特征融合方式: {'BiCAF双向交叉注意力融合' if USE_BICAF_FUSION else '简单相加融合'}")
    if USE_BICAF_FUSION:
        logger.info(f"BiCAF配置: reduction_factor={BICAF_REDUCTION_FACTOR}, multi_scale={USE_MULTI_SCALE_FUSION}")
        logger.info(f"BiCAF模块参数量: {bicaf_params:,} ({bicaf_params/total_params*100:.3f}%)")
    logger.info(f"GNN配置: adaptive_gnn={USE_ADAPTIVE_GNN}, clip_gnn={USE_CLIP_GNN}")
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    logger.info(f"===================")
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=PCT_START,
        anneal_strategy='cos', div_factor=25,
        final_div_factor=10000
    )
    
    # 初始化梯度缩放器用于混合精度训练
    scaler = GradScaler(enabled=USE_AMP)

    # 组合损失函数
    loss_fn = CombinedLoss(
        weight_tversky=1.0,
        weight_ce=1.0,
        weight_focal=0.5,
        class_weights_level2=CLASS_WEIGHTS_LEVEL2,
        class_weights_level3=CLASS_WEIGHTS_LEVEL3,
        level2_weight=LEVEL2_WEIGHT,
        level3_weight=LEVEL3_WEIGHT
    ).to(DEVICE)
    
    # 层次一致性损失函数
    hierarchy_loss_fn = HierarchyConsistencyLoss(weight=HIERARCHY_WEIGHT).to(DEVICE)

    # 训练配置总结
    logger.info(f"")
    logger.info(f"=== 训练配置总结 (Fold {fold_idx}) ===")
    logger.info(f"数据配置:")
    logger.info(f"  - 训练样本: {len(train_loader.dataset)} 个")
    logger.info(f"  - 验证样本: {len(val_loader.dataset)} 个")
    logger.info(f"  - 批量大小: {BATCH_SIZE}")
    logger.info(f"训练超参数:")
    logger.info(f"  - 训练轮数: {EPOCHS}")
    logger.info(f"  - 初始学习率: {LR}")
    logger.info(f"  - 权重衰减: {WEIGHT_DECAY}")
    logger.info(f"  - 早停轮数: {EARLY_STOP}")
    logger.info(f"  - 混合精度: {'启用' if USE_AMP else '禁用'}")
    logger.info(f"损失配置:")
    logger.info(f"  - 二级标签权重: {LEVEL2_WEIGHT}")
    logger.info(f"  - 三级标签权重: {LEVEL3_WEIGHT}")
    logger.info(f"  - 层次一致性权重: {HIERARCHY_WEIGHT}")
    logger.info(f"模型架构:")
    logger.info(f"  - 编码器: efficientnet-b0")
    logger.info(f"  - 解码器: UNet++")
    logger.info(f"  - 特征融合: {'BiCAF双向交叉注意力' if USE_BICAF_FUSION else '简单相加'}")
    if USE_BICAF_FUSION:
        logger.info(f"    * 通道缩减倍数: {BICAF_REDUCTION_FACTOR}")
        logger.info(f"    * 多尺度融合: {'启用' if USE_MULTI_SCALE_FUSION else '禁用'}")
    logger.info(f"  - 语义增强: {'CLIP-GNN' if USE_CLIP_GNN else ('Adaptive-GNN' if USE_ADAPTIVE_GNN else '无')}")
    logger.info(f"=====================================")
    logger.info(f"")

    best_miou = 0.0
    no_improve = 0
    
    # 用于绘图的数据收集
    epochs_list = []
    miou_list = []
    loss_list = []

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        train_losses = []
        for img, pm, sm, tm in train_loader:
            img, pm = img.to(DEVICE), pm.to(DEVICE)
            sm, tm = sm.to(DEVICE), tm.to(DEVICE)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast(enabled=USE_AMP):
                # 获取预测结果和CLIP损失
                outputs = model(img, pm, return_clip_loss=True)
                if len(outputs) == 3:
                    level2_pred, level3_pred, clip_loss = outputs
                else:
                    level2_pred, level3_pred = outputs
                    clip_loss = 0.0
                
                # 计算分割损失
                seg_loss = loss_fn((level2_pred, level3_pred), (sm, tm))
                # 计算层次一致性损失
                hierarchy_loss = hierarchy_loss_fn(level2_pred, level3_pred)
                # 总损失（包含CLIP蒸馏损失）
                loss = seg_loss + hierarchy_loss + clip_loss
                
            # 检查损失是否为nan或inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到异常损失值: {loss.item()}, 跳过此批次")
                continue
            
            # 使用梯度缩放器进行反向传播和优化
            if USE_AMP:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
            scheduler.step()
            train_losses.append(loss.item())

        # 验证
        with torch.no_grad():
            with autocast(enabled=USE_AMP):
                level2_ious, level3_ious = cal_val_iou_multi_level(
                    model, val_loader, 
                    num_classes_level2=6,  # GID数据集：6类（5土地利用大类+背景）
                    num_classes_level3=16  # GID数据集：16类（15土地利用子类+背景）
                )
        
        # 计算二级标签的 IoU
        level2_per_class_iou = np.stack(level2_ious, axis=0).mean(axis=0)  # [num_classes_level2]
        level2_mean_miou = level2_per_class_iou.mean()  # 标量
        
        # 计算三级标签的 IoU
        level3_per_class_iou = np.stack(level3_ious, axis=0).mean(axis=0)  # [num_classes_level3]
        level3_mean_miou = level3_per_class_iou.mean()  # 标量
        
        # 计算总体 mIoU (二级和三级的平均)
        mean_miou = (level2_mean_miou + level3_mean_miou) / 2
        avg_loss = np.mean(train_losses)
        
        # 检查训练指标是否异常
        if np.isnan(avg_loss) or np.isnan(mean_miou):
            logger.error(f"训练指标异常 - avg_loss: {avg_loss}, mean_miou: {mean_miou}")
            logger.error("训练异常终止")
            break
        
        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 收集绘图数据
        epochs_list.append(epoch)
        miou_list.append(mean_miou)
        loss_list.append(avg_loss)

        # 格式化每类 IoU 打印
        level2_per_class_str = ", ".join([f"{iou:.4f}" for iou in level2_per_class_iou])
        level3_per_class_str = ", ".join([f"{iou:.4f}" for iou in level3_per_class_iou])
        
        # 记录训练结果到日志
        log_message = (
            f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}, lr: {current_lr:.2e}\n"
            f"Level2 mIoU: {level2_mean_miou:.4f}, per-class IoU: [{level2_per_class_str}]\n"
            f"Level3 mIoU: {level3_mean_miou:.4f}, per-class IoU: [{level3_per_class_str}]\n"
            f"Overall mIoU: {mean_miou:.4f}, time: {(time.time()-t0)/60:.2f}m"
        )
        logger.info(log_message)
        
        # 保存 & 早停
        if mean_miou > best_miou:
            best_miou = mean_miou
            no_improve = 0
            # 根据融合方式生成模型文件名
            fusion_suffix = f"bicaf_r{BICAF_REDUCTION_FACTOR}" if USE_BICAF_FUSION else "simple"
            gnn_suffix = "clipgnn" if USE_CLIP_GNN else ("adaptivegnn" if USE_ADAPTIVE_GNN else "nognn")
            save_path = os.path.join(MODEL_DIR, f"fold{fold_idx}_epoch{epoch}_best_{fusion_suffix}_{gnn_suffix}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                logger.info("Early stopping.")
                break
    
    # 绘制训练过程图表
    plot_training_curves(epochs_list, miou_list, loss_list, fold_idx)

def plot_training_curves(epochs, mious, losses, fold_idx):
    """
    绘制训练过程中的mIoU和loss曲线
    
    参数:
        epochs: epoch列表
        mious: 对应的mIoU值列表
        losses: 对应的loss值列表
        fold_idx: 当前训练的折数索引
    """
    # 创建图表目录
    plot_dir = os.path.join(LOG_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(12, 5))
    
    # 绘制mIoU曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mious, 'b-', marker='o')
    plt.title(f'Fold {fold_idx} - Overall mIoU vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Overall mIoU')
    plt.grid(True)
    
    # 绘制Loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, 'r-', marker='o')
    plt.title(f'Fold {fold_idx} - Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(plot_dir, f'training_curves_fold{fold_idx}_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # 记录图表保存路径
    logger = logging.getLogger(f'fold{fold_idx}')
    logger.info(f"训练曲线图表已保存至: {plot_path}")

if __name__ == '__main__':
    print("=" * 60)
    print("BiCAF双向交叉注意力融合训练脚本")
    print("=" * 60)
    print(f"当前配置:")
    print(f"  - 特征融合: {'BiCAF双向交叉注意力' if USE_BICAF_FUSION else '简单相加'}")
    if USE_BICAF_FUSION:
        print(f"    * 通道缩减倍数: {BICAF_REDUCTION_FACTOR}")
        print(f"    * 多尺度融合: {'启用' if USE_MULTI_SCALE_FUSION else '禁用'}")
    print(f"  - 语义增强: {'CLIP-GNN' if USE_CLIP_GNN else ('Adaptive-GNN' if USE_ADAPTIVE_GNN else '无')}")
    print(f"  - 训练设备: {DEVICE}")
    print(f"  - 混合精度: {'启用' if USE_AMP else '禁用'}")
    print()
    print("BiCAF配置建议:")
    print("  - reduction_factor=8:  更多参数，可能获得更好精度")
    print("  - reduction_factor=16: 平衡选择，推荐使用")
    print("  - reduction_factor=32: 更少参数，训练更快")
    print("  - multi_scale=True:    在深层启用多尺度，略微增加参数但可能提升精度")
    print("=" * 60)
    print()
    
    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 开始5折交叉验证训练
    for fold in range(FOLDS):
        print(f"\n开始训练第 {fold+1}/{FOLDS} 折...")
        train_fold(fold)
        print(f"第 {fold+1} 折训练完成")
    
    print("\n" + "=" * 60)
    print("所有折次训练完成！")
    print("请查看LOG_DIR中的训练日志和图表。")
    print("=" * 60)

