# -*- coding: utf-8 -*-
"""
dataProcess.py

包含数据加载、划分及评估指标计算的工具
"""

import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def split_train_val(image_paths, primary_paths, secondary_paths, tertiary_paths,
                    val_index, n_splits=5, random_seed=42):
    """
    将数据按 n_splits 分成训练/验证集，并返回第 val_index 折的路径列表
    
    参数:
        image_paths: 原始影像路径列表
        primary_paths: 一级类标签路径列表
        secondary_paths: 二级类标签路径列表
        tertiary_paths: 三级类标签路径列表
        val_index: 当前验证折索引
        n_splits: 总折数
        random_seed: 随机种子
    
    返回:
        训练集和验证集的路径列表元组
    """
    assert len(image_paths) == len(primary_paths) == len(secondary_paths) == len(tertiary_paths)
    indices = list(range(len(image_paths)))
    random.seed(random_seed)
    random.shuffle(indices)

    fold_size = len(indices) // n_splits
    val_start = val_index * fold_size
    # 最后一折包含所有剩余样本
    val_end = len(indices) if val_index == n_splits - 1 else val_start + fold_size

    val_idx = indices[val_start:val_end]
    train_idx = indices[:val_start] + indices[val_end:]

    train_image_paths     = [image_paths[i]     for i in train_idx]
    train_primary_paths   = [primary_paths[i]   for i in train_idx]
    train_secondary_paths = [secondary_paths[i] for i in train_idx]
    train_tertiary_paths  = [tertiary_paths[i]  for i in train_idx]

    val_image_paths       = [image_paths[i]     for i in val_idx]
    val_primary_paths     = [primary_paths[i]   for i in val_idx]
    val_secondary_paths   = [secondary_paths[i] for i in val_idx]
    val_tertiary_paths    = [tertiary_paths[i]  for i in val_idx]

    return (train_image_paths, train_primary_paths, train_secondary_paths, train_tertiary_paths,
            val_image_paths,   val_primary_paths,   val_secondary_paths,   val_tertiary_paths)

class OurDataset(Dataset):
    """
    Dataset 同时加载：
      - 原始遥感影像 (3 通道 RGB)
      - 一级类标签（水体二值图，1 通道）
      - 二级类标签（多类别灰度图）
      - 三级类标签（更细粒度的多类别灰度图）
    """
    def __init__(self, image_paths, primary_paths, secondary_paths, tertiary_paths, mode):
        self.image_paths     = image_paths
        self.primary_paths   = primary_paths
        self.secondary_paths = secondary_paths
        self.tertiary_paths  = tertiary_paths  # 新增三级标签路径
        self.mode            = mode
        # ImageNet 预处理：归一化到 0-1 并做均值/方差归一化
        self.normalize = get_preprocessing_fn('efficientnet-b0', 'imagenet')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 1. 读取影像和标签
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 读取一级类标签 (二值水体图)
        pm = cv2.imread(self.primary_paths[index], cv2.IMREAD_GRAYSCALE)
        pm = (pm > 0).astype(np.uint8)            # 强制 0/1
        
        # 3. 读取二级类标签 (多类别灰度图)
        sm = cv2.imread(self.secondary_paths[index], cv2.IMREAD_GRAYSCALE)
        sm[sm == 255] = 0                         # 将 ignore 映射为 0
        
        # 4. 读取三级类标签 (更细粒度的多类别灰度图)
        tm = cv2.imread(self.tertiary_paths[index], cv2.IMREAD_GRAYSCALE)
        tm[tm == 255] = 0                         # 将 ignore 映射为 0
        
        # 5. 训练模式下应用数据增强（概率为80%）
        if self.mode == 'train' and np.random.random() < 0.8:
            # 导入数据增强函数
            from DataAugmentation import apply_data_augmentation
            # 应用数据增强
            img, pm, sm, tm = apply_data_augmentation(img, pm, sm, tm, apply_hsv=True)
        
        # 6. 预处理和转换为张量
        img = self.normalize(img)                        # float32 H×W×C
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # C×H×W
        pm = torch.from_numpy(pm).unsqueeze(0).float()  # 1×H×W
        sm = torch.from_numpy(sm).long()                # H×W
        tm = torch.from_numpy(tm).long()                # H×W

        return img, pm, sm, tm

def get_dataloader(image_paths, primary_paths, secondary_paths, tertiary_paths,
                   mode, batch_size, shuffle, num_workers,
                   pin_memory=True):
    """
    返回带有 OurDataset 的 DataLoader
    
    参数:
        image_paths: 原始影像路径列表
        primary_paths: 一级类标签路径列表
        secondary_paths: 二级类标签路径列表
        tertiary_paths: 三级类标签路径列表
        mode: 数据集模式 ('train' 或 'val')
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否将数据固定在内存中
    
    返回:
        配置好的 DataLoader
    """
    ds = OurDataset(image_paths, primary_paths, secondary_paths, tertiary_paths, mode)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      # persistent_workers=True,
                      # prefetch_factor=2,
                      pin_memory=pin_memory)

def cal_iou(pred, mask, num_classes=9):  # 更新为林地数据集的类别数
    """
    计算单张图像每个类别的 IoU
    pred, mask: H×W 的 numpy 数组
    返回 shape=(num_classes,) 的 numpy 数组
    """
    iou_list = []
    eps = 1e-6
    for cls in range(num_classes):
        p = (pred == cls)
        t = (mask == cls)
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        iou_list.append(inter / (union + eps))
    return np.array(iou_list)

def cal_val_iou(model, loader, num_classes=9):  # 更新为林地数据集的类别数
    """
    在验证集上计算 IoU。
    返回一个列表，每个元素是单张图像的 num_classes 维 IoU 向量。
    
    注意：此函数仅计算二级标签的 IoU，用于兼容旧代码。
    新代码应使用 cal_val_iou_multi_level 函数。
    """
    model.eval()
    all_ious = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for img, pm, sm, _ in loader:  # 忽略三级标签
            img = img.to(device)
            pm  = pm.to(device)
            logits = model(img, pm)           # 注意：model 需接收 (img, pm)
            # 如果模型返回元组，取第一个元素（二级标签预测）
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1).cpu().numpy()
            gts   = sm.numpy()

            for p, g in zip(preds, gts):
                all_ious.append(cal_iou(p, g, num_classes))

    return all_ious

def cal_val_iou_multi_level(model, loader, num_classes_level2=9, num_classes_level3=18):  # 更新为林地数据集的类别数
    """
    在验证集上计算二级和三级标签的 IoU。
    
    参数:
        model: 待评估的模型
        loader: 数据加载器
        num_classes_level2: 二级类别数
        num_classes_level3: 三级类别数
    
    返回:
        两个列表的元组，分别包含二级和三级标签的 IoU 向量
    """
    model.eval()
    level2_ious = []
    level3_ious = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for img, pm, sm, tm in loader:
            img = img.to(device)
            pm  = pm.to(device)
            outputs = model(img, pm)  # 模型应返回 (level2_logits, level3_logits) 元组
            
            # 处理模型输出
            if isinstance(outputs, tuple) and len(outputs) == 2:
                level2_logits, level3_logits = outputs
            else:
                # 兼容旧模型，只有一个输出
                level2_logits = outputs
                level3_logits = None
            
            # 二级标签预测和评估
            level2_preds = level2_logits.argmax(dim=1).cpu().numpy()
            level2_gts = sm.numpy()
            for p, g in zip(level2_preds, level2_gts):
                level2_ious.append(cal_iou(p, g, num_classes_level2))
            
            # 三级标签预测和评估（如果有）
            if level3_logits is not None:
                level3_preds = level3_logits.argmax(dim=1).cpu().numpy()
                level3_gts = tm.numpy()
                for p, g in zip(level3_preds, level3_gts):
                    level3_ious.append(cal_iou(p, g, num_classes_level3))

    return level2_ious, level3_ious
