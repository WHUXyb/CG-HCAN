# -*- coding: utf-8 -*-
"""
infer.py

推理脚本：DualEncoderUNet++ + TTA
支持同时预测二级和三级标签（GID土地利用分类）
使用 model.py 和 dataProcess.py 中的预处理函数
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from model import DualEncoderUNetPP
from tqdm import tqdm

# -------------------- 配置 --------------------
DEVICE      = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 2
NUM_WORKERS = 8

# 路径设置（请根据实际路径修改）
IMAGE_DIR     = r"./data/test/image"
PRIMARY_DIR   = r"./data/test/label1"
# 二级标签输出目录
OUTPUT_DIR_LEVEL2    = r"./results/result_level2"
OUTPUT_RGB_DIR_LEVEL2 = r"./results/result_level2_rgb"
# 三级标签输出目录
OUTPUT_DIR_LEVEL3    = r"./results/result_level3"
OUTPUT_RGB_DIR_LEVEL3 = r"./results/result_level3_rgb"
MODEL_PATH    = r"./model/best_model.pth"  # 请修改为实际的模型路径

# 创建输出目录
os.makedirs(OUTPUT_DIR_LEVEL2, exist_ok=True)
os.makedirs(OUTPUT_RGB_DIR_LEVEL2, exist_ok=True)
os.makedirs(OUTPUT_DIR_LEVEL3, exist_ok=True)
os.makedirs(OUTPUT_RGB_DIR_LEVEL3, exist_ok=True)

# 获取文件列表并排序
image_paths   = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
primary_paths = sorted(glob.glob(os.path.join(PRIMARY_DIR, "*.tif")))
assert len(image_paths) == len(primary_paths), "测试影像与一级标签数量不匹配"

# 预处理函数（ImageNet 归一化）
normalize = get_preprocessing_fn('efficientnet-b0', 'imagenet')

# 定义科学论文友好的彩色映射方案
# 参考自 ColorBrewer 和 Scientific Color Maps
# 背景为黑色，其他类别使用高区分度的颜色

# 二级标签颜色映射（6类，包括背景）- GID数据集
COLOR_MAP_LEVEL2 = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (255, 0, 0),    # 建设用地 - 红色
    2: (0, 255, 0),    # 耕地 - 绿色
    3: (0, 255, 255),    # 林地 - 蓝色
    4: (255, 255, 0),  # 草地 - 黄色
    5: (0, 0, 255)    # 水体 - 紫色
}

# 三级标签颜色映射（16类，包括背景）- GID数据集
COLOR_MAP_LEVEL3 = {
    0: (0, 0, 0),         # 背景 - 黑色
    1: (200, 0, 0),     # 建设用地1 - 红色
    2: (250, 0, 150),   # 建设用地2 - 浅红色
    3: (200, 150, 150),   # 建设用地3 - 浅橙色
    4: (250, 150, 150),   # 建设用地4 - 浅黄色
    5: (0, 200, 0),     # 耕地1 - 绿色
    6: (150, 250, 0),    # 耕地2 - 蓝色
    7: (150, 200, 150),   # 耕地3 - 黄色
    8: (200, 0, 200),     # 林地1 - 紫色
    9: (150, 0, 250),   # 林地2 - 蓝色
    10: (150, 150, 250),  # 林地3 - 蓝色
    11: (250, 200, 0),  # 草地1 - 黄色
    12: (200, 200, 0),  # 草地2 - 黄色
    13: (0, 0, 200),   # 水体1 - 蓝色
    14: (0, 150, 200),  # 水体2 - 蓝色
    15: (0, 200, 250)   # 水体3 - 蓝色
}

# 测试集 Dataset
class TestDataset(Dataset):
    def __init__(self, image_paths, primary_paths, normalize):
        self.image_paths   = image_paths
        self.primary_paths = primary_paths
        self.normalize     = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取影像
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # 归一化
        img = self.normalize(img)             # float32 H×W×C
        img = torch.from_numpy(img).permute(2, 0, 1)  # C×H×W

        # 一级二值标签
        pm = cv2.imread(self.primary_paths[idx], cv2.IMREAD_GRAYSCALE)
        pm = (pm > 0).astype(np.uint8)
        pm = torch.from_numpy(pm).unsqueeze(0)  # 1×H×W

        return img.float(), pm.float(), (w, h), self.image_paths[idx]

def create_rgb_image(pred, color_map):
    """
    将类别索引图转换为RGB彩色图
    
    Args:
        pred: 类别索引图
        color_map: 颜色映射字典
        
    Returns:
        rgb_image: RGB彩色图
    """
    h, w = pred.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in color_map.items():
        mask = (pred == class_idx)
        rgb_image[mask] = color
        
    return rgb_image

if __name__ == '__main__':
    # DataLoader
    test_loader = DataLoader(
        TestDataset(image_paths, primary_paths, normalize),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 构建模型并加载权重
    model = DualEncoderUNetPP(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        num_classes_level2=6,   # GID数据集：二级类别数（0背景 + 5土地利用大类）
        num_classes_level3=16,  # GID数据集：三级类别数（0背景 + 15土地利用子类）
                # BiCAF融合配置
        use_cross_attention_fusion=True,        # 使用BiCAF融合
        cross_attention_reduction=8,   # 通道缩减倍数
        use_multi_scale_fusion=False,      # 多尺度融合
        # GNN配置
        use_adaptive_gnn=False,                  # 基础自适应GNN
        use_clip_gnn=False).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"模型加载成功: {MODEL_PATH}")
    print(f"模型配置: 二级类别数={6}, 三级类别数={16}")
    print(f"测试图像数量: {len(image_paths)}")

    # 推理函数
    with torch.no_grad():
        for batch_idx, (imgs, pms, shapes, paths) in enumerate(
                tqdm(test_loader, desc="Inference", total=len(test_loader)), start=1):
            imgs = imgs.to(DEVICE)
            pms  = pms.to(DEVICE)

            # 初始化累加输出（二级和三级标签）
            B, _, H, W = imgs.shape
            logits_sum_level2 = torch.zeros(B, 6, H, W, device=DEVICE)   # 6类（GID二级）
            logits_sum_level3 = torch.zeros(B, 16, H, W, device=DEVICE)  # 16类（GID三级）

            # 原图预测
            level2_logits, level3_logits = model(imgs, pms)
            logits_sum_level2 += level2_logits
            logits_sum_level3 += level3_logits

            # TTA：上下翻转
            imgs_ud = torch.flip(imgs, [2]); pms_ud = torch.flip(pms, [2])
            level2_ud, level3_ud = model(imgs_ud, pms_ud)
            logits_sum_level2 += torch.flip(level2_ud, [2])
            logits_sum_level3 += torch.flip(level3_ud, [2])

            # TTA：左右翻转
            imgs_lr = torch.flip(imgs, [3]); pms_lr = torch.flip(pms, [3])
            level2_lr, level3_lr = model(imgs_lr, pms_lr)
            logits_sum_level2 += torch.flip(level2_lr, [3])
            logits_sum_level3 += torch.flip(level3_lr, [3])

            # 平均
            logits_avg_level2 = logits_sum_level2 / 3.0
            logits_avg_level3 = logits_sum_level3 / 3.0
            
            # 获取预测结果
            preds_level2 = logits_avg_level2.argmax(dim=1).cpu().numpy()  # B×H×W
            preds_level3 = logits_avg_level3.argmax(dim=1).cpu().numpy()  # B×H×W

            # 保存结果
            for i in range(B):
                # 检查shapes[i]格式并安全地解包
                try:
                    if isinstance(shapes[i], tuple):
                        if len(shapes[i]) == 2:
                            w, h = shapes[i]
                        elif len(shapes[i]) == 1:
                            w = shapes[i][0]
                            h = None
                        else:
                            w, h = None, None
                    else:
                        w, h = None, None
                except Exception as e:
                    print(f"警告：无法解析第{i}个样本的shapes: {shapes[i]}，错误: {e}")
                    w, h = None, None
                    
                filename = os.path.basename(paths[i])
                
                # 处理二级标签预测结果
                pred_level2 = preds_level2[i].astype(np.uint8)
                # 保存原始类别索引图（二级标签）
                save_path_level2 = os.path.join(OUTPUT_DIR_LEVEL2, filename)
                cv2.imwrite(save_path_level2, pred_level2)
                # 创建并保存RGB彩色图（二级标签）
                rgb_image_level2 = create_rgb_image(pred_level2, COLOR_MAP_LEVEL2)
                rgb_save_path_level2 = os.path.join(OUTPUT_RGB_DIR_LEVEL2, filename)
                cv2.imwrite(rgb_save_path_level2, cv2.cvtColor(rgb_image_level2, cv2.COLOR_RGB2BGR))
                
                # 处理三级标签预测结果
                pred_level3 = preds_level3[i].astype(np.uint8)
                # 保存原始类别索引图（三级标签）
                save_path_level3 = os.path.join(OUTPUT_DIR_LEVEL3, filename)
                cv2.imwrite(save_path_level3, pred_level3)
                # 创建并保存RGB彩色图（三级标签）
                rgb_image_level3 = create_rgb_image(pred_level3, COLOR_MAP_LEVEL3)
                rgb_save_path_level3 = os.path.join(OUTPUT_RGB_DIR_LEVEL3, filename)
                cv2.imwrite(rgb_save_path_level3, cv2.cvtColor(rgb_image_level3, cv2.COLOR_RGB2BGR))

            print(f"Batch {batch_idx}/{len(test_loader)} done.")

    print("Inference done.")
    print(f"二级标签原始类别索引图已保存至: {OUTPUT_DIR_LEVEL2}")
    print(f"二级标签RGB彩色图已保存至: {OUTPUT_RGB_DIR_LEVEL2}")
    print(f"三级标签原始类别索引图已保存至: {OUTPUT_DIR_LEVEL3}")
    print(f"三级标签RGB彩色图已保存至: {OUTPUT_RGB_DIR_LEVEL3}")
    
    # 输出二级标签颜色映射说明
    print("\n二级标签颜色映射说明 (GID数据集):")
    level2_names = ["背景", "建设用地", "耕地", "林地", "草地", "水体"]
    for class_idx, color in COLOR_MAP_LEVEL2.items():
        class_name = level2_names[class_idx] if class_idx < len(level2_names) else f"类别{class_idx}"
        print(f"  {class_name}: RGB{color}")
    
    # 输出三级标签颜色映射说明
    print("\n三级标签颜色映射说明 (GID数据集):")
    level3_names = ["背景", "建设用地1", "建设用地2", "建设用地3", "建设用地4", 
                   "耕地1", "耕地2", "耕地3", "林地1", "林地2", "林地3", 
                   "草地1", "草地2", "水体1", "水体2", "水体3"]
    for class_idx, color in COLOR_MAP_LEVEL3.items():
        class_name = level3_names[class_idx] if class_idx < len(level3_names) else f"类别{class_idx}"
        print(f"  {class_name}: RGB{color}")
