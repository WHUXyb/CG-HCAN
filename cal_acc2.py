# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:16:09 2025

@author: xiong
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings, time
import hierarchy_dict  # 导入层次关系映射
warnings.filterwarnings("ignore")

# ------------ 路径配置 ------------
# 二级标签评估
label_dir_level2 = Path(r"./data/test/label2")
pred_dir_level2  = Path(r"./results/result_level2")
# 三级标签评估
label_dir_level3 = Path(r"./data/test/label3")
pred_dir_level3  = Path(r"./results/result_level3")

max_workers = 14          # 根据硬件调整
# ----------------------------------

def pair_paths(lbl_dir, prd_dir, exts=[".png", ".tif", ".jpg"]):
    lbls = []
    for ext in exts:
        lbls += list(lbl_dir.glob(f"*{ext}"))
    lbls = sorted(lbls)
    prds = [prd_dir / p.name for p in lbls]
    return lbls, prds

def calculate_hierarchical_consistency(pred_level3_dir, pred_level2_dir, level3_labels):
    """
    计算层次一致性指标（Hierarchical Consistency）
    
    Args:
        pred_level3_dir: 三级预测结果目录
        pred_level2_dir: 二级预测结果目录
        level3_labels: 三级标签文件列表
    
    Returns:
        float: 层次一致性指标 (0-1之间)
    """
    print(f"\n{'='*20} 计算层次一致性指标 {'='*20}")
    
    # 创建映射数组以提高效率
    max_label = max(hierarchy_dict.map_3_to_2.keys())
    mapping_array = np.zeros(max_label + 1, dtype=np.int32)
    for k, v in hierarchy_dict.map_3_to_2.items():
        mapping_array[k] = v
    
    total_pixels = 0
    consistent_pixels = 0
    processed_files = 0
    
    for label_path in tqdm(level3_labels, desc="计算层次一致性"):
        pred3_path = pred_level3_dir / label_path.name
        pred2_path = pred_level2_dir / label_path.name
        
        # 检查文件是否存在
        if not pred3_path.exists() or not pred2_path.exists():
            print(f"警告: 预测文件不存在，跳过 {label_path.name}")
            continue
        
        # 读取预测结果
        pred3 = cv2.imread(str(pred3_path), cv2.IMREAD_GRAYSCALE)
        pred2 = cv2.imread(str(pred2_path), cv2.IMREAD_GRAYSCALE)
        
        if pred3 is None or pred2 is None:
            print(f"警告: 无法读取图像，跳过 {label_path.name}")
            continue
            
        if pred3.shape != pred2.shape:
            print(f"警告: 图像尺寸不匹配，跳过 {label_path.name}")
            continue
        
        # 将三级预测映射到二级
        pred3_mapped = mapping_array[pred3]
        
        # 计算一致性
        consistent = (pred3_mapped == pred2)
        consistent_pixels += np.sum(consistent)
        total_pixels += pred3.size
        processed_files += 1
    
    if total_pixels == 0:
        print("错误: 没有有效的图像对进行层次一致性计算")
        return 0.0
    
    hc_score = consistent_pixels / total_pixels
    print(f"处理文件数: {processed_files}")
    print(f"总像素数: {total_pixels:,}")
    print(f"一致像素数: {consistent_pixels:,}")
    print(f"层次一致性 (HC): {hc_score:.4f}")
    
    return hc_score

def calculate_hierarchical_distance_weighted_accuracy(pred_level3_dir, label_level3_dir, level3_labels):
    """
    计算层次距离加权准确率 (Hierarchical Distance Weighted Accuracy, HDWA)
    
    根据预测错误在层次树中的距离来加权，跨大类错误比类内错误惩罚更重
    
    Args:
        pred_level3_dir: 三级预测结果目录
        label_level3_dir: 三级真实标签目录  
        level3_labels: 三级标签文件列表
    
    Returns:
        float: HDWA指标值
    """
    print(f"\n{'='*20} 计算层次距离加权准确率 {'='*20}")
    
    # 构建层次距离矩阵
    max_label = max(max(hierarchy_dict.map_3_to_2.keys()), 8)  # 确保覆盖所有标签
    distance_matrix = np.zeros((max_label + 1, max_label + 1), dtype=np.float32)
    
    # 填充距离矩阵
    for i in range(max_label + 1):
        for j in range(max_label + 1):
            if i == j:
                distance_matrix[i, j] = 1.0  # 完全正确，权重1.0
            elif i in hierarchy_dict.map_3_to_2 and j in hierarchy_dict.map_3_to_2:
                # 检查是否属于同一二级类
                if hierarchy_dict.map_3_to_2[i] == hierarchy_dict.map_3_to_2[j]:
                    distance_matrix[i, j] = 0.5  # 同二级类内错误，权重0.5
                else:
                    distance_matrix[i, j] = 0.1  # 跨二级类错误，权重0.1
            else:
                distance_matrix[i, j] = 0.0  # 背景或无效类别错误，权重0.0
    
    total_weighted_score = 0.0
    total_pixels = 0
    processed_files = 0
    
    for label_path in tqdm(level3_labels, desc="计算HDWA"):
        pred_path = pred_level3_dir / label_path.name
        
        if not pred_path.exists():
            continue
            
        # 读取图像
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        if pred is None or label is None or pred.shape != label.shape:
            continue
        
        # 计算加权准确率
        pred_flat = pred.ravel()
        label_flat = label.ravel()
        
        # 使用距离矩阵计算权重
        weights = distance_matrix[label_flat, pred_flat]
        total_weighted_score += np.sum(weights)
        total_pixels += len(weights)
        processed_files += 1
    
    hdwa_score = total_weighted_score / total_pixels if total_pixels > 0 else 0.0
    print(f"处理文件数: {processed_files}")
    print(f"总像素数: {total_pixels:,}")
    print(f"层次距离加权准确率 (HDWA): {hdwa_score:.4f}")
    
    return hdwa_score

def calculate_hierarchical_iou(pred_level2_dir, pred_level3_dir, label_level2_dir, label_level3_dir, level3_labels):
    """
    计算层次IoU (Hierarchical IoU, HIoU)
    
    在每个层次上分别计算IoU，然后按重要性加权平均
    
    Args:
        pred_level2_dir: 二级预测结果目录
        pred_level3_dir: 三级预测结果目录
        label_level2_dir: 二级真实标签目录
        label_level3_dir: 三级真实标签目录
        level3_labels: 三级标签文件列表
    
    Returns:
        tuple: (HIoU得分, 二级mIoU, 三级mIoU)
    """
    print(f"\n{'='*20} 计算层次IoU {'='*20}")
    
    def compute_miou_for_level(pred_dir, label_dir, labels, level_name):
        """计算单个层次的mIoU"""
        conf_matrix = {}
        
        for label_path in labels:
            pred_path = pred_dir / label_path.name
            if not pred_path.exists():
                continue
                
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if level_name == "二级":
                label = cv2.imread(str(label_dir / label_path.name), cv2.IMREAD_GRAYSCALE)
            else:  # 三级
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                
            if pred is None or label is None or pred.shape != label.shape:
                continue
            
            # 累积混淆矩阵
            unique_labels = np.unique(np.concatenate([pred.ravel(), label.ravel()]))
            for class_id in unique_labels:
                if class_id not in conf_matrix:
                    conf_matrix[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                pred_mask = (pred == class_id)
                label_mask = (label == class_id)
                
                conf_matrix[class_id]['tp'] += np.sum(pred_mask & label_mask)
                conf_matrix[class_id]['fp'] += np.sum(pred_mask & ~label_mask)
                conf_matrix[class_id]['fn'] += np.sum(~pred_mask & label_mask)
        
        # 计算每类IoU
        ious = []
        for class_id, metrics in conf_matrix.items():
            if class_id == 0:  # 跳过背景类
                continue
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    # 计算二级和三级的mIoU
    level2_labels = [label_level2_dir / lp.name for lp in level3_labels]
    miou_level2 = compute_miou_for_level(pred_level2_dir, label_level2_dir, level2_labels, "二级")
    miou_level3 = compute_miou_for_level(pred_level3_dir, label_level3_dir, level3_labels, "三级")
    
    # 加权平均 (可以根据具体需求调整权重)
    weight_level2 = 0.4  # 二级权重
    weight_level3 = 0.6  # 三级权重
    hiou_score = weight_level2 * miou_level2 + weight_level3 * miou_level3
    
    print(f"二级mIoU: {miou_level2:.4f}")
    print(f"三级mIoU: {miou_level3:.4f}")
    print(f"层次IoU (HIoU): {hiou_score:.4f} (权重: 二级{weight_level2}, 三级{weight_level3})")
    
    return hiou_score, miou_level2, miou_level3

def calculate_error_severity_score(pred_level3_dir, label_level3_dir, level3_labels):
    """
    计算错误严重性评分 (Error Severity Score, ESS)
    
    根据错误类型分配不同权重：
    - 完全正确: 1.0
    - 同二级类内错误: 0.6
    - 跨二级类错误: 0.2
    - 背景错误: 0.0
    
    Args:
        pred_level3_dir: 三级预测结果目录
        label_level3_dir: 三级真实标签目录
        level3_labels: 三级标签文件列表
    
    Returns:
        tuple: (ESS得分, 错误类型统计)
    """
    print(f"\n{'='*20} 计算错误严重性评分 {'='*20}")
    
    error_stats = {
        'correct': 0,           # 完全正确
        'intra_class': 0,       # 同二级类内错误
        'inter_class': 0,       # 跨二级类错误
        'background': 0         # 背景错误
    }
    
    total_score = 0.0
    total_pixels = 0
    processed_files = 0
    
    for label_path in tqdm(level3_labels, desc="计算ESS"):
        pred_path = pred_level3_dir / label_path.name
        
        if not pred_path.exists():
            continue
            
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        if pred is None or label is None or pred.shape != label.shape:
            continue
        
        pred_flat = pred.ravel()
        label_flat = label.ravel()
        
        # 分类错误类型
        for p, l in zip(pred_flat, label_flat):
            if p == l:
                error_stats['correct'] += 1
                total_score += 1.0
            elif l == 0 or p == 0:  # 涉及背景的错误
                error_stats['background'] += 1
                total_score += 0.0
            elif (l in hierarchy_dict.map_3_to_2 and p in hierarchy_dict.map_3_to_2 and
                  hierarchy_dict.map_3_to_2[l] == hierarchy_dict.map_3_to_2[p]):
                error_stats['intra_class'] += 1  # 同二级类内错误
                total_score += 0.6
            else:
                error_stats['inter_class'] += 1  # 跨二级类错误
                total_score += 0.2
            
            total_pixels += 1
        
        processed_files += 1
    
    ess_score = total_score / total_pixels if total_pixels > 0 else 0.0
    
    print(f"处理文件数: {processed_files}")
    print(f"总像素数: {total_pixels:,}")
    print(f"错误类型统计:")
    print(f"  完全正确: {error_stats['correct']:,} ({error_stats['correct']/total_pixels*100:.2f}%)")
    print(f"  同类内错误: {error_stats['intra_class']:,} ({error_stats['intra_class']/total_pixels*100:.2f}%)")
    print(f"  跨类错误: {error_stats['inter_class']:,} ({error_stats['inter_class']/total_pixels*100:.2f}%)")
    print(f"  背景错误: {error_stats['background']:,} ({error_stats['background']/total_pixels*100:.2f}%)")
    print(f"错误严重性评分 (ESS): {ess_score:.4f}")
    
    return ess_score, error_stats

def calculate_semantic_consistency(pred_level3_dir, level3_labels):
    """
    计算语义一致性指标 (Semantic Consistency, SC)
    
    评估预测结果在语义层面的合理性，检查相邻像素的类别转换是否合理
    
    Args:
        pred_level3_dir: 三级预测结果目录
        level3_labels: 三级标签文件列表
    
    Returns:
        float: 语义一致性得分
    """
    print(f"\n{'='*20} 计算语义一致性指标 {'='*20}")
    
    # 定义合理的类别转换 (基于语义相似性)
    # 这里简化为同二级类内的转换是合理的
    def is_reasonable_transition(class1, class2):
        if class1 == class2:
            return True
        if (class1 in hierarchy_dict.map_3_to_2 and class2 in hierarchy_dict.map_3_to_2 and
            hierarchy_dict.map_3_to_2[class1] == hierarchy_dict.map_3_to_2[class2]):
            return True
        return False
    
    total_transitions = 0
    reasonable_transitions = 0
    processed_files = 0
    
    for label_path in tqdm(level3_labels, desc="计算语义一致性"):
        pred_path = pred_level3_dir / label_path.name
        
        if not pred_path.exists():
            continue
            
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        
        h, w = pred.shape
        
        # 检查水平相邻像素
        for i in range(h):
            for j in range(w-1):
                class1, class2 = pred[i, j], pred[i, j+1]
                if is_reasonable_transition(class1, class2):
                    reasonable_transitions += 1
                total_transitions += 1
        
        # 检查垂直相邻像素
        for i in range(h-1):
            for j in range(w):
                class1, class2 = pred[i, j], pred[i+1, j]
                if is_reasonable_transition(class1, class2):
                    reasonable_transitions += 1
                total_transitions += 1
        
        processed_files += 1
    
    sc_score = reasonable_transitions / total_transitions if total_transitions > 0 else 0.0
    
    print(f"处理文件数: {processed_files}")
    print(f"总转换数: {total_transitions:,}")
    print(f"合理转换数: {reasonable_transitions:,}")
    print(f"语义一致性 (SC): {sc_score:.4f}")
    
    return sc_score

def calculate_metrics(label_dir, pred_dir, level_name=""):
    print(f"\n{'='*20} {level_name} 标签评估 {'='*20}")
    labels, preds = pair_paths(label_dir, pred_dir)
    
    if not labels:
        print(f"警告：{level_name}标签路径 {label_dir} 中未找到图像文件")
        return

    # ---------- Pass-1：统计真实出现的类别 ----------
    label_set = set()
    for lp, pp in tqdm(zip(labels, preds), total=len(labels), desc=f"扫描{level_name}类别"):
        if not pp.exists():
            print(f"警告：预测文件 {pp} 不存在，跳过")
            continue
            
        gt  = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        pd  = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        label_set.update(np.unique(gt))
        label_set.update(np.unique(pd))

    class_numbers = sorted(label_set)          # 例如 [0,1,2,5]
    classNum      = len(class_numbers)         # 4
    print(f"类别数量: {classNum}")
    print(f"类别编号: {class_numbers}")

    # ---------- 建立"原标签值 → 连续索引"映射 ----------
    max_val = max(class_numbers)
    mapping = np.full(max_val + 1, -1, dtype=np.int32)
    for idx, v in enumerate(class_numbers):
        mapping[v] = idx            # e.g. 0→0, 1→1, 2→2, 5→3

    # ---------- Pass-2：累积混淆矩阵 ----------
    def read_and_bincount(args):
        lp, pp, map_arr, k = args
        if not pp.exists():
            return np.zeros(k*k, dtype=np.int64)
            
        gt = map_arr[cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)].ravel()
        pd = map_arr[cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)].ravel()
        return np.bincount(k * gt + pd, minlength=k*k)

    conf_mat = np.zeros((classNum, classNum), dtype=np.int64)
    mapping_arr = mapping                         # 给线程用一个只读副本

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        iterator = ex.map(read_and_bincount,
                        [(lp, pp, mapping_arr, classNum) for lp, pp in zip(labels, preds)])
        for local in tqdm(iterator, total=len(labels), desc=f"计算{level_name}混淆矩阵"):
            conf_mat += local.reshape(classNum, classNum)
    print(f"累积完成，用时 {time.time()-t0:.1f}s")

    # ---------- 指标 ----------
    diag   = np.diag(conf_mat).astype(np.float64)
    sum_r  = conf_mat.sum(axis=1).astype(np.float64)
    sum_c  = conf_mat.sum(axis=0).astype(np.float64)
    total  = conf_mat.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.nan_to_num(diag / sum_c)
        recall    = np.nan_to_num(diag / sum_r)
        f1        = np.nan_to_num(2 * precision * recall / (precision + recall))
        iou       = np.nan_to_num(diag / (sum_r + sum_c - diag))

    oa    = diag.sum() / total
    miou  = np.nanmean(iou)
    freq  = sum_r / total
    fwiou = (freq * iou).sum()
    pe    = (sum_r @ sum_c) / total**2
    kappa = (oa - pe) / (1 - pe)

    # ---------- 输出 ----------
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n{level_name}混淆矩阵:\n", conf_mat)
    print(f"{level_name}Precision:", precision)
    print(f"{level_name}Recall:   ", recall)
    print(f"{level_name}F1-score: ", f1)
    print(f"{level_name}IoU:      ", iou)
    print(f"{level_name}OA={oa:.4f}, mIoU={miou:.4f}, FWIoU={fwiou:.4f}, Kappa={kappa:.4f}")
    
    # 保存每个类别的指标到CSV
    class_metrics = np.column_stack((precision, recall, f1, iou))
    np.savetxt(f"metrics_{level_name}.csv", class_metrics, delimiter=",", 
               header="Precision,Recall,F1,IoU", comments="",
               fmt="%.4f")
    
    overall_metrics = np.array([oa, miou, fwiou, kappa])
    with open(f"metrics_{level_name}.csv", 'a') as f:
        f.write("\nOverall Metrics\nOA,mIoU,FWIoU,Kappa\n")
        f.write(",".join([f"{m:.4f}" for m in overall_metrics]))
    
    print(f"详细指标已保存到 metrics_{level_name}.csv")
    
    return oa, miou, fwiou, kappa

if __name__ == "__main__":
    print("开始评估分割精度...")
    
    # 评估二级标签
    level2_metrics = calculate_metrics(label_dir_level2, pred_dir_level2, "二级")
    
    # 评估三级标签
    level3_metrics = calculate_metrics(label_dir_level3, pred_dir_level3, "三级")
    
    # 计算层次特定指标
    labels_level3, _ = pair_paths(label_dir_level3, pred_dir_level3)
    if labels_level3:
        print(f"\n{'='*60}")
        print("开始计算层次特定指标...")
        print(f"{'='*60}")
        
        # 1. 层次一致性指标 (HC)
        hc_score = calculate_hierarchical_consistency(pred_dir_level3, pred_dir_level2, labels_level3)
        
        # 2. 层次距离加权准确率 (HDWA)
        hdwa_score = calculate_hierarchical_distance_weighted_accuracy(pred_dir_level3, label_dir_level3, labels_level3)
        
        # 3. 层次IoU (HIoU)
        hiou_score, hiou_level2, hiou_level3 = calculate_hierarchical_iou(
            pred_dir_level2, pred_dir_level3, label_dir_level2, label_dir_level3, labels_level3)
        
        # # 4. 错误严重性评分 (ESS)
        # ess_score, error_stats = calculate_error_severity_score(pred_dir_level3, label_dir_level3, labels_level3)
        
        # # 5. 语义一致性指标 (SC)
        # sc_score = calculate_semantic_consistency(pred_dir_level3, labels_level3)
        
        # 初始化变量以避免NameError
        ess_score = 0.0
        sc_score = 0.0
        error_stats = {'correct': 0, 'intra_class': 0, 'inter_class': 0, 'background': 0}
        
        # 保存所有层次特定指标
        with open("metrics_hierarchical_all.txt", 'w', encoding='utf-8') as f:
            f.write("层次分类任务特定指标评估报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 层次一致性指标 (Hierarchical Consistency, HC)\n")
            f.write("-"*50 + "\n")
            f.write("定义: 三级预测映射到二级后与二级预测的一致性\n")
            f.write("公式: HC = 一致像素数 / 总像素数\n")
            f.write("取值范围: [0, 1], 1表示完全一致\n")
            f.write(f"得分: {hc_score:.4f}\n\n")
            
            f.write("2. 层次距离加权准确率 (Hierarchical Distance Weighted Accuracy, HDWA)\n")
            f.write("-"*50 + "\n")
            f.write("定义: 根据预测错误在层次树中的距离来加权的准确率\n")
            f.write("权重设置: 完全正确=1.0, 同二级类内错误=0.5, 跨二级类错误=0.1\n")
            f.write("取值范围: [0, 1], 值越高表示错误越轻微\n")
            f.write(f"得分: {hdwa_score:.4f}\n\n")
            
            f.write("3. 层次IoU (Hierarchical IoU, HIoU)\n")
            f.write("-"*50 + "\n")
            f.write("定义: 在每个层次上分别计算IoU，然后加权平均\n")
            f.write("权重设置: 二级=0.4, 三级=0.6\n")
            f.write("取值范围: [0, 1], 综合考虑各层次的分割质量\n")
            f.write(f"二级mIoU: {hiou_level2:.4f}\n")
            f.write(f"三级mIoU: {hiou_level3:.4f}\n")
            f.write(f"层次IoU得分: {hiou_score:.4f}\n\n")
            
            f.write("4. 错误严重性评分 (Error Severity Score, ESS)\n")
            f.write("-"*50 + "\n")
            f.write("定义: 根据错误类型分配不同权重的综合评分\n")
            f.write("权重设置: 完全正确=1.0, 同类内错误=0.6, 跨类错误=0.2, 背景错误=0.0\n")
            f.write("取值范围: [0, 1], 值越高表示错误影响越小\n")
            f.write("注意: 此指标当前已禁用\n")
            f.write(f"得分: {ess_score:.4f}\n\n")
            
            f.write("5. 语义一致性指标 (Semantic Consistency, SC)\n")
            f.write("-"*50 + "\n")
            f.write("定义: 评估预测结果在空间上的语义合理性\n")
            f.write("计算方法: 检查相邻像素类别转换的合理性\n")
            f.write("取值范围: [0, 1], 值越高表示空间一致性越好\n")
            f.write("注意: 此指标当前已禁用\n")
            f.write(f"得分: {sc_score:.4f}\n\n")
            
            f.write("="*60 + "\n")
            f.write("层次特定指标总结:\n")
            f.write(f"HC (层次一致性):        {hc_score:.4f}\n")
            f.write(f"HDWA (距离加权准确率):  {hdwa_score:.4f}\n")
            f.write(f"HIoU (层次IoU):        {hiou_score:.4f}\n")
            f.write(f"ESS (错误严重性):      {ess_score:.4f}\n")
            f.write(f"SC (语义一致性):       {sc_score:.4f}\n")
        
        print(f"\n所有层次特定指标已保存到 metrics_hierarchical_all.txt")
        
        # 保存CSV格式的汇总数据
        hierarchical_metrics = [hc_score, hdwa_score, hiou_score, ess_score, sc_score]
        metric_names = ["HC", "HDWA", "HIoU", "ESS", "SC"]
        
        with open("metrics_hierarchical_summary.csv", 'w', encoding='utf-8') as f:
            f.write("Metric,Score,Description\n")
            f.write(f"HC,{hc_score:.4f},层次一致性\n")
            f.write(f"HDWA,{hdwa_score:.4f},距离加权准确率\n")
            f.write(f"HIoU,{hiou_score:.4f},层次IoU\n")
            f.write(f"ESS,{ess_score:.4f},错误严重性评分\n")
            f.write(f"SC,{sc_score:.4f},语义一致性\n")
        
    else:
        print("\n警告: 未找到三级标签文件，无法计算层次特定指标")
        hc_score = hdwa_score = hiou_score = ess_score = sc_score = 0.0
    
    print("\n所有评估完成！")
    
    # 输出总结
    print(f"\n{'='*60}")
    print("🏆 评估结果总结")
    print(f"{'='*60}")
    
    print("\n📊 传统分割指标:")
    if level2_metrics:
        print(f"  二级标签 - OA: {level2_metrics[0]:.4f}, mIoU: {level2_metrics[1]:.4f}, FWIoU: {level2_metrics[2]:.4f}, Kappa: {level2_metrics[3]:.4f}")
    if level3_metrics:
        print(f"  三级标签 - OA: {level3_metrics[0]:.4f}, mIoU: {level3_metrics[1]:.4f}, FWIoU: {level3_metrics[2]:.4f}, Kappa: {level3_metrics[3]:.4f}")
    
    print("\n🔗 层次特定指标:")
    if 'hc_score' in locals():
        print(f"  HC  (层次一致性):       {hc_score:.4f}")
        print(f"  HDWA(距离加权准确率):   {hdwa_score:.4f}")
        print(f"  HIoU(层次IoU):         {hiou_score:.4f}")
        print(f"  ESS (错误严重性):      {ess_score:.4f}")
        print(f"  SC  (语义一致性):      {sc_score:.4f}")
    else:
        print("  未计算层次特定指标")
    
    print(f"\n📁 输出文件:")
    print(f"  - metrics_二级.csv")
    print(f"  - metrics_三级.csv")
    if 'hc_score' in locals():
        print(f"  - metrics_hierarchical_all.txt (详细报告)")
        print(f"  - metrics_hierarchical_summary.csv (汇总数据)")
    
    print(f"\n{'='*60}")
    print("✅ 评估完成！层次分类任务的所有指标已计算完毕。")
    print(f"{'='*60}")
