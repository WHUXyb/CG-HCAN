# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:52:43 2022

@author: xiong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
# 修正 UNet++ Decoder 导入路径
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
# 导入层次映射关系
from hierarchy_dict import map_3_to_2, map_2_to_1

# 层次一致性损失函数
class HierarchyConsistencyLoss(nn.Module):
    """
    确保三级标签预测与二级标签预测之间保持正确的层次关系
    
    参数:
        map_3_to_2: 三级类别到二级类别的映射字典
        weight: 损失权重
    """
    def __init__(self, map_3_to_2=map_3_to_2, weight=0.5):
        super().__init__()
        self.weight = weight
        # 创建映射张量，用于将三级预测转换为二级预测
        max_key = max(map_3_to_2.keys())
        self.mapping = torch.zeros(max_key + 1, dtype=torch.long)
        for k, v in map_3_to_2.items():
            self.mapping[k] = v
        
    def forward(self, level2_pred, level3_pred):
        """
        计算层次一致性损失
        
        参数:
            level2_pred: [B, C2, H, W] 二级类别预测的 logits
            level3_pred: [B, C3, H, W] 三级类别预测的 logits
            
        返回:
            一致性损失值
        """
        # 获取三级预测的类别索引
        level3_indices = level3_pred.argmax(dim=1)  # [B, H, W]
        
        # 将三级索引映射到二级索引
        mapping = self.mapping.to(level3_indices.device)
        mapped_level2 = mapping[level3_indices]  # [B, H, W]
        
        # 获取二级预测的类别索引
        level2_indices = level2_pred.argmax(dim=1)  # [B, H, W]
        
        # 计算不一致的像素比例作为损失
        inconsistency = (mapped_level2 != level2_indices).float().mean()
        return self.weight * inconsistency

class AdaptiveSemanticGNN(nn.Module):
    """
    自适应语义图神经网络 - 高效实现版本
    创新点：
    1. 动态学习类别间的语义相似性关系，而非仅基于层次结构
    2. 结合空间上下文信息进行图推理
    3. 多尺度特征融合的图卷积
    
    优化：
    1. 使用向量化操作替代循环
    2. 批量计算邻接矩阵
    3. 减少动态图构建
    """
    def __init__(self, num_classes=16, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 可学习的语义嵌入
        self.semantic_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # 简化的邻接矩阵生成器 - 使用矩阵乘法替代循环
        self.adj_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adj_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 图卷积层 - 使用标准的线性层实现
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)  # 减少到2层
        ])
        
        # 特征投影层
        self.feat_proj = nn.Conv2d(num_classes, hidden_dim, 1)  # 使用1x1卷积
        self.output_proj = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 门控机制
        self.gate = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值
        
    def generate_adjacency_matrix_efficient(self):
        """高效生成邻接矩阵 - 使用向量化操作"""
        # 将语义嵌入投影到低维空间
        embed_proj1 = torch.relu(self.adj_fc1(self.semantic_embeddings))  # [C, D/2]
        embed_proj2 = torch.relu(self.adj_fc2(self.semantic_embeddings))  # [C, D/2]
        
        # 计算相似性矩阵 - 使用点积
        adj_matrix = torch.matmul(embed_proj1, embed_proj2.T)  # [C, C]
        
        # 归一化并添加自环
        adj_matrix = torch.sigmoid(adj_matrix)
        adj_matrix = adj_matrix + torch.eye(self.num_classes, device=adj_matrix.device)
        
        # 行归一化
        adj_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)
        
        return adj_matrix
    
    def forward(self, class_logits):
        """
        Args:
            class_logits: [B, C, H, W] 类别预测logits
        Returns:
            refined_logits: [B, C, H, W] 增强后的预测logits
        """
        B, C, H, W = class_logits.shape
        
        # 生成邻接矩阵（所有批次共享）
        adj_matrix = self.generate_adjacency_matrix_efficient()  # [C, C]
        
        # 将logits投影到隐藏空间
        hidden_feats = self.feat_proj(class_logits)  # [B, D, H, W]
        
        # 转换为 [B, H*W, D] 格式进行图卷积
        hidden_feats = hidden_feats.view(B, self.hidden_dim, -1).transpose(1, 2)  # [B, H*W, D]
        
        # 获取类别概率分布用于加权
        class_probs = F.softmax(class_logits, dim=1)  # [B, C, H, W]
        class_probs = class_probs.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 图卷积操作
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 计算加权特征聚合
            # 使用类别概率作为权重，通过邻接矩阵传播信息
            weighted_probs = torch.matmul(class_probs, adj_matrix)  # [B, H*W, C]
            
            # 将聚合的概率转换为特征向量（使用语义嵌入）
            # 扩展语义嵌入以匹配批次大小
            semantic_embed_batch = self.semantic_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
            
            # 使用加权概率聚合语义特征
            aggregated_feats = torch.bmm(weighted_probs, semantic_embed_batch)  # [B, H*W, D]
            
            # 应用GNN层并添加残差连接
            hidden_feats = gnn_layer(hidden_feats) + aggregated_feats
            
            # 非线性激活（最后一层除外）
            if i < len(self.gnn_layers) - 1:
                hidden_feats = F.relu(hidden_feats)
        
        # 转换回 [B, D, H, W] 格式
        hidden_feats = hidden_feats.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # 输出投影
        refined_logits = self.output_proj(hidden_feats)  # [B, C, H, W]
        
        # 门控残差连接
        return self.gate * refined_logits + class_logits

class CLIPGuidedSemanticGNN(nn.Module):
    """
    CLIP引导的语义图神经网络
    
    创新点：
    1. 利用CLIP预训练模型提供的语义嵌入作为先验知识
    2. 通过知识蒸馏让可学习嵌入向CLIP嵌入对齐
    3. 结合视觉-语言跨模态信息增强类别关系建模
    4. 动态融合CLIP先验和自适应学习的语义关系
    """
    def __init__(self, num_classes=16, hidden_dim=128, 
                 clip_model_name='ViT-B/32', 
                 temperature=0.1,
                 distill_weight=0.3,
                 dataset_type='gid'):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.dataset_type = dataset_type
        
        # 导入CLIP
        try:
            import clip
            self.clip_available = True
        except ImportError:
            print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.clip_available = False
        
        # 设备信息 - 从外部传入或自动检测
        # 优先使用外部传入的设备，如果没有则自动检测
        if hasattr(self, '_device_override'):
            self.device = self._device_override
        else:
            import os
            if torch.cuda.is_available():
                gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                if ',' in gpu_id:
                    # 如果设置了多个GPU，使用第一个
                    gpu_id = gpu_id.split(',')[0]
                # 当使用CUDA_VISIBLE_DEVICES=6时，实际上是cuda:0（因为只看到一个GPU）
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        
        # 初始化CLIP模型（如果可用）
        if self.clip_available:
            try:
                # 确保CLIP模型使用与主模型相同的设备
                self.clip_model, _ = clip.load(clip_model_name, device=self.device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                
                # CLIP特征维度（ViT-B/32: 512）
                self.clip_dim = 512
                
                # 投影层：将CLIP嵌入投影到隐藏维度
                self.clip_proj = nn.Linear(self.clip_dim, hidden_dim)
                
                print(f"CLIP模型加载成功，使用设备: {self.device}")
            except Exception as e:
                print(f"CLIP模型加载失败: {e}")
                self.clip_available = False
        
        # 可学习的语义嵌入（与原始版本相同）
        self.semantic_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # 融合CLIP嵌入和可学习嵌入的门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 邻接矩阵生成器（考虑CLIP语义相似性）
        self.adj_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adj_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 图卷积层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # 特征投影层
        self.feat_proj = nn.Conv2d(num_classes, hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 门控残差
        self.gate = nn.Parameter(torch.tensor(0.1))
        
        # 预定义的类别文本描述（需要根据具体数据集调整）
        self.class_descriptions = self._get_class_descriptions()
        
        # 预计算CLIP嵌入
        if self.clip_available:
            try:
                clip_embeddings = self._compute_clip_embeddings()
                if clip_embeddings is not None:
                    # 确保嵌入在CPU上注册，稍后会随模型一起移动到正确设备
                    self.register_buffer('clip_embeddings', clip_embeddings.cpu())
                else:
                    self.clip_embeddings = None
            except Exception as e:
                print(f"CLIP嵌入初始化失败: {e}")
                self.clip_embeddings = None
                self.clip_available = False
        else:
            self.clip_embeddings = None
    
    def _get_class_descriptions(self):
        """
        获取类别的文本描述
        根据实际的num_classes和数据集类型动态生成描述
        """
        if self.dataset_type.lower() == 'gid':
            # GID数据集类别描述（16类）
            full_descriptions = [
                "background or unknown area",  # 0: 背景
                "industrial building area",    # 1: 工业建筑
                "urban residential area",      # 2: 城市住宅
                "rural residential area",      # 3: 农村住宅
                "traffic and road area",       # 4: 交通道路
                "paddy field cropland",        # 5: 水田
                "irrigated cropland",          # 6: 灌溉农田
                "dry cropland farmland",       # 7: 旱地
                "garden plot land",            # 8: 园地
                "arbor forest woodland",       # 9: 乔木林
                "shrub land area",             # 10: 灌木林
                "natural grassland",           # 11: 天然草地
                "artificial grassland",        # 12: 人工草地
                "river water body",            # 13: 河流
                "lake water body",             # 14: 湖泊
                "pond water body"              # 15: 池塘
            ]
        elif self.dataset_type.lower() == 'forest':
            # 林地数据集类别描述（18类）
            full_descriptions = [
                "background or unknown area",     # 0: 背景
                "coniferous forest",              # 1: 针叶林
                "broadleaf forest",               # 2: 阔叶林
                "mixed forest",                   # 3: 混交林
                "bamboo forest",                  # 4: 竹林
                "dense shrubland",                # 5: 密灌丛
                "sparse shrubland",               # 6: 疏灌丛
                "other shrubland",                # 7: 其他灌丛
                "grassland",                      # 8: 草地
                "closed forest",                  # 9: 郁闭林
                "open forest",                    # 10: 疏林
                "young forest",                   # 11: 幼林
                "plantation forest",              # 12: 人工林
                "natural forest",                 # 13: 天然林
                "disturbed forest",               # 14: 受干扰林地
                "burned forest area",             # 15: 火烧迹地
                "logged forest area",             # 16: 采伐迹地
                "other forest land"               # 17: 其他林地
            ]
        else:
            # 通用描述
            full_descriptions = [f"land cover class {i}" for i in range(max(16, self.num_classes))]
        
        # 根据实际类别数量调整描述
        if self.num_classes <= len(full_descriptions):
            descriptions = full_descriptions[:self.num_classes]
        else:
            # 如果需要更多类别，扩展通用描述
            descriptions = full_descriptions[:]
            for i in range(len(full_descriptions), self.num_classes):
                descriptions.append(f"land cover class {i}")
        
        print(f"为{self.dataset_type}数据集的{self.num_classes}个类别生成描述: {len(descriptions)}个")
        return descriptions
    
    def _compute_clip_embeddings(self):
        """
        使用CLIP文本编码器计算类别的语义嵌入
        """
        if not self.clip_available:
            return None
        
        try:
            import clip
            
            # 构建更丰富的文本提示
            text_prompts = []
            for desc in self.class_descriptions:
                # 使用多个模板增强语义表示
                templates = [
                    f"a satellite image of {desc}",
                    f"an aerial view of {desc}",
                    f"remote sensing image showing {desc}",
                    f"{desc} in satellite imagery"
                ]
                text_prompts.extend(templates)
            
            print(f"正在计算{len(text_prompts)}个文本提示的CLIP嵌入...")
            
            # 编码文本
            with torch.no_grad():
                text_tokens = clip.tokenize(text_prompts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features.float()
                
                # 对每个类别的多个模板取平均
                num_templates = 4
                text_features = text_features.view(self.num_classes, num_templates, -1).mean(dim=1)
                
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            print(f"CLIP嵌入计算完成，形状: {text_features.shape}")
            return text_features
            
        except Exception as e:
            print(f"计算CLIP嵌入时出错: {e}")
            print("将禁用CLIP功能")
            self.clip_available = False
            return None
    
    def get_fused_embeddings(self):
        """
        融合CLIP嵌入和可学习嵌入
        """
        if self.clip_available and self.clip_embeddings is not None:
            # 确保CLIP嵌入在正确的设备上
            device = self.semantic_embeddings.device
            clip_embeddings_on_device = self.clip_embeddings.to(device)
            
            # 投影CLIP嵌入到隐藏维度
            clip_proj = self.clip_proj(clip_embeddings_on_device)
            
            # 计算融合权重
            concat_embed = torch.cat([self.semantic_embeddings, clip_proj], dim=-1)
            fusion_weights = self.fusion_gate(concat_embed)  # [num_classes, 1]
            
            # 加权融合
            fused_embeddings = fusion_weights * clip_proj + (1 - fusion_weights) * self.semantic_embeddings
        else:
            # 如果CLIP不可用，只使用可学习嵌入
            fused_embeddings = self.semantic_embeddings
        
        return fused_embeddings
    
    def generate_clip_guided_adjacency(self):
        """
        生成CLIP引导的邻接矩阵
        """
        # 获取融合后的嵌入
        fused_embeddings = self.get_fused_embeddings()
        
        # 投影到低维空间
        embed_proj1 = torch.relu(self.adj_fc1(fused_embeddings))  # [C, D/2]
        embed_proj2 = torch.relu(self.adj_fc2(fused_embeddings))  # [C, D/2]
        
        # 计算相似性矩阵
        adj_matrix = torch.matmul(embed_proj1, embed_proj2.T)  # [C, C]
        
        # 如果有CLIP嵌入，额外考虑CLIP空间的语义相似性
        if self.clip_available and self.clip_embeddings is not None:
            # 计算CLIP嵌入的余弦相似度
            clip_sim = torch.matmul(self.clip_embeddings, self.clip_embeddings.T)
            # 温度缩放
            clip_sim = clip_sim / self.temperature
            # 融合两种相似性
            adj_matrix = 0.7 * torch.sigmoid(adj_matrix) + 0.3 * torch.sigmoid(clip_sim)
        else:
            adj_matrix = torch.sigmoid(adj_matrix)
        
        # 添加自环
        adj_matrix = adj_matrix + torch.eye(self.num_classes, device=adj_matrix.device)
        
        # 行归一化
        adj_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)
        
        return adj_matrix
    
    def compute_distillation_loss(self):
        """
        计算知识蒸馏损失，让可学习嵌入向CLIP嵌入对齐
        """
        if not self.clip_available or self.clip_embeddings is None:
            return 0.0
        
        # 投影CLIP嵌入
        clip_proj = self.clip_proj(self.clip_embeddings)
        
        # 计算余弦相似度损失
        cos_sim = F.cosine_similarity(self.semantic_embeddings, clip_proj, dim=-1)
        distill_loss = (1 - cos_sim).mean()
        
        return self.distill_weight * distill_loss
    
    def forward(self, class_logits):
        """
        前向传播
        
        Args:
            class_logits: [B, C, H, W] 类别预测logits
            
        Returns:
            refined_logits: [B, C, H, W] 增强后的预测logits
        """
        B, C, H, W = class_logits.shape
        
        # 生成CLIP引导的邻接矩阵
        adj_matrix = self.generate_clip_guided_adjacency()  # [C, C]
        
        # 获取融合后的语义嵌入
        fused_embeddings = self.get_fused_embeddings()  # [C, D]
        
        # 将logits投影到隐藏空间
        hidden_feats = self.feat_proj(class_logits)  # [B, D, H, W]
        
        # 转换格式进行图卷积
        hidden_feats = hidden_feats.view(B, self.hidden_dim, -1).transpose(1, 2)  # [B, H*W, D]
        
        # 获取类别概率分布
        class_probs = F.softmax(class_logits, dim=1)  # [B, C, H, W]
        class_probs = class_probs.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 图卷积操作
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 通过邻接矩阵传播概率
            weighted_probs = torch.matmul(class_probs, adj_matrix)  # [B, H*W, C]
            
            # 扩展语义嵌入
            semantic_embed_batch = fused_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
            
            # 聚合语义特征
            aggregated_feats = torch.bmm(weighted_probs, semantic_embed_batch)  # [B, H*W, D]
            
            # 应用GNN层
            hidden_feats = gnn_layer(hidden_feats) + aggregated_feats
            
            if i < len(self.gnn_layers) - 1:
                hidden_feats = F.relu(hidden_feats)
        
        # 转换回原始格式
        hidden_feats = hidden_feats.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # 输出投影
        refined_logits = self.output_proj(hidden_feats)  # [B, C, H, W]
        
        # 门控残差连接
        return self.gate * refined_logits + class_logits

class EfficientSelfAttention(nn.Module):
    """
    内存高效的自注意力模块，通过空间降采样减少内存占用
    
    参数:
        in_channels: 输入特征通道数
        reduction_ratio: 空间降采样比例，默认为8
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 计算降采样后的空间尺寸
        h, w = max(1, H // self.reduction_ratio), max(1, W // self.reduction_ratio)
        
        # 生成查询、键、值
        q = self.query(x)
        # 对键和值进行空间降采样以节省内存
        k = F.adaptive_avg_pool2d(self.key(x), (h, w))
        v = F.adaptive_avg_pool2d(self.value(x), (h, w))
        
        # 重塑张量以计算注意力
        q = q.view(batch_size, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        k = k.view(batch_size, -1, h * w)  # B x C' x (h*w)
        v = v.view(batch_size, -1, h * w)  # B x C x (h*w)
        
        # 计算注意力权重
        attn = torch.bmm(q, k)  # B x (H*W) x (h*w)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        return self.gamma * out + x

class EfficientCrossAttention(nn.Module):
    """
    内存高效的交叉注意力模块，通过空间降采样减少内存占用
    
    参数:
        in_channels: 输入特征通道数
        reduction_ratio: 空间降采样比例，默认为8
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        """
        x: 第一个特征图 (例如：图像特征)
        y: 第二个特征图 (例如：掩码特征)
        """
        batch_size, C, H, W = x.size()
        
        # 计算降采样后的空间尺寸
        h, w = max(1, H // self.reduction_ratio), max(1, W // self.reduction_ratio)
        
        # x作为查询，y作为键和值
        q = self.query_conv(x)
        # 对键和值进行空间降采样以节省内存
        k = F.adaptive_avg_pool2d(self.key_conv(y), (h, w))
        v = F.adaptive_avg_pool2d(self.value_conv(y), (h, w))
        
        # 重塑张量以计算注意力
        q = q.view(batch_size, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        k = k.view(batch_size, -1, h * w)  # B x C' x (h*w)
        v = v.view(batch_size, -1, h * w)  # B x C x (h*w)
        
        # 计算注意力权重
        attn = torch.bmm(q, k)  # B x (H*W) x (h*w)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        return self.gamma * out + x

class LinearSelfAttention(nn.Module):
    """
    线性自注意力模块，复杂度为O(N)而非O(N²)，大幅降低内存占用
    
    参数:
        in_channels: 输入特征通道数
    """
    def __init__(self, in_channels):
        super().__init__()
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 生成查询、键、值
        q = self.query(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        k = self.key(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        v = self.value(x).view(batch_size, -1, H * W)  # B x C x (H*W)
        
        # 对键应用softmax进行归一化
        k_softmax = F.softmax(k, dim=-1)
        
        # 计算上下文向量（线性注意力）
        context = torch.bmm(v, k_softmax.transpose(1, 2))  # B x C x C'
        out = torch.bmm(context, q)  # B x C x (H*W)
        
        # 重塑并应用残差连接
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class LinearCrossAttention(nn.Module):
    """
    线性交叉注意力模块，复杂度为O(N)而非O(N²)，大幅降低内存占用
    
    参数:
        in_channels: 输入特征通道数
    """
    def __init__(self, in_channels):
        super().__init__()
        # 确保通道数至少为8，避免除以8后为0
        mid_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        """
        x: 第一个特征图 (例如：图像特征)
        y: 第二个特征图 (例如：掩码特征)
        """
        batch_size, C, H, W = x.size()
        
        # x作为查询，y作为键和值
        q = self.query_conv(x).view(batch_size, -1, H * W)  # B x C' x (H*W)
        k = self.key_conv(y).view(batch_size, -1, H * W)  # B x C' x (H*W)
        v = self.value_conv(y).view(batch_size, -1, H * W)  # B x C x (H*W)
        
        # 对键应用softmax进行归一化
        k_softmax = F.softmax(k, dim=-1)
        
        # 计算上下文向量（线性注意力）
        context = torch.bmm(v, k_softmax.transpose(1, 2))  # B x C x C'
        out = torch.bmm(context, q)  # B x C x (H*W)
        
        # 重塑并应用残差连接
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class BiCAF(nn.Module):
    """
    BiCAF: Bidirectional Cross-Attention Fusion
    双向交叉注意力融合模块，专门设计用于特征融合
    
    核心创新：
    1. 双向交叉注意力：图像→掩码增强 + 掩码→图像增强
    2. 深度可分离卷积：大幅减少参数量（相比标准卷积节省约9倍）
    3. 局部注意力机制：8×8块分割，复杂度从O(N²)降到O(N)
    4. 自适应融合权重：可学习的α、β参数控制融合比例
    5. 智能归一化：自动适配GroupNorm组数
    
    参数:
        in_channels: 输入特征通道数
        reduction_factor: 中间通道降维因子，默认16
        use_multi_scale: 是否使用多尺度注意力
    """
    def __init__(self, in_channels, reduction_factor=16, use_multi_scale=False):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        
        # 极度压缩的中间维度
        mid_channels = max(1, in_channels // reduction_factor)
        
        # 使用深度可分离卷积降低参数量
        # 深度卷积部分（3×3卷积，每个通道独立）
        self.query_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.key_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.value_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        
        # 逐点卷积部分（1×1卷积，通道间信息交互）
        self.query_pw = nn.Conv2d(in_channels, mid_channels, 1)
        self.key_pw = nn.Conv2d(in_channels, mid_channels, 1)
        self.value_pw = nn.Conv2d(in_channels, in_channels, 1)
        
        # 双向融合的门控参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 图像→掩码增强权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 掩码→图像增强权重
        
        # 如果使用多尺度注意力
        if use_multi_scale:
            self.multi_scale_conv = nn.Conv2d(in_channels * 3, in_channels, 1)
        
        # 智能GroupNorm设置，确保能被通道数整除
        def find_best_num_groups(channels, max_groups=32):
            """找到最大的能整除通道数的组数"""
            for groups in range(min(max_groups, channels), 0, -1):
                if channels % groups == 0:
                    return groups
            return 1  # 最坏情况下使用1个组（等效于LayerNorm）
        
        num_groups = find_best_num_groups(in_channels)
        self.norm = nn.GroupNorm(num_groups, in_channels)
        
    def depthwise_separable_conv(self, x, dw_conv, pw_conv):
        """深度可分离卷积"""
        x = dw_conv(x)
        x = pw_conv(x)
        return x
    
    def compute_attention(self, query_source, key_source, value_source):
        """计算注意力，使用高效的实现"""
        B, C, H, W = query_source.shape
        
        # 深度可分离卷积提取特征
        q = self.depthwise_separable_conv(query_source, self.query_dw, self.query_pw)
        k = self.depthwise_separable_conv(key_source, self.key_dw, self.key_pw)
        v = self.depthwise_separable_conv(value_source, self.value_dw, self.value_pw)
        
        # 使用局部注意力降低计算复杂度
        # 将空间维度分成小块
        block_size = 8
        h_blocks = H // block_size + (1 if H % block_size != 0 else 0)
        w_blocks = W // block_size + (1 if W % block_size != 0 else 0)
        
        # Padding确保可以整除
        pad_h = h_blocks * block_size - H
        pad_w = w_blocks * block_size - W
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))
        
        # 重塑为块
        _, C_mid, H_pad, W_pad = q.shape
        q = q.view(B, C_mid, h_blocks, block_size, w_blocks, block_size)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, h_blocks, w_blocks, C_mid, block_size, block_size
        q = q.view(B * h_blocks * w_blocks, C_mid, block_size * block_size)
        
        k = k.view(B, C_mid, h_blocks, block_size, w_blocks, block_size)
        k = k.permute(0, 2, 4, 1, 3, 5).contiguous()
        k = k.view(B * h_blocks * w_blocks, C_mid, block_size * block_size)
        
        v = v.view(B, C, h_blocks, block_size, w_blocks, block_size)
        v = v.permute(0, 2, 4, 1, 3, 5).contiguous()
        v = v.view(B * h_blocks * w_blocks, C, block_size * block_size)
        
        # 计算局部注意力
        attn = torch.bmm(q.transpose(1, 2), k) / (C_mid ** 0.5)  # (B*blocks, block_size^2, block_size^2)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.transpose(1, 2))  # (B*blocks, C, block_size^2)
        
        # 重塑回原始维度
        out = out.view(B, h_blocks, w_blocks, C, block_size, block_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, C, H_pad, W_pad)
        
        # 移除padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return out
    
    def forward(self, img_feat, mask_feat):
        """
        双向交叉注意力融合
        
        Args:
            img_feat: 图像特征 [B, C, H, W]
            mask_feat: 掩码特征 [B, C, H, W]
            
        Returns:
            fused_feat: 融合后的特征 [B, C, H, W]
        """
        # 图像引导的掩码增强
        mask_enhanced = self.compute_attention(mask_feat, img_feat, img_feat)
        
        # 掩码引导的图像增强
        img_enhanced = self.compute_attention(img_feat, mask_feat, mask_feat)
        
        # 自适应融合
        if self.use_multi_scale:
            # 多尺度特征提取
            img_feat_2x = F.interpolate(img_feat, scale_factor=0.5, mode='bilinear', align_corners=False)
            img_feat_2x = F.interpolate(img_feat_2x, size=(img_feat.shape[2], img_feat.shape[3]), 
                                      mode='bilinear', align_corners=False)
            
            img_feat_4x = F.interpolate(img_feat, scale_factor=0.25, mode='bilinear', align_corners=False)
            img_feat_4x = F.interpolate(img_feat_4x, size=(img_feat.shape[2], img_feat.shape[3]), 
                                      mode='bilinear', align_corners=False)
            
            # 拼接多尺度特征
            multi_scale = torch.cat([img_feat, img_feat_2x, img_feat_4x], dim=1)
            multi_scale = self.multi_scale_conv(multi_scale)
            
            # 融合
            fused = self.alpha * mask_enhanced + self.beta * img_enhanced + multi_scale
        else:
            # 简单融合
            fused = self.alpha * mask_enhanced + self.beta * img_enhanced + img_feat + mask_feat
        
        # 归一化
        fused = self.norm(fused)
        
        return fused

class SimpleMaskEncoder(nn.Module):
    """
    简化的掩码编码器，用于提取mask的边界先验信息
    参数量远少于标准encoder，仅用于提取空间边界特征
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # 简单的多层卷积结构，逐步降采样并增加通道数
        # 与主干encoder的5个层级对应
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 定义输出通道数，与主干encoder保持一致的接口
        self.out_channels = [0, 16, 32, 64, 128, 256]  # 0是占位符，对应第0层
        
    def forward(self, x):
        # 模拟encoder的5层输出结构
        x1 = self.conv1(x)      # 1/1 分辨率
        x2 = self.conv2(x1)     # 1/2 分辨率  
        x3 = self.conv3(x2)     # 1/4 分辨率
        x4 = self.conv4(x3)     # 1/8 分辨率
        x5 = self.conv5(x4)     # 1/16 分辨率
        
        return [x, x1, x2, x3, x4, x5]  # 返回6个特征层，第0个是原始输入

class ChannelAdapter(nn.Module):
    """
    通道适配器，将简化mask特征的通道数调整为与主干特征一致
    """
    def __init__(self, mask_channels, img_channels):
        super().__init__()
        self.adapters = nn.ModuleList()
        
        for mc, ic in zip(mask_channels, img_channels):
            if mc == 0:  # 跳过占位符
                self.adapters.append(nn.Identity())
            elif mc != ic:
                # 使用1x1卷积调整通道数
                self.adapters.append(
                    nn.Sequential(
                        nn.Conv2d(mc, ic, 1),
                        nn.BatchNorm2d(ic),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.adapters.append(nn.Identity())
    
    def forward(self, mask_features):
        adapted_features = []
        for feat, adapter in zip(mask_features, self.adapters):
            adapted_features.append(adapter(feat))
        return adapted_features

class DualEncoderUNetPP(nn.Module):
    """
    双分支 UNet++ 模型：
    - 图像分支提取 RGB 特征
    - 一级标签分支使用简化编码器提取边界先验特征（大幅减少参数量）
    - 使用最简单的相加方法融合两个分支的特征
    - 层级对齐后解码到多级分割
    - 新增：支持同时预测二级和三级标签
    - 新增：集成类别关系GNN增强细粒度分类
    - 新增：支持CLIP引导的语义图神经网络
    """
    def __init__(self,
                 encoder_name: str = 'efficientnet-b0',
                 encoder_weights: str = 'imagenet',
                 num_classes_level2: int = 6,  # 二级类别数（GID数据集：0背景 + 5土地利用大类 = 6类）
                 num_classes_level3: int = 16, # 三级类别数（GID数据集：0背景 + 15土地利用子类 = 16类）
                 use_adaptive_gnn: bool = False,  # 使用基础自适应图神经网络
                 use_clip_gnn: bool = True,  # 使用CLIP引导的图神经网络
                 use_cross_attention_fusion: bool = True,  # 使用Cross-Attention融合
                 cross_attention_reduction: int = 16,  # Cross-Attention的通道缩减倍数
                 use_multi_scale_fusion: bool = False):  # 是否使用多尺度融合
        super().__init__()
        # 设置 decoder channels
        decoder_channels = (256, 128, 64, 32, 16)
        
        # RGB 图像 Encoder（保持不变）
        self.enc_img = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # 简化的掩码编码器（大幅减少参数量）
        self.enc_mask = SimpleMaskEncoder(in_channels=1)
        
        # 获取编码器的输出通道数
        img_channels = self.enc_img.out_channels
        mask_channels = self.enc_mask.out_channels
        
        # 检查输出通道数是否为列表形式
        if not isinstance(img_channels, (list, tuple)):
            img_channels = [img_channels]
            mask_channels = [mask_channels]
        
        # 通道适配器，将mask特征通道数调整为与图像特征一致
        self.channel_adapter = ChannelAdapter(mask_channels, img_channels)
        
        # 特征融合方式
        self.use_cross_attention_fusion = use_cross_attention_fusion
        
        if use_cross_attention_fusion:
            # 为每个编码器层级创建Cross-Attention融合模块
            self.fusion_modules = nn.ModuleList()
            for i, channels in enumerate(img_channels):
                if i == 0:  # 跳过第0层（原始输入）
                    self.fusion_modules.append(nn.Identity())
                else:
                    # 根据层级调整reduction factor，深层使用更大的压缩
                    reduction = cross_attention_reduction if i < 3 else cross_attention_reduction * 2
                    self.fusion_modules.append(
                        BiCAF(
                            in_channels=channels,
                            reduction_factor=reduction,
                            use_multi_scale=use_multi_scale_fusion and i >= 3  # 仅在深层使用多尺度
                        )
                    )
            print(f"使用BiCAF双向交叉注意力融合，通道缩减倍数: {cross_attention_reduction}")
        else:
            print("使用简单相加融合")
        
        # UNet++ Decoder
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.enc_img.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=True,
            attention_type='scse'
        )
        
        # 二级分割头
        self.seg_head_level2 = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=num_classes_level2,
            kernel_size=1
        )
        
        # 三级分割头
        self.seg_head_level3 = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=num_classes_level3,
            kernel_size=1
        )
        
        # 新增：基于类别关系的图神经网络，用于增强三级类别预测
        self.use_adaptive_gnn = use_adaptive_gnn
        self.use_clip_gnn = use_clip_gnn
        
        # 初始化GNN模块（平行关系）
        self.adaptive_gnn = None
        self.clip_gnn = None
        
        if use_adaptive_gnn:
            # 使用基础自适应语义图神经网络
            self.adaptive_gnn = AdaptiveSemanticGNN(num_classes=num_classes_level3)
            
        if use_clip_gnn:
            # 使用CLIP引导的语义图神经网络
            self.clip_gnn = CLIPGuidedSemanticGNN(
                num_classes=num_classes_level3,
                hidden_dim=128,
                clip_model_name='ViT-B/32',
                temperature=0.1,
                distill_weight=0.3,
                dataset_type='gid'  # 默认使用GID数据集描述，可根据需要调整
            )
        
        # 检查参数组合的合理性
        if use_adaptive_gnn and use_clip_gnn:
            print("Warning: 同时启用两个GNN模块，将使用融合策略")
        elif not use_adaptive_gnn and not use_clip_gnn:
            print("Info: 未启用任何GNN模块，仅使用基础网络")
        
        # 兼容旧代码
        self.seg_head = self.seg_head_level2

    def forward(self, x: torch.Tensor, m: torch.Tensor, return_clip_loss: bool = False) -> tuple:
        """
        x: RGB 图像张量，shape = [B,3,H,W]
        m: 一级标签张量，shape = [B,1,H,W]
        return_clip_loss: 是否返回CLIP蒸馏损失
        返回: 
            - 如果 return_clip_loss=False: (level2_logits, level3_logits)
            - 如果 return_clip_loss=True: (level2_logits, level3_logits, clip_loss)
        """
        # 主干特征
        feats_img = self.enc_img(x)
        
        # 支路特征（简化版本）
        feats_mask_raw = self.enc_mask(m)
        
        # 通道适配，将mask特征调整为与图像特征相同的通道数
        feats_mask = self.channel_adapter(feats_mask_raw)
        
        # 特征交互：Cross-Attention融合或简单相加
        refined_feats = []
        
        for i, (fi, fm) in enumerate(zip(feats_img, feats_mask)):
            # 获取当前特征图的空间尺寸
            _, _, H_img, W_img = fi.shape
            _, _, H_mask, W_mask = fm.shape
            
            # 如果空间尺寸不匹配，将mask特征调整到与图像特征相同的尺寸
            if H_img != H_mask or W_img != W_mask:
                fm = F.interpolate(fm, size=(H_img, W_img), mode='bilinear', align_corners=False)
            
            # 使用Cross-Attention融合或简单相加
            if self.use_cross_attention_fusion and i > 0:  # 第0层是原始输入，跳过
                # 使用对应层级的Cross-Attention模块
                fused = self.fusion_modules[i](fi, fm)
            else:
                # 简单相加融合（用于第0层或未启用Cross-Attention时）
                fused = fi + fm
            
            refined_feats.append(fused)
        
        # Decoder 解码
        decoder_outs = self.decoder(*refined_feats)
        
        # UNet++ decoder 返回多个中间特征，取最后一层
        if isinstance(decoder_outs, (list, tuple)):
            d = decoder_outs[-1]
        else:
            d = decoder_outs
            
        # 二级和三级分割头
        level2_logits = self.seg_head_level2(d)
        level3_logits = self.seg_head_level3(d)
        
        # 使用基于类别关系的GNN增强三级类别预测
        clip_loss = 0.0
        
        # 平行处理：可以同时使用两个GNN模块
        if self.use_adaptive_gnn and self.use_clip_gnn:
            # 同时使用两个GNN：融合策略
            adaptive_output = self.adaptive_gnn(level3_logits)
            clip_output = self.clip_gnn(level3_logits)
            # 加权融合两个输出（可以调整权重）
            level3_logits = 0.4 * adaptive_output + 0.6 * clip_output
            # 计算CLIP蒸馏损失
            if hasattr(self.clip_gnn, 'compute_distillation_loss'):
                clip_loss = self.clip_gnn.compute_distillation_loss()
                
        elif self.use_adaptive_gnn:
            # 仅使用基础自适应GNN
            level3_logits = self.adaptive_gnn(level3_logits)
            
        elif self.use_clip_gnn:
            # 仅使用CLIP引导GNN
            level3_logits = self.clip_gnn(level3_logits)
            # 计算CLIP蒸馏损失
            if hasattr(self.clip_gnn, 'compute_distillation_loss'):
                clip_loss = self.clip_gnn.compute_distillation_loss()
        
        # 如果都不使用，level3_logits保持原样
        
        # 根据参数决定是否返回CLIP损失
        if return_clip_loss:
            return level2_logits, level3_logits, clip_loss
        else:
            return level2_logits, level3_logits


if __name__ == '__main__':
    # 简单测试网络前向
    print("=== 测试BiCAF融合版本 ===")
    model_ca = DualEncoderUNetPP(
        num_classes_level2=6,  # 二级类别数
        num_classes_level3=16, # 三级类别数
        use_adaptive_gnn=False,  # 不使用基础GNN
        use_clip_gnn=True,       # 使用CLIP GNN
        use_cross_attention_fusion=True,  # 使用Cross-Attention融合
        cross_attention_reduction=8,  # 通道缩减倍数
        use_multi_scale_fusion=False  # 不使用多尺度融合
    )
    
    # 计算参数量
    total_params_ca = sum(p.numel() for p in model_ca.parameters())
    trainable_params_ca = sum(p.numel() for p in model_ca.parameters() if p.requires_grad)
    
    # 统计BiCAF融合模块的参数量
    fusion_params = 0
    if hasattr(model_ca, 'fusion_modules'):
        for module in model_ca.fusion_modules:
            if not isinstance(module, nn.Identity):
                fusion_params += sum(p.numel() for p in module.parameters())
    
    print(f'总参数量: {total_params_ca:,}')
    print(f'可训练参数量: {trainable_params_ca:,}')
    print(f'BiCAF融合模块参数量: {fusion_params:,} ({fusion_params/total_params_ca*100:.2f}%)')
    
    print("\n=== 对比简单相加融合版本 ===")
    model_simple = DualEncoderUNetPP(
        num_classes_level2=6,
        num_classes_level3=16,
        use_adaptive_gnn=False,
        use_clip_gnn=True,
        use_cross_attention_fusion=False  # 使用简单相加
    )
    
    total_params_simple = sum(p.numel() for p in model_simple.parameters())
    print(f'简单相加版本总参数量: {total_params_simple:,}')
    print(f'BiCAF版本增加参数量: {total_params_ca - total_params_simple:,} '
          f'({(total_params_ca - total_params_simple)/total_params_simple*100:.2f}%)')
    
    # 使用较小的输入尺寸进行测试
    img = torch.randn(2, 3, 512, 512)  # 降低测试分辨率
    pm  = torch.randn(2, 1, 512, 512)
    
    # 使用torch.cuda.amp进行混合精度计算以进一步节省内存
    with torch.cuda.amp.autocast(enabled=True):
        level2_out, level3_out = model_ca(img, pm)
    
    print('\n=== 输出形状 ===')
    print('Level2 output shape:', level2_out.shape)  # 预期 [2,6,512,512]
    print('Level3 output shape:', level3_out.shape)  # 预期 [2,16,512,512]
    print('\n特征融合已更新为BiCAF双向交叉注意力方法，在保持低参数量的同时提升融合效果')
