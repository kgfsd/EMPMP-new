"""
Optimized RCN with Hierarchical Relations and Advanced Feature Extraction
第三阶段：优化关系类型和特征提取，提高效率和性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRelationType(nn.Module):
    """
    层次化关系类型编码器
    支持多层次的关系建模：
    1. 空间关系（Spatial）- 基于位置距离
    2. 时序关系（Temporal）- 基于运动模式
    3. 交互关系（Interactive）- 基于行为相似性
    """
    def __init__(self, feature_dim, relation_dim, use_learnable_weights=True):
        """
        Args:
            feature_dim: 特征维度
            relation_dim: 关系表示维度
            use_learnable_weights: 是否使用可学习的关系权重
        """
        super(HierarchicalRelationType, self).__init__()
        self.feature_dim = feature_dim
        self.relation_dim = relation_dim
        self.use_learnable_weights = use_learnable_weights
        
        # 空间关系编码 - 轻量级设计
        self.spatial_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, relation_dim // 2),  # +3 for (distance, angle, height_diff)
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim // 2, relation_dim)
        )
        
        # 时序关系编码 - 捕获运动趋势
        self.temporal_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, relation_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim // 2, relation_dim)
        )
        
        # 交互关系编码 - 学习高层语义
        self.interaction_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, relation_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim // 2, relation_dim)
        )
        
        # 可学习的关系类型权重
        if use_learnable_weights:
            self.relation_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            self.register_buffer('relation_weights', torch.ones(3) / 3)
        
        # 关系融合网络 - 使用1x1卷积减少参数
        self.fusion_net = nn.Conv1d(relation_dim * 3, relation_dim, kernel_size=1)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(relation_dim)
        
    def compute_spatial_features(self, positions_i, positions_j):
        """
        计算空间几何特征
        Args:
            positions_i, positions_j: [B, P, P, 3] - 人的3D位置
        Returns:
            spatial_feats: [B, P, P, 3] - (distance, angle, height_diff)
        """
        # 欧式距离
        diff = positions_j - positions_i
        distance = torch.norm(diff, dim=-1, keepdim=True)
        
        # 水平角度 (XZ平面)
        angle = torch.atan2(diff[..., 0:1], diff[..., 2:3])
        
        # 高度差
        height_diff = diff[..., 1:2]
        
        return torch.cat([distance, angle, height_diff], dim=-1)
    
    def forward(self, features_i, features_j, positions_i=None, positions_j=None, velocities_i=None, velocities_j=None):
        """
        Args:
            features_i, features_j: [B, P, P, D] - 成对的人特征
            positions_i, positions_j: [B, P, P, 3] - 位置信息（可选）
            velocities_i, velocities_j: [B, P, P, D] - 速度信息（可选）
        Returns:
            relations: [B, P, P, relation_dim] - 层次化关系表示
        """
        B, P, _, D = features_i.shape
        
        # 1. 空间关系
        if positions_i is not None and positions_j is not None:
            spatial_geom = self.compute_spatial_features(positions_i, positions_j)
            spatial_input = torch.cat([features_i, features_j, spatial_geom], dim=-1)
        else:
            # 如果没有位置信息，使用特征距离
            feat_dist = torch.norm(features_i - features_j, dim=-1, keepdim=True)
            spatial_input = torch.cat([features_i, features_j, feat_dist.expand(-1, -1, -1, 2)], dim=-1)
        
        spatial_relations = self.spatial_encoder(spatial_input)
        
        # 2. 时序关系
        if velocities_i is not None and velocities_j is not None:
            temporal_input = torch.cat([velocities_i, velocities_j], dim=-1)
        else:
            # 使用特征作为替代
            temporal_input = torch.cat([features_i, features_j], dim=-1)
        
        temporal_relations = self.temporal_encoder(temporal_input)
        
        # 3. 交互关系 - 使用点积和加法捕获相似性和互补性
        interaction_input = torch.cat([features_i * features_j, features_i + features_j], dim=-1)
        interaction_relations = self.interaction_encoder(interaction_input)
        
        # 加权融合
        weights = F.softmax(self.relation_weights, dim=0)
        weighted_relations = torch.cat([
            spatial_relations * weights[0],
            temporal_relations * weights[1],
            interaction_relations * weights[2]
        ], dim=-1)
        
        # 1x1卷积融合（降低计算量）
        # 重塑为 [B*P*P, 3*relation_dim, 1]
        weighted_relations = weighted_relations.view(B * P * P, -1, 1)
        relations = self.fusion_net(weighted_relations).squeeze(-1)  # [B*P*P, relation_dim]
        relations = relations.view(B, P, P, self.relation_dim)
        
        # 归一化
        relations = self.layer_norm(relations)
        
        return relations


class EfficientFeatureExtractor(nn.Module):
    """
    高效特征提取器
    使用深度可分离卷积和组卷积减少参数量
    """
    def __init__(self, in_channels, out_channels, groups=4):
        super(EfficientFeatureExtractor, self).__init__()
        
        # 深度可分离卷积
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 激活函数
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: [B, P, T, D] - 输入特征
        Returns:
            out: [B, P, T, D'] - 提取后的特征
        """
        B, P, T, D = x.shape
        
        # 重塑为 [B*P, D, T]
        x = x.view(B * P, D, T).contiguous()
        
        # 深度可分离卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # 重塑回 [B, P, T, D']
        _, D_out, _ = x.shape
        x = x.view(B, P, T, D_out).contiguous()
        
        return x


class LearnableRelationWeighting(nn.Module):
    """
    可学习的关系权重模块
    动态调整不同人之间的重要性
    """
    def __init__(self, relation_dim, temperature=1.0):
        super(LearnableRelationWeighting, self).__init__()
        self.temperature = temperature
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(relation_dim, relation_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim // 4, 1)
        )
        
    def forward(self, relations, mask=None):
        """
        Args:
            relations: [B, P, P, D] - 关系特征
            mask: [B, P, P] - 有效关系掩码
        Returns:
            weighted_relations: [B, P, P, D] - 加权后的关系
            weights: [B, P, P] - 权重矩阵
        """
        # 预测权重
        weights = self.weight_predictor(relations).squeeze(-1)  # [B, P, P]
        
        # 应用温度参数
        weights = weights / self.temperature
        
        # 应用掩码并归一化
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(weights, dim=-1)  # 对每个人的所有关系归一化
        
        # 加权
        weighted_relations = relations * weights.unsqueeze(-1)
        
        return weighted_relations, weights


class OptimizedRCN(nn.Module):
    """
    优化后的RCN模型
    第三阶段：完整实现，包含层次化关系、高效特征提取和可学习权重
    """
    def __init__(self, input_dim, hidden_dim, relation_dim, num_layers=3, 
                 num_heads=4, max_persons=5, lightweight=True):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            relation_dim: 关系表示维度
            num_layers: 网络层数
            num_heads: 注意力头数（如果使用）
            max_persons: 支持的最大人数
            lightweight: 是否使用轻量化设计
        """
        super(OptimizedRCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim
        self.num_layers = num_layers
        self.lightweight = lightweight
        
        # 高效特征提取
        if lightweight:
            self.feature_extractor = EfficientFeatureExtractor(input_dim, hidden_dim, groups=4)
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
        
        # 层次化关系编码器
        self.relation_encoder = HierarchicalRelationType(
            hidden_dim, relation_dim, use_learnable_weights=True
        )
        
        # 可学习关系权重
        self.relation_weighting = LearnableRelationWeighting(relation_dim)
        
        # RCN层
        self.rcn_layers = nn.ModuleList([
            self._build_rcn_layer(hidden_dim, relation_dim)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # 可选：添加简化的注意力机制
        if not lightweight:
            from .rcn_attention import EfficientAttention
            self.attention = EfficientAttention(hidden_dim, num_heads)
        else:
            self.attention = None
        
    def _build_rcn_layer(self, hidden_dim, relation_dim):
        """构建单个RCN层"""
        return nn.ModuleDict({
            'relation_conv': nn.Linear(hidden_dim + relation_dim, hidden_dim),
            'self_conv': nn.Linear(hidden_dim, hidden_dim),
            'norm': nn.LayerNorm(hidden_dim),
            'activation': nn.ReLU(inplace=True)
        })
    
    def _apply_rcn_layer(self, layer, node_features, relation_features, mask=None):
        """
        应用单个RCN层
        """
        B, P, D = node_features.shape
        
        # 节点自身的变换
        self_feat = layer['self_conv'](node_features)
        
        # 关系聚合
        # [B, P, P, D] -> 对每个节点聚合所有关系
        relation_agg = relation_features.mean(dim=2)  # [B, P, D_rel]
        
        # 拼接节点特征和关系特征
        combined = torch.cat([node_features, relation_agg], dim=-1)
        relation_feat = layer['relation_conv'](combined)
        
        # 残差连接
        output = self_feat + relation_feat
        output = layer['norm'](output)
        output = layer['activation'](output)
        
        return output
    
    def forward(self, features, distances=None, positions=None, velocities=None, mask=None):
        """
        Args:
            features: [B, P, T, D] - 输入特征
            distances: [B, T, P, P] - 距离矩阵（可选）
            positions: [B, P, T, 3] - 位置信息（可选）
            velocities: [B, P, T, D] - 速度信息（可选）
            mask: [B, P] - 有效节点掩码
        Returns:
            output: [B, P, T, D] - 增强后的特征
            info: dict - 包含中间信息（用于分析）
        """
        B, P, T, D = features.shape
        
        # 高效特征提取
        if self.lightweight:
            features_extracted = self.feature_extractor(features)  # [B, P, T, hidden_dim]
        else:
            features_extracted = self.feature_extractor(features.view(B * P * T, D))
            features_extracted = features_extracted.view(B, P, T, self.hidden_dim)
        
        # 处理每个时间步
        outputs = []
        relation_weights_list = []
        
        for t in range(T):
            feat_t = features_extracted[:, :, t, :]  # [B, P, hidden_dim]
            
            # 准备成对特征
            feat_i = feat_t.unsqueeze(2).expand(B, P, P, self.hidden_dim)
            feat_j = feat_t.unsqueeze(1).expand(B, P, P, self.hidden_dim)
            
            # 准备位置和速度（如果有）
            pos_i = pos_j = None
            vel_i = vel_j = None
            if positions is not None:
                pos_t = positions[:, :, t, :]  # [B, P, 3]
                pos_i = pos_t.unsqueeze(2).expand(B, P, P, 3)
                pos_j = pos_t.unsqueeze(1).expand(B, P, P, 3)
            if velocities is not None:
                vel_t = velocities[:, :, t, :]
                vel_i = vel_t.unsqueeze(2).expand(B, P, P, self.hidden_dim)
                vel_j = vel_t.unsqueeze(1).expand(B, P, P, self.hidden_dim)
            
            # 编码层次化关系
            relations = self.relation_encoder(feat_i, feat_j, pos_i, pos_j, vel_i, vel_j)
            
            # 可学习的关系权重
            mask_pair = None
            if mask is not None:
                mask_pair = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, P, P]
            weighted_relations, rel_weights = self.relation_weighting(relations, mask_pair)
            relation_weights_list.append(rel_weights)
            
            # 应用RCN层
            h = feat_t
            for rcn_layer in self.rcn_layers:
                h = self._apply_rcn_layer(rcn_layer, h, weighted_relations, mask)
            
            # 可选：应用注意力
            if self.attention is not None:
                h = h + self.attention(h, mask)
            
            outputs.append(h)
        
        # 合并时间维度
        output = torch.stack(outputs, dim=2)  # [B, P, T, hidden_dim]
        
        # 输出投影
        output = self.output_proj(output.view(B * P * T, self.hidden_dim)).view(B, P, T, D)
        
        # 残差连接
        output = features + output
        
        # 返回额外信息
        info = {
            'relation_weights': relation_weights_list,
            'num_params': sum(p.numel() for p in self.parameters())
        }
        
        return output, info


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimized_rcn(input_dim=39, hidden_dim=64, relation_dim=32, 
                         num_layers=3, lightweight=True):
    """
    创建优化的RCN模型
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        relation_dim: 关系维度
        num_layers: 层数
        lightweight: 是否轻量化
    
    Returns:
        model: OptimizedRCN实例
        num_params: 参数量（M）
    """
    model = OptimizedRCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        relation_dim=relation_dim,
        num_layers=num_layers,
        lightweight=lightweight
    )
    
    num_params = count_parameters(model) / 1e6  # 转换为M
    
    print(f"OptimizedRCN created:")
    print(f"  - Parameters: {num_params:.2f}M")
    print(f"  - Lightweight: {lightweight}")
    print(f"  - Layers: {num_layers}")
    
    return model, num_params
