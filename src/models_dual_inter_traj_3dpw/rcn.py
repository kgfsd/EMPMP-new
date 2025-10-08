"""
Relation Convolutional Network (RCN) Module
实现基础RCN架构，支持多人及可变人数场景的关系建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationEncoder(nn.Module):
    """
    关系编码器：编码人与人之间的空间和时间关系
    支持可变人数的动态掩码机制
    """
    def __init__(self, input_dim, hidden_dim, relation_types=3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            relation_types: 关系类型数量（空间、时间、交互等）
        """
        super(RelationEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_types = relation_types
        
        # 空间关系编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间关系编码
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 交互关系编码
        self.interaction_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 关系融合层
        self.relation_fusion = nn.Linear(hidden_dim * relation_types, hidden_dim)
        
    def forward(self, features, distances, mask=None):
        """
        Args:
            features: [B, P, D] - 人的特征
            distances: [B, P, P] - 人与人之间的距离
            mask: [B, P] - 有效人的掩码（1表示有效，0表示padding）
        Returns:
            relations: [B, P, P, hidden_dim] - 编码后的关系特征
        """
        B, P, D = features.shape
        
        # 扩展特征用于计算成对关系
        feat_i = features.unsqueeze(2).expand(B, P, P, D)  # [B, P, P, D]
        feat_j = features.unsqueeze(1).expand(B, P, P, D)  # [B, P, P, D]
        
        # 空间关系：基于位置距离
        dist_feat = distances.unsqueeze(-1)  # [B, P, P, 1]
        spatial_input = torch.cat([feat_i, feat_j, dist_feat], dim=-1)
        spatial_relations = self.spatial_encoder(spatial_input)  # [B, P, P, hidden_dim]
        
        # 时间关系：基于特征差异
        temporal_input = torch.cat([feat_i, feat_j], dim=-1)
        temporal_relations = self.temporal_encoder(temporal_input)
        
        # 交互关系：基于特征乘积
        interaction_input = torch.cat([feat_i * feat_j, feat_i + feat_j], dim=-1)
        interaction_relations = self.interaction_encoder(interaction_input)
        
        # 融合所有关系类型
        all_relations = torch.cat([spatial_relations, temporal_relations, interaction_relations], dim=-1)
        relations = self.relation_fusion(all_relations)  # [B, P, P, hidden_dim]
        
        # 应用掩码（如果提供）
        if mask is not None:
            mask_pair = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, P, P]
            relations = relations * mask_pair.unsqueeze(-1)
        
        return relations


class RelationConvLayer(nn.Module):
    """
    关系卷积层：在关系图上进行卷积操作
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(RelationConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 节点特征转换
        self.node_transform = nn.Linear(in_channels, out_channels)
        
        # 关系特征转换
        self.relation_transform = nn.Linear(in_channels, out_channels)
        
        # 聚合权重
        self.aggregation_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, node_features, relation_features, mask=None):
        """
        Args:
            node_features: [B, P, D_in] - 节点特征
            relation_features: [B, P, P, D_in] - 关系特征
            mask: [B, P] - 有效节点掩码
        Returns:
            output: [B, P, D_out] - 更新后的节点特征
        """
        B, P, D_in = node_features.shape
        
        # 转换节点特征
        node_feat = self.node_transform(node_features)  # [B, P, D_out]
        
        # 转换关系特征
        relation_feat = self.relation_transform(relation_features)  # [B, P, P, D_out]
        
        # 聚合邻居信息
        # 对每个节点i，聚合所有与它相关的节点j的信息
        aggregated = torch.sum(relation_feat, dim=2)  # [B, P, D_out]
        
        # 应用掩码
        if mask is not None:
            aggregated = aggregated * mask.unsqueeze(-1)
            # 归一化：除以有效邻居数
            valid_neighbors = mask.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated = aggregated / valid_neighbors.unsqueeze(-1)
        
        # 残差连接
        output = node_feat + self.aggregation_weight * aggregated
        
        return output


class BasicRCN(nn.Module):
    """
    基础RCN模块
    第一阶段：实现基础架构，支持可变人数
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, max_persons=5):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: RCN层数
            max_persons: 支持的最大人数
        """
        super(BasicRCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_persons = max_persons
        
        # 关系编码器
        self.relation_encoder = RelationEncoder(input_dim, hidden_dim)
        
        # 多层关系卷积
        self.conv_layers = nn.ModuleList([
            RelationConvLayer(hidden_dim if i > 0 else input_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, features, distances, mask=None):
        """
        Args:
            features: [B, P, T, D] - 输入特征（人，时间，特征维度）
            distances: [B, T, P, P] - 距离矩阵
            mask: [B, P] - 有效人的掩码
        Returns:
            output: [B, P, T, D] - 增强后的特征
        """
        B, P, T, D = features.shape
        
        # 处理每个时间步
        outputs = []
        for t in range(T):
            feat_t = features[:, :, t, :]  # [B, P, D]
            dist_t = distances[:, t, :, :] if distances.dim() == 4 else distances  # [B, P, P]
            
            # 编码关系
            relations = self.relation_encoder(feat_t, dist_t, mask)  # [B, P, P, hidden_dim]
            
            # 应用关系卷积层
            h = feat_t
            for i, (conv_layer, norm_layer) in enumerate(zip(self.conv_layers, self.layer_norms)):
                if i == 0:
                    # 第一层需要将relation也映射到hidden_dim
                    relations_h = self.relation_encoder.relation_fusion(
                        relations.view(B, P, P, -1)
                    )
                    h = conv_layer(h, relations_h, mask)
                else:
                    h = conv_layer(h, relations, mask)
                h = norm_layer(h)
                h = F.relu(h)
            
            # 输出投影
            out_t = self.output_proj(h)  # [B, P, D]
            outputs.append(out_t)
        
        # 合并时间维度
        output = torch.stack(outputs, dim=2)  # [B, P, T, D]
        
        # 残差连接
        output = features + output
        
        return output


def create_person_mask(num_persons, max_persons, device):
    """
    创建动态人数掩码
    Args:
        num_persons: [B] - 每个batch的实际人数
        max_persons: 最大人数
        device: 设备
    Returns:
        mask: [B, max_persons] - 掩码矩阵
    """
    B = len(num_persons)
    mask = torch.zeros(B, max_persons, device=device)
    for i, n in enumerate(num_persons):
        mask[i, :n] = 1.0
    return mask
