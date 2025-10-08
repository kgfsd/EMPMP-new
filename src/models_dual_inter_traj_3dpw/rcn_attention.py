"""
RCN with Attention Mechanism
第二阶段：为RCN添加注意力机制，提高关系建模能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadRelationAttention(nn.Module):
    """
    多头关系注意力机制
    用于自适应地学习人与人之间的重要关系
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            dropout: dropout比率
        """
        super(MultiHeadRelationAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V投影
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # 关系偏置编码
        self.relation_bias = nn.Linear(dim, num_heads)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, relation_features, mask=None):
        """
        Args:
            features: [B, P, D] - 节点特征
            relation_features: [B, P, P, D] - 关系特征
            mask: [B, P] - 有效节点掩码
        Returns:
            output: [B, P, D] - 注意力增强后的特征
            attention_weights: [B, num_heads, P, P] - 注意力权重（用于可视化）
        """
        B, P, D = features.shape
        
        # Q, K, V投影并重塑为多头形式
        Q = self.query(features).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, P, d]
        K = self.key(features).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, P, d]
        V = self.value(features).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, P, d]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, P, P]
        
        # 添加关系偏置
        relation_bias = self.relation_bias(relation_features)  # [B, P, P, H]
        relation_bias = relation_bias.permute(0, 3, 1, 2)  # [B, H, P, P]
        attention_scores = attention_scores + relation_bias
        
        # 应用掩码
        if mask is not None:
            # 创建成对掩码
            mask_pair = mask.unsqueeze(1).unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)  # [B, 1, P, P]
            attention_scores = attention_scores.masked_fill(mask_pair == 0, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, H, P, P]
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [B, H, P, d]
        
        # 重塑并投影输出
        output = output.transpose(1, 2).contiguous().view(B, P, D)  # [B, P, D]
        output = self.out_proj(output)
        
        return output, attention_weights


class AdaptiveRelationWeighting(nn.Module):
    """
    自适应关系权重计算
    根据输入特征动态调整不同关系类型的重要性
    """
    def __init__(self, dim, relation_types=3):
        """
        Args:
            dim: 特征维度
            relation_types: 关系类型数量
        """
        super(AdaptiveRelationWeighting, self).__init__()
        self.relation_types = relation_types
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, relation_types),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, P, D] - 输入特征
        Returns:
            weights: [B, relation_types] - 关系类型权重
        """
        B, P, D = features.shape
        
        # 全局池化
        global_feat = features.mean(dim=1)  # [B, D]
        
        # 生成权重
        weights = self.weight_generator(global_feat)  # [B, relation_types]
        
        return weights


class AttentionRCNLayer(nn.Module):
    """
    带注意力机制的RCN层
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, relation_types=3):
        super(AttentionRCNLayer, self).__init__()
        
        # 多头关系注意力
        self.relation_attention = MultiHeadRelationAttention(dim, num_heads, dropout)
        
        # 自适应权重
        self.adaptive_weighting = AdaptiveRelationWeighting(dim, relation_types)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, features, relation_features, mask=None):
        """
        Args:
            features: [B, P, D] - 节点特征
            relation_features: [B, P, P, D] - 关系特征
            mask: [B, P] - 有效节点掩码
        Returns:
            output: [B, P, D] - 更新后的特征
            attention_weights: 注意力权重
        """
        # 计算自适应权重
        adaptive_weights = self.adaptive_weighting(features)  # [B, relation_types]
        
        # 多头关系注意力 + 残差连接
        attn_output, attention_weights = self.relation_attention(features, relation_features, mask)
        features = self.norm1(features + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)
        
        return features, attention_weights


class RCNWithAttention(nn.Module):
    """
    完整的带注意力机制的RCN模块
    第二阶段：集成注意力机制
    """
    def __init__(self, input_dim, hidden_dim, num_layers=4, num_heads=8, dropout=0.1, max_persons=5):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: RCN层数
            num_heads: 注意力头数
            dropout: dropout比率
            max_persons: 支持的最大人数
        """
        super(RCNWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_persons = max_persons
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 关系编码器（复用第一阶段的）
        from .rcn import RelationEncoder
        self.relation_encoder = RelationEncoder(hidden_dim, hidden_dim)
        
        # 多层注意力RCN
        self.attention_layers = nn.ModuleList([
            AttentionRCNLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # 用于存储注意力权重（可选，用于可视化）
        self.attention_weights = []
        
    def forward(self, features, distances, mask=None, return_attention=False):
        """
        Args:
            features: [B, P, T, D] - 输入特征
            distances: [B, T, P, P] - 距离矩阵
            mask: [B, P] - 有效节点掩码
            return_attention: 是否返回注意力权重
        Returns:
            output: [B, P, T, D] - 增强后的特征
            attention_weights: (可选) 注意力权重列表
        """
        B, P, T, D = features.shape
        
        self.attention_weights = []
        
        # 输入投影
        features_proj = self.input_proj(features.view(B * P * T, D)).view(B, P, T, self.hidden_dim)
        
        # 处理每个时间步
        outputs = []
        for t in range(T):
            feat_t = features_proj[:, :, t, :]  # [B, P, hidden_dim]
            dist_t = distances[:, t, :, :] if distances.dim() == 4 else distances
            
            # 编码关系
            relations = self.relation_encoder(feat_t, dist_t, mask)  # [B, P, P, hidden_dim]
            
            # 应用注意力RCN层
            h = feat_t
            layer_attention_weights = []
            for attention_layer in self.attention_layers:
                h, attn_weights = attention_layer(h, relations, mask)
                if return_attention:
                    layer_attention_weights.append(attn_weights)
            
            if return_attention:
                self.attention_weights.append(layer_attention_weights)
            
            outputs.append(h)
        
        # 合并时间维度
        output = torch.stack(outputs, dim=2)  # [B, P, T, hidden_dim]
        
        # 输出投影
        output = self.output_proj(output.view(B * P * T, self.hidden_dim)).view(B, P, T, D)
        
        # 残差连接
        output = features + output
        
        if return_attention:
            return output, self.attention_weights
        return output


class EfficientAttention(nn.Module):
    """
    高效注意力机制
    使用线性复杂度的注意力计算，适合多人场景
    """
    def __init__(self, dim, num_heads=8):
        super(EfficientAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        """
        线性注意力：O(P*d^2) 而非 O(P^2*d)
        """
        B, P, D = x.shape
        
        Q = self.query(x).view(B, P, self.num_heads, self.head_dim)
        K = self.key(x).view(B, P, self.num_heads, self.head_dim)
        V = self.value(x).view(B, P, self.num_heads, self.head_dim)
        
        # 使用核技巧降低复杂度
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # K^T * V: [B, H, d, d]
        KV = torch.einsum('bphd,bphe->bhde', K, V)
        
        # Q * (K^T * V): [B, P, H, d]
        output = torch.einsum('bphd,bhde->bphe', Q, KV)
        
        # 归一化
        Z = torch.einsum('bphd,bhd->bph', Q, K.sum(dim=1))
        output = output / (Z.unsqueeze(-1) + 1e-6)
        
        output = output.reshape(B, P, D)
        output = self.out_proj(output)
        
        return output
