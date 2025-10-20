"""
GCN+Stylization多人运动预测模型
集成IPLM交互先验学习模块，支持可变人数场景
"""
import copy
from torch import nn
from .mlp_gcn_stylization import build_mlps_gcn_stylization
from einops.layers.torch import Rearrange
import torch


def compute_distances_hierarchical_normalization(positions, zero_score=True, padding_mask=None):
    """
    计算人与人之间的标准化距离矩阵
    
    Args:
        positions: [B, P, T, J, K] - 人员位置数据
        zero_score: bool - 是否进行零均值标准化
        padding_mask: [B, P] - 有效人员掩码
    
    Returns:
        distances: [B, T, P, P] - 标准化距离矩阵
    """
    # 使用根节点位置计算距离
    positions = positions[:, :, :, 0, :]  # [B, P, T, K]
    B, P, T, K = positions.size()
    
    # 计算每帧质心并归一化位置
    if padding_mask is not None:
        mask_expanded = padding_mask.unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]
        valid_positions = positions * mask_expanded.float()
        num_valid = padding_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1, 1]
        num_valid = torch.clamp(num_valid, min=1)
        centroid = valid_positions.sum(dim=1, keepdim=True) / num_valid  # [B, 1, T, K]
    else:
        centroid = positions.mean(dim=1, keepdim=True)  # [B, 1, T, K]
    
    normalized_positions = positions - centroid  # [B, P, T, K]

    # 计算成对欧氏距离
    positions_exp1 = normalized_positions.unsqueeze(2)  # [B, P, 1, T, K]
    positions_exp2 = normalized_positions.unsqueeze(1)  # [B, 1, P, T, K]
    dist = torch.norm(positions_exp1 - positions_exp2, dim=-1)  # [B, P, P, T]
    distances = dist.permute(0, 3, 1, 2)  # [B, T, P, P]

    # 标准化距离
    if zero_score:
        if padding_mask is not None:
            valid_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)  # [B, P, P]
            valid_mask = valid_mask.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, P, P]
            valid_distances = distances * valid_mask.float()
            
            num_valid_connections = valid_mask.sum(dim=(2, 3), keepdim=True).float()
            num_valid_connections = torch.clamp(num_valid_connections, min=1)
            
            mean_distances = valid_distances.sum(dim=(2, 3), keepdim=True) / num_valid_connections
            diff_squared = (valid_distances - mean_distances) ** 2 * valid_mask.float()
            std_distances = torch.sqrt(diff_squared.sum(dim=(2, 3), keepdim=True) / num_valid_connections)
            
            standardized_distances = (distances - mean_distances) / (std_distances + 1e-8)
        else:
            mean_distances = distances.mean(dim=(2, 3), keepdim=True)  # [B, T, 1, 1]
            std_distances = distances.std(dim=(2, 3), keepdim=True)  # [B, T, 1, 1]
            standardized_distances = (distances - mean_distances) / (std_distances + 1e-8)
    else:
        standardized_distances = distances
    
    return standardized_distances  # [B, T, P, P]


class siMLPe_GCN_Stylization(nn.Module):
    """
    GCN+Stylization多人运动预测模型
    
    特点:
    - 图卷积网络建模人际交互
    - Stylization机制进行特征调制
    - 支持可变人数场景
    - 集成IPLM交互先验学习
    """
    def __init__(self, config):
        super(siMLPe_GCN_Stylization, self).__init__()
        self.config = copy.deepcopy(config)
        
        # 维度重排层
        self.arr0 = Rearrange('b p n d -> b p d n')  # [B,P,T,D] -> [B,P,D,T]
        self.arr1 = Rearrange('b p d n -> b p n d')  # [B,P,D,T] -> [B,P,T,D]

        # 核心处理模块: GCN+Stylization+IPLM
        self.motion_mlp = build_mlps_gcn_stylization(self.config.motion_mlp)

        # 输入输出配置
        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
    
        # 输入特征变换
        if self.temporal_fc_in:
            # 时序维度变换: [B,P,D,T] -> [B,P,D,T]
            self.motion_fc_in = nn.Linear(
                self.config.motion.h36m_input_length_dct, 
                self.config.motion.h36m_input_length_dct
            )
        else:
            # 特征维度变换: [B,P,T,D] -> [B,P,T,hidden_dim]
            self.motion_fc_in = nn.Linear(
                self.config.motion.dim, 
                self.config.motion_mlp.hidden_dim
            )
            
        # 输出特征变换
        if self.temporal_fc_out:
            # 时序维度变换: [B,P,D,T] -> [B,P,D,T]
            self.motion_fc_out = nn.Linear(
                self.config.motion.h36m_input_length_dct, 
                self.config.motion.h36m_input_length_dct
            )
        else:
            # 特征维度变换: [B,P,T,hidden_dim] -> [B,P,T,D]
            self.motion_fc_out = nn.Linear(
                self.config.motion_mlp.hidden_dim, 
                self.config.motion.dim
            )

        self.reset_parameters()

    def reset_parameters(self):
        """初始化输出层参数"""
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, traj, padding_mask=None):
        """
        前向传播
        
        Args:
            motion_input: [B, P, T, D] - 运动特征输入
            traj: [B, P, T, J, K] - 轨迹数据，用于计算人际距离
            padding_mask: [B, P] - 有效人员掩码
        
        Returns:
            motion_feats: [B, P, T, D] - 预测的运动特征
            loss_lk: float - IPLM损失 (如果启用)
        """
        # 1. 计算人际距离矩阵
        distances = compute_distances_hierarchical_normalization(
            traj, 
            zero_score=False, 
            padding_mask=padding_mask
        )  # [B, T, P, P]

        # 2. 输入特征处理
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)  # [B, P, T, D] -> [B, P, D, T]
            motion_feats = self.motion_fc_in(motion_feats)  # [B, P, D, T]
        else:
            motion_feats = self.motion_fc_in(motion_input)  # [B, P, T, D] -> [B, P, T, hidden_dim]
            motion_feats = self.arr0(motion_feats)  # [B, P, T, hidden_dim] -> [B, P, hidden_dim, T]

        # 3. 核心处理: GCN+Stylization+IPLM
        if hasattr(self.motion_mlp, 'use_iplm') and self.motion_mlp.use_iplm:
            motion_feats, loss_lk = self.motion_mlp(
                motion_feats, 
                distances=distances, 
                padding_mask=padding_mask
            )  # [B, P, D, T], loss
        else:
            motion_feats = self.motion_mlp(
                motion_feats, 
                distances=distances, 
                padding_mask=padding_mask
            )  # [B, P, D, T]
            loss_lk = None

        # 4. 输出特征处理
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)  # [B, P, D, T]
            motion_feats = self.arr1(motion_feats)  # [B, P, D, T] -> [B, P, T, D]
        else:
            motion_feats = self.arr1(motion_feats)  # [B, P, D, T] -> [B, P, T, D]
            motion_feats = self.motion_fc_out(motion_feats)  # [B, P, T, D]

        return motion_feats, loss_lk


# 模型别名，保持向后兼容
Model = siMLPe_GCN_Stylization
