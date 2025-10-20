"""
Model file for GCN+Stylization version
配套 mlp_gcn_stylization.py 使用
"""
import copy
from torch import nn
from .mlp_gcn_stylization import build_mlps_gcn_stylization
from einops.layers.torch import Rearrange
import torch


def compute_distances_hierarchical_normalization(positions, zero_score=True, padding_mask=None):
    """
    计算人与人之间的标准化距离
    positions: [B, P, T, J, K]
    返回: [B, T, P, P]
    """
    positions = positions[:, :, :, 0, :]  # B, P, T, K (使用第一个关节,如根节点)
    B, P, T, K = positions.size()
    
    # Step 1: 计算每帧的质心并归一化位置
    if padding_mask is not None:
        # 只使用有效人员计算质心
        mask_expanded = padding_mask.unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]
        valid_positions = positions * mask_expanded.float()
        num_valid = padding_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1, 1]
        num_valid = torch.clamp(num_valid, min=1)  # 避免除零
        centroid = valid_positions.sum(dim=1, keepdim=True) / num_valid  # [B, 1, T, K]
    else:
        centroid = positions.mean(dim=1, keepdim=True)
    
    normalized_positions = positions - centroid

    # Step 2: 计算成对欧氏距离
    positions_exp1 = normalized_positions.unsqueeze(2)  # Shape: [B, P, 1, T, K]
    positions_exp2 = normalized_positions.unsqueeze(1)  # Shape: [B, 1, P, T, K]
    dist = torch.norm(positions_exp1 - positions_exp2, dim=-1)  # Shape: [B, P, P, T]
    distances = dist.permute(0, 3, 1, 2)  # Shape: [B, T, P, P]

    # Step 3: 在 P, P 维度上标准化距离
    if zero_score:
        if padding_mask is not None:
            # 只使用有效距离进行标准化
            valid_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)  # [B, P, P]
            valid_mask = valid_mask.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, P, P]
            valid_distances = distances * valid_mask.float()
            
            # 只对有效连接计算均值和标准差
            num_valid_connections = valid_mask.sum(dim=(2, 3), keepdim=True).float()
            num_valid_connections = torch.clamp(num_valid_connections, min=1)
            
            mean_distances = valid_distances.sum(dim=(2, 3), keepdim=True) / num_valid_connections
            
            # 计算标准差
            diff_squared = (valid_distances - mean_distances) ** 2 * valid_mask.float()
            std_distances = torch.sqrt(diff_squared.sum(dim=(2, 3), keepdim=True) / num_valid_connections)
            
            standardized_distances = (distances - mean_distances) / (std_distances + 1e-8)
        else:
            mean_distances = distances.mean(dim=(2, 3), keepdim=True)
            std_distances = distances.std(dim=(2, 3), keepdim=True)
            standardized_distances = (distances - mean_distances) / (std_distances + 1e-8)
    else:
        standardized_distances = distances
    
    return standardized_distances  # Shape: [B, T, P, P]


class siMLPe_GCN_Stylization(nn.Module):
    """
    GCN + Stylization版本的siMLPe模型
    - 用GCN建模人际交互
    - 保留scale/shift调制机制
    - 完全消除人数P依赖
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe_GCN_Stylization, self).__init__()
        
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')

        # 使用GCN+Stylization版本的MLP
        self.motion_mlp = build_mlps_gcn_stylization(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
    
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.motion.h36m_input_length_dct, 
                self.config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_in = nn.Linear(
                self.config.motion.dim, 
                self.config.motion_mlp.hidden_dim
            )
            
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.motion.h36m_input_length_dct, 
                self.config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_out = nn.Linear(
                self.config.motion_mlp.hidden_dim, 
                self.config.motion.dim
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, traj, padding_mask=None):
        """
        motion_input: [B, P, T, D] - 运动特征输入
        traj: [B, P, T, J, K] - 轨迹数据,用于计算距离
        padding_mask: [B, P] - 有效人员掩码
        """
        # 计算人与人之间的距离矩阵
        distances = compute_distances_hierarchical_normalization(
            traj, 
            zero_score=False, 
            padding_mask=padding_mask
        )
        
        # 输入特征处理
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)  # B, P, D, T
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)  # B, P, T, D
            motion_feats = self.arr0(motion_feats)  # B, P, D, T

        # GCN+Stylization处理
        motion_feats = self.motion_mlp(motion_feats, distances, padding_mask)

        # 输出特征处理
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)  # B, P, T, D
            motion_feats = self.motion_fc_out(motion_feats)  # B, P, T, D

        return motion_feats
