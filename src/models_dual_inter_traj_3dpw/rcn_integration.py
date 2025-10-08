"""
RCN Integration Module
展示如何将RCN模块集成到现有的siMLPe模型中
"""

import torch
import torch.nn as nn
from .rcn import BasicRCN, create_person_mask
from .rcn_attention import RCNWithAttention
from .rcn_optimized import OptimizedRCN


class RCNEnhancedModel(nn.Module):
    """
    RCN增强的模型
    可以选择使用三个阶段中的任意一个
    """
    def __init__(self, original_model, rcn_type='optimized', **rcn_kwargs):
        """
        Args:
            original_model: 原始的siMLPe模型
            rcn_type: RCN类型 - 'basic', 'attention', 'optimized'
            rcn_kwargs: RCN模型的参数
        """
        super(RCNEnhancedModel, self).__init__()
        self.original_model = original_model
        self.rcn_type = rcn_type
        
        # 根据类型创建RCN模块
        if rcn_type == 'basic':
            self.rcn = BasicRCN(**rcn_kwargs)
        elif rcn_type == 'attention':
            self.rcn = RCNWithAttention(**rcn_kwargs)
        elif rcn_type == 'optimized':
            self.rcn = OptimizedRCN(**rcn_kwargs)
        else:
            raise ValueError(f"Unknown RCN type: {rcn_type}")
        
        # 特征融合
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, motion_input, traj, num_persons=None):
        """
        Args:
            motion_input: [B, P, T, D] - 输入运动序列
            traj: [B, P, T, J, K] - 轨迹信息
            num_persons: [B] - 每个batch的实际人数（可选）
        Returns:
            output: 融合后的输出
        """
        # 原始模型的输出
        original_output = self.original_model(motion_input, traj)
        
        # 计算距离矩阵
        from .model import compute_distances_hierarchical_normalization
        distances = compute_distances_hierarchical_normalization(traj, zero_score=False)
        
        # 创建掩码（如果提供人数）
        mask = None
        if num_persons is not None:
            B, P = motion_input.shape[:2]
            mask = create_person_mask(num_persons, P, motion_input.device)
        
        # RCN增强
        if self.rcn_type == 'optimized':
            # OptimizedRCN返回两个值
            rcn_output, info = self.rcn(motion_input, distances, mask=mask)
        elif self.rcn_type == 'attention':
            # RCNWithAttention可能返回注意力权重
            rcn_output = self.rcn(motion_input, distances, mask=mask)
        else:
            # BasicRCN
            rcn_output = self.rcn(motion_input, distances, mask=mask)
        
        # 融合原始输出和RCN输出
        fusion_weight = torch.sigmoid(self.fusion_weight)
        output = fusion_weight * original_output + (1 - fusion_weight) * rcn_output
        
        return output


def integrate_rcn_phase1(config):
    """
    第一阶段集成：基础RCN架构
    
    使用方法:
    ```python
    from src.models_dual_inter_traj_3dpw.model import siMLPe
    from src.models_dual_inter_traj_3dpw.rcn_integration import integrate_rcn_phase1
    
    # 创建原始模型
    original_model = siMLPe(config)
    
    # 集成RCN
    enhanced_model = integrate_rcn_phase1(config)
    ```
    """
    from .model import siMLPe
    
    # 创建原始模型
    original_model = siMLPe(config)
    
    # RCN参数配置
    rcn_kwargs = {
        'input_dim': config.motion.dim,
        'hidden_dim': 64,  # 较小的隐藏维度保持轻量
        'num_layers': 2,
        'max_persons': config.n_p
    }
    
    # 创建RCN增强模型
    enhanced_model = RCNEnhancedModel(
        original_model,
        rcn_type='basic',
        **rcn_kwargs
    )
    
    return enhanced_model


def integrate_rcn_phase2(config):
    """
    第二阶段集成：添加注意力机制
    
    使用方法同上，但使用attention类型
    """
    from .model import siMLPe
    
    original_model = siMLPe(config)
    
    rcn_kwargs = {
        'input_dim': config.motion.dim,
        'hidden_dim': 64,
        'num_layers': 3,  # 可以增加层数
        'num_heads': 4,   # 注意力头数
        'dropout': 0.1,
        'max_persons': config.n_p
    }
    
    enhanced_model = RCNEnhancedModel(
        original_model,
        rcn_type='attention',
        **rcn_kwargs
    )
    
    return enhanced_model


def integrate_rcn_phase3(config, lightweight=True):
    """
    第三阶段集成：优化的RCN
    
    Args:
        config: 配置对象
        lightweight: 是否使用轻量化版本
    """
    from .model import siMLPe
    
    original_model = siMLPe(config)
    
    rcn_kwargs = {
        'input_dim': config.motion.dim,
        'hidden_dim': 64 if lightweight else 128,
        'relation_dim': 32 if lightweight else 64,
        'num_layers': 3,
        'num_heads': 4,
        'max_persons': config.n_p,
        'lightweight': lightweight
    }
    
    enhanced_model = RCNEnhancedModel(
        original_model,
        rcn_type='optimized',
        **rcn_kwargs
    )
    
    return enhanced_model


class VariablePersonRCN(nn.Module):
    """
    支持可变人数的RCN模型
    可以处理2-N人的场景，通过动态掩码机制
    """
    def __init__(self, config, min_persons=2, max_persons=5, rcn_type='optimized'):
        super(VariablePersonRCN, self).__init__()
        self.config = config
        self.min_persons = min_persons
        self.max_persons = max_persons
        self.rcn_type = rcn_type
        
        # 创建RCN模型
        if rcn_type == 'optimized':
            self.rcn = OptimizedRCN(
                input_dim=config.motion.dim,
                hidden_dim=64,
                relation_dim=32,
                num_layers=3,
                max_persons=max_persons,
                lightweight=True
            )
        elif rcn_type == 'attention':
            self.rcn = RCNWithAttention(
                input_dim=config.motion.dim,
                hidden_dim=64,
                num_layers=3,
                num_heads=4,
                max_persons=max_persons
            )
        else:
            self.rcn = BasicRCN(
                input_dim=config.motion.dim,
                hidden_dim=64,
                num_layers=2,
                max_persons=max_persons
            )
    
    def forward(self, motion_input, traj, num_persons):
        """
        Args:
            motion_input: [B, P_max, T, D] - 输入（P_max为最大人数，多余的用padding填充）
            traj: [B, P_max, T, J, K] - 轨迹
            num_persons: [B] or int - 每个样本的实际人数
        Returns:
            output: [B, P_max, T, D] - 输出（padding位置输出为0）
        """
        B, P_max, T, D = motion_input.shape
        
        # 创建掩码
        if isinstance(num_persons, int):
            num_persons = [num_persons] * B
        mask = create_person_mask(num_persons, P_max, motion_input.device)
        
        # 计算距离
        from .model import compute_distances_hierarchical_normalization
        distances = compute_distances_hierarchical_normalization(traj, zero_score=False)
        
        # RCN处理
        if self.rcn_type == 'optimized':
            output, _ = self.rcn(motion_input, distances, mask=mask)
        else:
            output = self.rcn(motion_input, distances, mask=mask)
        
        # 应用掩码到输出
        output = output * mask.unsqueeze(-1).unsqueeze(-1)
        
        return output


def demo_usage():
    """
    演示如何使用RCN模块
    """
    import torch
    from easydict import EasyDict as edict
    
    # 创建示例配置
    config = edict()
    config.motion = edict()
    config.motion.dim = 39
    config.motion.h36m_input_length_dct = 16
    config.n_p = 3
    
    print("=" * 60)
    print("RCN Integration Demo")
    print("=" * 60)
    
    # 示例1：第一阶段 - 基础RCN
    print("\n第一阶段：基础RCN架构")
    print("-" * 60)
    model_phase1 = integrate_rcn_phase1(config)
    print(f"模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model_phase1.parameters()) / 1e6:.2f}M")
    
    # 示例2：第二阶段 - 带注意力的RCN
    print("\n第二阶段：添加注意力机制")
    print("-" * 60)
    model_phase2 = integrate_rcn_phase2(config)
    print(f"模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model_phase2.parameters()) / 1e6:.2f}M")
    
    # 示例3：第三阶段 - 优化的RCN
    print("\n第三阶段：优化的RCN（轻量化）")
    print("-" * 60)
    model_phase3_light = integrate_rcn_phase3(config, lightweight=True)
    print(f"模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model_phase3_light.parameters()) / 1e6:.2f}M")
    
    print("\n第三阶段：优化的RCN（标准）")
    print("-" * 60)
    model_phase3_std = integrate_rcn_phase3(config, lightweight=False)
    print(f"模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model_phase3_std.parameters()) / 1e6:.2f}M")
    
    # 示例4：可变人数模型
    print("\n可变人数RCN模型")
    print("-" * 60)
    variable_model = VariablePersonRCN(config, min_persons=2, max_persons=5)
    print(f"模型创建成功，支持2-5人场景")
    print(f"参数量: {sum(p.numel() for p in variable_model.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    print("\n测试前向传播（可变人数）")
    print("-" * 60)
    B, P, T, D = 4, 5, 16, 39
    J, K = 13, 3
    motion_input = torch.randn(B, P, T, D)
    traj = torch.randn(B, P, T, J, K)
    num_persons = [2, 3, 4, 5]  # 每个batch不同的人数
    
    output = variable_model(motion_input, traj, num_persons)
    print(f"输入形状: {motion_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"各batch人数: {num_persons}")
    
    # 验证掩码效果
    for i, n in enumerate(num_persons):
        valid_output = output[i, :n].abs().sum()
        padding_output = output[i, n:].abs().sum()
        print(f"Batch {i}: {n}人 - 有效输出={valid_output:.4f}, 填充输出={padding_output:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo完成!")
    print("=" * 60)


if __name__ == '__main__':
    demo_usage()
