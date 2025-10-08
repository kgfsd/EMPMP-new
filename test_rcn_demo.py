"""
RCN模块演示和测试脚本
展示三个阶段的RCN实现
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_phase1_basic_rcn():
    """测试第一阶段：基础RCN架构"""
    print("\n" + "=" * 70)
    print("第一阶段：基础RCN架构测试")
    print("=" * 70)
    
    from src.models_dual_inter_traj_3dpw.rcn import BasicRCN, create_person_mask
    
    # 配置
    B, P, T, D = 4, 3, 16, 39
    
    # 创建模型
    rcn = BasicRCN(
        input_dim=D,
        hidden_dim=64,
        num_layers=2,
        max_persons=P
    )
    
    # 计算参数量
    num_params = sum(p.numel() for p in rcn.parameters()) / 1e6
    print(f"✓ BasicRCN创建成功")
    print(f"  - 参数量: {num_params:.2f}M")
    print(f"  - 层数: 2")
    print(f"  - 隐藏维度: 64")
    
    # 创建测试数据
    features = torch.randn(B, P, T, D)
    distances = torch.randn(B, T, P, P)
    mask = torch.ones(B, P)
    
    # 前向传播
    output = rcn(features, distances, mask)
    
    print(f"\n✓ 前向传播测试")
    print(f"  - 输入形状: {features.shape}")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试可变人数
    print(f"\n✓ 可变人数测试")
    num_persons = [2, 2, 3, 3]
    mask = create_person_mask(num_persons, P, features.device)
    output = rcn(features, distances, mask)
    
    for i, n in enumerate(num_persons):
        valid_out = output[i, :n].abs().sum().item()
        padding_out = output[i, n:].abs().sum().item()
        print(f"  - Batch {i}: {n}人 -> 有效输出={valid_out:.2f}, 填充输出={padding_out:.6f}")
    
    print("\n✓ 第一阶段测试通过!")
    return True


def test_phase2_attention_rcn():
    """测试第二阶段：带注意力机制的RCN"""
    print("\n" + "=" * 70)
    print("第二阶段：注意力机制测试")
    print("=" * 70)
    
    from src.models_dual_inter_traj_3dpw.rcn_attention import (
        RCNWithAttention, 
        MultiHeadRelationAttention,
        EfficientAttention
    )
    
    # 配置
    B, P, T, D = 4, 3, 16, 39
    
    # 测试多头注意力
    print("\n1. 多头关系注意力")
    attention = MultiHeadRelationAttention(dim=64, num_heads=8)
    features = torch.randn(B, P, 64)
    relations = torch.randn(B, P, P, 64)
    output, attn_weights = attention(features, relations)
    
    print(f"  ✓ 输入: {features.shape}, 关系: {relations.shape}")
    print(f"  ✓ 输出: {output.shape}, 注意力权重: {attn_weights.shape}")
    
    # 测试高效注意力
    print("\n2. 高效注意力（线性复杂度）")
    efficient_attn = EfficientAttention(dim=64, num_heads=4)
    output = efficient_attn(features)
    print(f"  ✓ 线性注意力输出: {output.shape}")
    
    # 测试完整RCN with Attention
    print("\n3. 完整RCN with Attention")
    rcn_attn = RCNWithAttention(
        input_dim=D,
        hidden_dim=64,
        num_layers=3,
        num_heads=4
    )
    
    num_params = sum(p.numel() for p in rcn_attn.parameters()) / 1e6
    print(f"  ✓ 模型参数量: {num_params:.2f}M")
    
    features = torch.randn(B, P, T, D)
    distances = torch.randn(B, T, P, P)
    
    # 不返回注意力权重
    output = rcn_attn(features, distances)
    print(f"  ✓ 输出形状: {output.shape}")
    
    # 返回注意力权重
    output, attn_weights = rcn_attn(features, distances, return_attention=True)
    print(f"  ✓ 注意力权重数量: {len(attn_weights)} (时间步)")
    print(f"  ✓ 每个时间步的层数: {len(attn_weights[0])}")
    
    print("\n✓ 第二阶段测试通过!")
    return True


def test_phase3_optimized_rcn():
    """测试第三阶段：优化的RCN"""
    print("\n" + "=" * 70)
    print("第三阶段：优化RCN测试")
    print("=" * 70)
    
    from src.models_dual_inter_traj_3dpw.rcn_optimized import (
        OptimizedRCN,
        create_optimized_rcn,
        HierarchicalRelationType,
        EfficientFeatureExtractor
    )
    
    # 配置
    B, P, T, D = 4, 3, 16, 39
    
    # 测试层次化关系类型
    print("\n1. 层次化关系类型编码")
    relation_encoder = HierarchicalRelationType(
        feature_dim=64,
        relation_dim=32,
        use_learnable_weights=True
    )
    
    features_i = torch.randn(B, P, P, 64)
    features_j = torch.randn(B, P, P, 64)
    positions_i = torch.randn(B, P, P, 3)
    positions_j = torch.randn(B, P, P, 3)
    
    relations = relation_encoder(features_i, features_j, positions_i, positions_j)
    print(f"  ✓ 关系编码输出: {relations.shape}")
    print(f"  ✓ 关系类型权重: {relation_encoder.relation_weights.data}")
    
    # 测试高效特征提取
    print("\n2. 高效特征提取（深度可分离卷积）")
    extractor = EfficientFeatureExtractor(in_channels=D, out_channels=64)
    features = torch.randn(B, P, T, D)
    extracted = extractor(features)
    print(f"  ✓ 输入: {features.shape} -> 输出: {extracted.shape}")
    
    # 测试优化RCN（轻量化）
    print("\n3. 优化RCN（轻量化版本）")
    rcn_light, num_params = create_optimized_rcn(
        input_dim=D,
        hidden_dim=64,
        relation_dim=32,
        num_layers=3,
        lightweight=True
    )
    
    features = torch.randn(B, P, T, D)
    distances = torch.randn(B, T, P, P)
    output, info = rcn_light(features, distances)
    
    print(f"  ✓ 输出形状: {output.shape}")
    print(f"  ✓ 参数量: {num_params:.2f}M")
    print(f"  ✓ 关系权重数量: {len(info['relation_weights'])}")
    
    # 测试优化RCN（标准版本）
    print("\n4. 优化RCN（标准版本）")
    rcn_std, num_params = create_optimized_rcn(
        input_dim=D,
        hidden_dim=128,
        relation_dim=64,
        num_layers=3,
        lightweight=False
    )
    print(f"  ✓ 参数量: {num_params:.2f}M")
    
    output, info = rcn_std(features, distances)
    print(f"  ✓ 输出形状: {output.shape}")
    
    print("\n✓ 第三阶段测试通过!")
    return True


def test_integration():
    """测试集成模块"""
    print("\n" + "=" * 70)
    print("集成测试")
    print("=" * 70)
    
    from src.models_dual_inter_traj_3dpw.rcn_integration import VariablePersonRCN
    from easydict import EasyDict as edict
    
    # 创建配置
    config = edict()
    config.motion = edict()
    config.motion.dim = 39
    config.motion.h36m_input_length_dct = 16
    config.n_p = 5
    
    print("\n可变人数RCN模型")
    model = VariablePersonRCN(
        config,
        min_persons=2,
        max_persons=5,
        rcn_type='optimized'
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ 模型创建成功")
    print(f"  ✓ 支持人数范围: 2-5人")
    print(f"  ✓ 参数量: {num_params:.2f}M")
    
    # 测试不同人数的batch
    B, P_max, T, D = 4, 5, 16, 39
    J, K = 13, 3
    
    motion_input = torch.randn(B, P_max, T, D)
    traj = torch.randn(B, P_max, T, J, K)
    num_persons = [2, 3, 4, 5]
    
    print(f"\n  测试数据:")
    print(f"    - Batch size: {B}")
    print(f"    - 最大人数: {P_max}")
    print(f"    - 各batch人数: {num_persons}")
    
    output = model(motion_input, traj, num_persons)
    print(f"\n  ✓ 前向传播成功")
    print(f"    - 输出形状: {output.shape}")
    
    # 验证掩码效果
    print(f"\n  验证掩码效果:")
    for i, n in enumerate(num_persons):
        valid_out = output[i, :n].abs().sum().item()
        padding_out = output[i, n:].abs().sum().item()
        print(f"    - Batch {i} ({n}人): 有效={valid_out:.2f}, 填充={padding_out:.6f}")
    
    print("\n✓ 集成测试通过!")
    return True


def compare_models():
    """对比不同模型的参数量和性能"""
    print("\n" + "=" * 70)
    print("模型对比")
    print("=" * 70)
    
    from src.models_dual_inter_traj_3dpw.rcn import BasicRCN
    from src.models_dual_inter_traj_3dpw.rcn_attention import RCNWithAttention
    from src.models_dual_inter_traj_3dpw.rcn_optimized import OptimizedRCN
    
    D = 39
    models = {
        'BasicRCN (2层)': BasicRCN(D, 64, 2),
        'RCNWithAttention (3层, 4头)': RCNWithAttention(D, 64, 3, 4),
        'OptimizedRCN (轻量, 3层)': OptimizedRCN(D, 64, 32, 3, lightweight=True),
        'OptimizedRCN (标准, 3层)': OptimizedRCN(D, 128, 64, 3, lightweight=False),
    }
    
    print(f"\n{'模型':<35} {'参数量(M)':<15} {'状态'}")
    print("-" * 70)
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        status = "✓ 轻量" if num_params < 2.0 else "○ 标准"
        print(f"{name:<35} {num_params:<15.2f} {status}")
    
    print("\n推荐配置:")
    print("  - 快速部署 → BasicRCN")
    print("  - 平衡性能 → OptimizedRCN (轻量)")
    print("  - 最佳性能 → OptimizedRCN (标准)")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("RCN实现完整测试")
    print("支持多人及可变人数场景的轻量化关系网络")
    print("=" * 70)
    
    try:
        # 运行所有测试
        success = True
        success &= test_phase1_basic_rcn()
        success &= test_phase2_attention_rcn()
        success &= test_phase3_optimized_rcn()
        success &= test_integration()
        
        # 模型对比
        compare_models()
        
        # 总结
        print("\n" + "=" * 70)
        if success:
            print("✓✓✓ 所有测试通过! ✓✓✓")
        else:
            print("✗✗✗ 部分测试失败 ✗✗✗")
        print("=" * 70)
        
        print("\n三阶段实现完成:")
        print("  ✓ 第一阶段: 基础RCN架构 (rcn.py)")
        print("  ✓ 第二阶段: 注意力机制 (rcn_attention.py)")
        print("  ✓ 第三阶段: 优化实现 (rcn_optimized.py)")
        print("\n集成模块: rcn_integration.py")
        print("使用文档: docs/RCN_Implementation_Guide.md")
        print()
        
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success


if __name__ == '__main__':
    main()
