#!/usr/bin/env python
"""
Example script demonstrating the use of GCN-based global flow module.

This script shows:
1. How to enable GCN in configuration
2. How to use the GCN module directly
3. How to integrate with existing model
4. Comparison between GCN and original approach
"""

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _project_root)

import torch
from easydict import EasyDict as edict


def example1_basic_gcn_usage():
    """Example 1: Basic usage of GCN module"""
    print("\n" + "=" * 60)
    print("Example 1: Basic GCN Module Usage")
    print("=" * 60)
    
    from src.models_dual_inter_traj_big.gcn import DynamicGCN
    
    # Setup
    B, P, D, T = 2, 3, 64, 15
    x = torch.randn(B, P, D, T)
    distances = torch.rand(B, T, P, P)
    
    # Make distances symmetric
    distances = (distances + distances.transpose(-1, -2)) / 2
    for b in range(B):
        for t in range(T):
            distances[b, t].fill_diagonal_(0)
    
    # Create GCN module with k-NN
    gcn = DynamicGCN(dim=D, num_layers=2, k_neighbors=2)
    
    # Forward pass
    output = gcn(x, distances)
    
    print(f"Input shape: {x.shape}")
    print(f"Distance shape: {distances.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Basic GCN usage successful!")


def example2_config_based_usage():
    """Example 2: Using GCN through configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Configuration-Based Usage")
    print("=" * 60)
    
    from src.models_dual_inter_traj_big.mlp import build_mlps
    
    # Create configuration with GCN enabled
    config = edict()
    config.hidden_dim = 45
    config.seq_len = 15
    config.with_normalization = True
    config.spatial_fc_only = False
    config.num_layers = 64
    config.norm_axis = 'spatial'
    config.interaction_interval = 16
    config.n_p = 3
    
    # Enable GCN
    config.use_gcn = True
    config.k_neighbors = 2
    config.gcn_layers = 2
    
    # Build model
    mlp = build_mlps(config)
    print(f"Created model: {type(mlp).__name__}")
    
    # Test forward pass
    B, P, D, T = 2, 3, 45, 15
    x = torch.randn(B, P, D, T)
    distances = torch.rand(B, T, P, P)
    distances = (distances + distances.transpose(-1, -2)) / 2
    
    output = mlp(x, distances)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Configuration-based usage successful!")


def example3_compare_with_original():
    """Example 3: Compare GCN with original TransMLP"""
    print("\n" + "=" * 60)
    print("Example 3: Comparison with Original TransMLP")
    print("=" * 60)
    
    from src.models_dual_inter_traj_big.mlp import build_mlps
    
    # Shared config
    base_config = edict()
    base_config.hidden_dim = 45
    base_config.seq_len = 15
    base_config.with_normalization = True
    base_config.spatial_fc_only = False
    base_config.num_layers = 64
    base_config.norm_axis = 'spatial'
    base_config.interaction_interval = 16
    base_config.n_p = 3
    
    # Original model
    config_original = base_config.copy()
    config_original.use_gcn = False
    mlp_original = build_mlps(config_original)
    print(f"Original model: {type(mlp_original).__name__}")
    
    # GCN-based model
    config_gcn = base_config.copy()
    config_gcn.use_gcn = True
    config_gcn.k_neighbors = 2
    config_gcn.gcn_layers = 2
    mlp_gcn = build_mlps(config_gcn)
    print(f"GCN model: {type(mlp_gcn).__name__}")
    
    # Test with same input
    B, P, D, T = 2, 3, 45, 15
    x = torch.randn(B, P, D, T)
    distances = torch.rand(B, T, P, P)
    distances = (distances + distances.transpose(-1, -2)) / 2
    
    with torch.no_grad():
        output_original = mlp_original(x, distances)
        output_gcn = mlp_gcn(x, distances)
    
    print(f"Original output shape: {output_original.shape}")
    print(f"GCN output shape: {output_gcn.shape}")
    print("✓ Both models work with same inputs!")


def example4_variable_sized_inputs():
    """Example 4: Variable numbers of people"""
    print("\n" + "=" * 60)
    print("Example 4: Variable-Sized Inputs (Different P)")
    print("=" * 60)
    
    from src.models_dual_inter_traj_big.gcn import DynamicGCN
    
    D, T = 64, 15
    
    # Test different numbers of people
    for P in [1, 3, 5, 10]:
        B = 2
        x = torch.randn(B, P, D, T)
        distances = torch.rand(B, T, P, P)
        distances = (distances + distances.transpose(-1, -2)) / 2
        for b in range(B):
            for t in range(T):
                distances[b, t].fill_diagonal_(0)
        
        # Create GCN with appropriate k_neighbors
        k = min(2, max(0, P - 1))
        gcn = DynamicGCN(dim=D, num_layers=2, k_neighbors=k)
        
        output = gcn(x, distances)
        print(f"P={P:2d}: Input {x.shape} -> Output {output.shape} ✓")
    
    print("✓ Variable-sized inputs handled correctly!")


def example5_graph_construction_strategies():
    """Example 5: Different graph construction strategies"""
    print("\n" + "=" * 60)
    print("Example 5: Graph Construction Strategies")
    print("=" * 60)
    
    from src.models_dual_inter_traj_big.gcn import DynamicGraphConstruction
    
    B, T, P = 2, 10, 5
    distances = torch.rand(B, T, P, P) * 10
    distances = (distances + distances.transpose(-1, -2)) / 2
    for b in range(B):
        for t in range(T):
            distances[b, t].fill_diagonal_(0)
    
    print("Distance matrix sample (first batch, first frame):")
    print(distances[0, 0])
    
    # 1. K-Nearest Neighbors
    print("\n1. K-Nearest Neighbors (k=2):")
    graph_knn = DynamicGraphConstruction(k_neighbors=2, self_loop=True)
    adj_knn = graph_knn(distances)
    print(f"Adjacency shape: {adj_knn.shape}")
    print("Sample adjacency matrix:")
    print(adj_knn[0, 0])
    
    # 2. Distance Threshold
    print("\n2. Distance Threshold (threshold=5.0):")
    graph_thresh = DynamicGraphConstruction(distance_threshold=5.0, self_loop=True)
    adj_thresh = graph_thresh(distances)
    print(f"Adjacency shape: {adj_thresh.shape}")
    print("Sample adjacency matrix:")
    print(adj_thresh[0, 0])
    
    # 3. Gaussian Kernel (fully connected)
    print("\n3. Gaussian Kernel (fully connected):")
    graph_gaussian = DynamicGraphConstruction(
        k_neighbors=None, 
        distance_threshold=None,
        self_loop=True
    )
    adj_gaussian = graph_gaussian(distances)
    print(f"Adjacency shape: {adj_gaussian.shape}")
    print("Sample adjacency matrix:")
    print(adj_gaussian[0, 0])
    
    print("\n✓ All graph construction strategies work!")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GCN-Based Global Flow Module - Usage Examples")
    print("=" * 60)
    
    try:
        example1_basic_gcn_usage()
        example2_config_based_usage()
        example3_compare_with_original()
        example4_variable_sized_inputs()
        example5_graph_construction_strategies()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully! ✓")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - docs/GCN_MODULE_USAGE.md")
        print("  - docs/GCN_IMPLEMENTATION_DETAILS.md")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
