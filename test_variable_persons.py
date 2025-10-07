"""
Test script for variable person count support (9-15 people)
Demonstrates how to use the updated model architecture with dynamic person counts
"""
import torch
import numpy as np
from easydict import EasyDict as edict

# Test the model with different person counts
def test_variable_persons():
    print("Testing variable person count support (9-15 people)...")
    
    # Create a mock config
    config = edict()
    config.motion_mlp = edict()
    config.motion_mlp.hidden_dim = 45
    config.motion_mlp.seq_len = 30
    config.motion_mlp.with_normalization = True
    config.motion_mlp.spatial_fc_only = False
    config.motion_mlp.num_layers = 48
    config.motion_mlp.norm_axis = 'spatial'
    config.motion_mlp.interaction_interval = 4
    config.motion_mlp.n_p = 3  # Default min persons
    config.motion_mlp.max_p = 15  # Support up to 15 persons
    
    config.motion = edict()
    config.motion.dim = 45
    config.motion.h36m_input_length_dct = 30
    
    config.motion_fc_in = edict()
    config.motion_fc_in.temporal_fc = False
    
    config.motion_fc_out = edict()
    config.motion_fc_out.temporal_fc = False
    
    # Import the model
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    from src.models_dual_inter_traj_big.model import siMLPe
    
    # Create model
    model = siMLPe(config)
    model.eval()
    
    batch_size = 2
    seq_len = 30
    n_joints = 15
    n_coords = 3
    
    # Test with different person counts
    person_counts = [9, 12, 15]
    
    for n_persons in person_counts:
        print(f"\nTesting with {n_persons} persons...")
        
        # Create dummy input data: B, P, T, J, K
        motion_input = torch.randn(batch_size, n_persons, seq_len, config.motion.dim)
        traj = torch.randn(batch_size, n_persons, seq_len, n_joints, n_coords)
        
        # Create padding mask (all valid for this test)
        padding_mask = torch.ones(batch_size, n_persons)
        
        # Test forward pass
        with torch.no_grad():
            try:
                output = model(motion_input, traj, padding_mask=padding_mask)
                print(f"  Input shape: {motion_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  ✓ Successfully processed {n_persons} persons")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                raise
    
    # Test with variable persons in same batch (simulating enter/exit)
    print(f"\nTesting with variable persons in same batch...")
    n_persons_max = 15
    actual_persons = [10, 15]  # Different person counts in batch
    
    motion_input_list = []
    traj_list = []
    padding_mask_list = []
    
    for n in actual_persons:
        # Pad to max_p
        motion = torch.randn(1, n, seq_len, config.motion.dim)
        if n < n_persons_max:
            motion_pad = torch.zeros(1, n_persons_max - n, seq_len, config.motion.dim)
            motion = torch.cat([motion, motion_pad], dim=1)
        motion_input_list.append(motion)
        
        traj_item = torch.randn(1, n, seq_len, n_joints, n_coords)
        if n < n_persons_max:
            traj_pad = torch.zeros(1, n_persons_max - n, seq_len, n_joints, n_coords)
            traj_item = torch.cat([traj_item, traj_pad], dim=1)
        traj_list.append(traj_item)
        
        mask = torch.ones(1, n)
        if n < n_persons_max:
            mask_pad = torch.zeros(1, n_persons_max - n)
            mask = torch.cat([mask, mask_pad], dim=1)
        padding_mask_list.append(mask)
    
    motion_input_batch = torch.cat(motion_input_list, dim=0)
    traj_batch = torch.cat(traj_list, dim=0)
    padding_mask_batch = torch.cat(padding_mask_list, dim=0)
    
    with torch.no_grad():
        try:
            output = model(motion_input_batch, traj_batch, padding_mask=padding_mask_batch)
            print(f"  Input shape: {motion_input_batch.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Padding mask shape: {padding_mask_batch.shape}")
            print(f"  ✓ Successfully processed variable persons in batch")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            raise
    
    print("\n✓ All tests passed! Model supports variable person counts (9-15).")
    print("\nKey features:")
    print("- Supports 9-15 persons dynamically")
    print("- Uses padding masks for variable person counts")
    print("- Maintains lightweight architecture with parameter reuse")
    print("- Handles persons entering/exiting in the same dataset")

if __name__ == "__main__":
    test_variable_persons()
