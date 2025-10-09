"""
Example usage of Dynamic GCN Model for variable person count.

This file demonstrates how to use the new DynamicGCNModel to handle
varying numbers of persons in multi-person motion prediction, removing
the fixed n_p limitation.
"""

import torch
import torch.nn as nn
from collections import namedtuple


def create_sample_config():
    """
    Create a sample configuration for the Dynamic GCN Model.
    
    Returns:
        Configuration object compatible with DynamicGCNModel
    """
    # Create nested config structure using namedtuples
    MotionConfig = namedtuple('MotionConfig', ['dim', 'h36m_input_length_dct'])
    MotionFCConfig = namedtuple('MotionFCConfig', ['temporal_fc'])
    MLPConfig = namedtuple('MLPConfig', [
        'hidden_dim', 'seq_len', 'with_normalization', 'spatial_fc_only',
        'num_layers', 'norm_axis', 'interaction_interval', 'n_p', 'use_distance'
    ])
    
    Config = namedtuple('Config', [
        'motion', 'motion_fc_in', 'motion_fc_out', 'motion_mlp',
        'hidden_dim', 'num_gcn_layers', 'use_distance_based_adj', 'adj_temperature'
    ])
    
    motion_config = MotionConfig(dim=39, h36m_input_length_dct=15)  # 13 joints * 3 coords
    motion_fc_in = MotionFCConfig(temporal_fc=False)
    motion_fc_out = MotionFCConfig(temporal_fc=False)
    
    mlp_config = MLPConfig(
        hidden_dim=256,
        seq_len=15,
        with_normalization=True,
        spatial_fc_only=False,
        num_layers=8,
        norm_axis='spatial',
        interaction_interval=2,
        n_p=3,  # This can be ignored in dynamic mode
        use_distance=True
    )
    
    config = Config(
        motion=motion_config,
        motion_fc_in=motion_fc_in,
        motion_fc_out=motion_fc_out,
        motion_mlp=mlp_config,
        hidden_dim=256,
        num_gcn_layers=2,
        use_distance_based_adj=True,
        adj_temperature=1.0
    )
    
    return config


def example_basic_usage():
    """
    Example 1: Basic usage with variable number of persons.
    """
    print("=" * 70)
    print("Example 1: Basic Usage with Variable Person Count")
    print("=" * 70)
    
    # Import the model
    try:
        from .dynamic_gcn_model import DynamicGCNModel
    except ImportError:
        # If running as script
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from dynamic_gcn_model import DynamicGCNModel
    
    # Create configuration
    config = create_sample_config()
    
    # Initialize model
    model = DynamicGCNModel(config)
    model.eval()
    
    # Create sample data with VARIABLE number of persons per batch
    batch_size = 4
    max_persons = 5  # Maximum persons in any sample
    seq_len = 15
    input_dim = 39  # 13 joints * 3 coordinates
    
    # Number of persons in each sample (variable!)
    num_persons = [2, 3, 5, 4]  # Different for each batch sample
    
    print(f"\nBatch size: {batch_size}")
    print(f"Max persons: {max_persons}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of persons per sample: {num_persons}")
    
    # Create input data (padded to max_persons)
    motion_input = torch.randn(batch_size, max_persons, seq_len, input_dim)
    
    # Mask out padded positions
    for b in range(batch_size):
        motion_input[b, num_persons[b]:, :, :] = 0
    
    print(f"\nInput shape: {motion_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(motion_input, num_persons=num_persons)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Successfully processed batch with variable person counts!")
    print()


def example_with_distances():
    """
    Example 2: Using distance-based adjacency matrix construction.
    """
    print("=" * 70)
    print("Example 2: Distance-Based Adjacency Matrix")
    print("=" * 70)
    
    try:
        from .dynamic_gcn_model import DynamicGCNModel
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from dynamic_gcn_model import DynamicGCNModel
    
    config = create_sample_config()
    model = DynamicGCNModel(config)
    model.eval()
    
    # Sample parameters
    batch_size = 3
    max_persons = 4
    seq_len = 15
    input_dim = 39
    
    num_persons = [2, 4, 3]
    
    # Create input and distance matrix
    motion_input = torch.randn(batch_size, max_persons, seq_len, input_dim)
    
    # Distance matrix: [batch_size, time, max_persons, max_persons]
    distances = torch.zeros(batch_size, seq_len, max_persons, max_persons)
    
    # Fill with sample distances (e.g., from trajectory data)
    for b in range(batch_size):
        for t in range(seq_len):
            for i in range(num_persons[b]):
                for j in range(num_persons[b]):
                    # Simulate distance between person i and j at time t
                    distances[b, t, i, j] = torch.randn(1).abs() * 2.0
    
    print(f"\nBatch size: {batch_size}")
    print(f"Max persons: {max_persons}")
    print(f"Number of persons per sample: {num_persons}")
    print(f"Input shape: {motion_input.shape}")
    print(f"Distance matrix shape: {distances.shape}")
    
    # Forward pass with distances
    with torch.no_grad():
        output = model(motion_input, num_persons=num_persons, distances=distances)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Successfully used distance-based graph construction!")
    print()


def example_batch_processing():
    """
    Example 3: Processing batches with different person counts.
    """
    print("=" * 70)
    print("Example 3: Batch Processing with Dynamic Padding")
    print("=" * 70)
    
    try:
        from .dynamic_gcn_model import DynamicGCNModel
        from .gcn import pad_features_to_max_persons, create_person_mask
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from dynamic_gcn_model import DynamicGCNModel
        from gcn import pad_features_to_max_persons, create_person_mask
    
    config = create_sample_config()
    model = DynamicGCNModel(config)
    model.eval()
    
    # Simulate data with different person counts
    seq_len = 15
    input_dim = 39
    
    # List of samples with different person counts
    sample_1 = torch.randn(2, input_dim, seq_len)  # 2 persons
    sample_2 = torch.randn(4, input_dim, seq_len)  # 4 persons
    sample_3 = torch.randn(1, input_dim, seq_len)  # 1 person
    sample_4 = torch.randn(3, input_dim, seq_len)  # 3 persons
    
    samples = [sample_1, sample_2, sample_3, sample_4]
    num_persons = [s.size(0) for s in samples]
    max_persons = max(num_persons)
    
    print(f"\nProcessing {len(samples)} samples:")
    for i, n in enumerate(num_persons):
        print(f"  Sample {i+1}: {n} person(s)")
    print(f"Max persons: {max_persons}")
    
    # Pad samples to max_persons
    padded_samples, mask = pad_features_to_max_persons(samples, max_persons)
    
    print(f"\nPadded batch shape: {padded_samples.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Rearrange to [batch, persons, seq_len, dim] format
    padded_samples = padded_samples.permute(0, 1, 3, 2)
    
    # Forward pass
    with torch.no_grad():
        output = model(padded_samples, num_persons=num_persons)
    
    print(f"Output shape: {output.shape}")
    
    # Extract valid outputs using mask
    print("\nExtracting valid predictions:")
    for b in range(len(samples)):
        valid_output = output[b, :num_persons[b], :, :]
        print(f"  Sample {b+1}: {valid_output.shape} (only {num_persons[b]} person(s))")
    
    print(f"\n✓ Successfully processed variable-sized batch with padding!")
    print()


def example_training_loop():
    """
    Example 4: Training loop with dynamic person count.
    """
    print("=" * 70)
    print("Example 4: Training Loop Simulation")
    print("=" * 70)
    
    try:
        from .dynamic_gcn_model import DynamicGCNModel
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from dynamic_gcn_model import DynamicGCNModel
    
    config = create_sample_config()
    model = DynamicGCNModel(config)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("\nSimulating training with variable person counts...")
    print("(3 mini-batches with different configurations)\n")
    
    # Simulate 3 training iterations
    for iteration in range(3):
        # Generate random batch configuration
        batch_size = torch.randint(2, 5, (1,)).item()
        max_persons = torch.randint(2, 6, (1,)).item()
        num_persons = [torch.randint(1, max_persons + 1, (1,)).item() 
                      for _ in range(batch_size)]
        
        seq_len = 15
        input_dim = 39
        
        # Generate data
        motion_input = torch.randn(batch_size, max_persons, seq_len, input_dim)
        motion_target = torch.randn(batch_size, max_persons, seq_len, input_dim)
        
        # Mask padded positions
        for b in range(batch_size):
            motion_input[b, num_persons[b]:, :, :] = 0
            motion_target[b, num_persons[b]:, :, :] = 0
        
        # Forward pass
        optimizer.zero_grad()
        output = model(motion_input, num_persons=num_persons)
        
        # Compute loss only on valid persons
        loss = 0
        for b in range(batch_size):
            valid_output = output[b, :num_persons[b], :, :]
            valid_target = motion_target[b, :num_persons[b], :, :]
            loss += criterion(valid_output, valid_target)
        loss = loss / batch_size
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {iteration + 1}:")
        print(f"  Batch size: {batch_size}, Max persons: {max_persons}")
        print(f"  Persons per sample: {num_persons}")
        print(f"  Loss: {loss.item():.6f}")
    
    print(f"\n✓ Successfully trained with dynamic person counts!")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Dynamic GCN Model Examples")
    print("Variable Person Count Support for Multi-Person Motion Prediction")
    print("=" * 70 + "\n")
    
    # Run examples
    example_basic_usage()
    example_with_distances()
    example_batch_processing()
    example_training_loop()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
