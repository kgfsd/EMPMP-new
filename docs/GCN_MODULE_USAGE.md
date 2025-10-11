# GCN-Based Global Flow Module Documentation

## Overview

This document describes the Graph Convolutional Network (GCN) based global flow module implementation for handling variable numbers of people in multi-person motion prediction tasks.

## Features

- **Dynamic Graph Construction**: Automatically constructs adjacency matrices based on Euclidean distances between people
- **Variable-Sized Input Support**: Handles different numbers of people across batches and frames
- **Multiple Graph Construction Strategies**:
  - K-nearest neighbors (k-NN)
  - Distance thresholding
  - Gaussian kernel-based weights
- **Backward Compatibility**: Original TransMLP implementation remains available
- **Edge Case Handling**: Properly handles single person and many people scenarios

## Architecture Components

### 1. DynamicGraphConstruction

Constructs adjacency matrices dynamically based on spatial distances between people.

**Parameters:**
- `k_neighbors` (int, optional): Number of nearest neighbors to connect. If set, creates k-NN graph.
- `distance_threshold` (float, optional): Distance threshold for edge creation.
- `self_loop` (bool): Whether to add self-loops to the graph. Default: True.

**Input:**
- `distances`: Tensor of shape `[B, T, P, P]` - pairwise Euclidean distances between people

**Output:**
- `adjacency`: Tensor of shape `[B, T, P, P]` - normalized adjacency matrix

### 2. GCNLayer

Basic graph convolutional layer that operates on node features with a given adjacency matrix.

**Parameters:**
- `in_features` (int): Input feature dimension
- `out_features` (int): Output feature dimension
- `bias` (bool): Whether to use bias. Default: True
- `activation` (str): Activation function ('relu', 'tanh', 'elu', or None)

### 3. DynamicGCN

Complete dynamic GCN module that combines graph construction and GCN layers.

**Parameters:**
- `dim` (int): Feature dimension
- `num_layers` (int): Number of GCN layers. Default: 2
- `k_neighbors` (int, optional): Number of neighbors for k-NN graph
- `distance_threshold` (float, optional): Distance threshold for edge creation

## Usage

### Basic Usage with Configuration

To enable the GCN-based approach in your model configuration:

```python
from easydict import EasyDict as edict

# Create config
config = edict()
config.hidden_dim = 45
config.seq_len = 15
config.with_normalization = True
config.spatial_fc_only = False
config.num_layers = 64
config.norm_axis = 'spatial'
config.interaction_interval = 16
config.n_p = 3

# Enable GCN-based approach
config.use_gcn = True           # Set to True to use GCN
config.k_neighbors = 2          # Optional: number of neighbors
config.gcn_layers = 2           # Optional: number of GCN layers

# Build the model
from src.models_dual_inter_traj_big.mlp import build_mlps
mlp = build_mlps(config)
```

### Using Original TransMLP (Backward Compatibility)

To use the original TransMLP without GCN:

```python
config.use_gcn = False  # or simply omit use_gcn
mlp = build_mlps(config)
```

### Forward Pass

```python
import torch

# Input features
B, P, D, T = 2, 3, 45, 15
x = torch.randn(B, P, D, T)  # [Batch, People, Dimension, Time]

# Pairwise distances between people
distances = torch.rand(B, T, P, P)  # [Batch, Time, People, People]

# Forward pass
output = mlp(x, distances)  # Output shape: [B, P, D, T]
```

## Graph Construction Strategies

### 1. K-Nearest Neighbors (Recommended)

Connects each person to their k nearest neighbors based on Euclidean distance.

```python
from src.models_dual_inter_traj_big.gcn import DynamicGraphConstruction

graph_constructor = DynamicGraphConstruction(k_neighbors=2, self_loop=True)
```

**Advantages:**
- Fixed number of edges per node
- Efficient for varying numbers of people
- Good for sparse graphs

### 2. Distance Threshold

Creates edges between people if their distance is below a threshold.

```python
graph_constructor = DynamicGraphConstruction(
    distance_threshold=5.0, 
    self_loop=True
)
```

**Advantages:**
- Intuitive physical interpretation
- Adaptable to scene density

### 3. Gaussian Kernel (Default)

Uses all connections with Gaussian-weighted edges based on distance.

```python
graph_constructor = DynamicGraphConstruction(
    k_neighbors=None,
    distance_threshold=None,
    self_loop=True
)
```

**Advantages:**
- Smooth weight decay with distance
- No hard cutoffs
- Works well for dense graphs

## Edge Cases

### Single Person

The implementation handles single person scenarios gracefully:

```python
B, P, D, T = 2, 1, 64, 15
x_single = torch.randn(B, P, D, T)
distances_single = torch.zeros(B, T, P, P)

from src.models_dual_inter_traj_big.gcn import DynamicGCN
gcn = DynamicGCN(dim=D, num_layers=2, k_neighbors=0)
output = gcn(x_single, distances_single)  # Works correctly
```

### Many People

The implementation scales to handle many people efficiently:

```python
B, P, D, T = 2, 10, 64, 15
x_many = torch.randn(B, P, D, T)
distances_many = compute_distances(trajectories)  # Your distance function

gcn = DynamicGCN(dim=D, num_layers=2, k_neighbors=3)
output = gcn(x_many, distances_many)  # Scales well
```

## Configuration Parameters

When using GCN with your model, add these parameters to your config:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gcn` | bool | False | Enable GCN-based approach |
| `k_neighbors` | int | None | Number of nearest neighbors for k-NN graph |
| `gcn_layers` | int | 2 | Number of GCN layers in each block |

## Integration Examples

### Example 1: Training with GCN

```python
from src.models_dual_inter_traj_big.model import siMLPe
from src.baseline_h36m_15to45.config import config

# Enable GCN in config
config.motion_mlp.use_gcn = True
config.motion_mlp.k_neighbors = 2
config.motion_mlp.gcn_layers = 2

# Create model
model = siMLPe(config)

# Training loop
for batch in dataloader:
    motion_input, traj = batch
    output = model(motion_input, traj)
    # ... compute loss and backprop
```

### Example 2: Direct GCN Usage

```python
from src.models_dual_inter_traj_big.gcn import DynamicGCN
import torch

# Create GCN module
gcn = DynamicGCN(dim=64, num_layers=2, k_neighbors=2)

# Apply GCN
B, P, D, T = 2, 3, 64, 15
x = torch.randn(B, P, D, T)
distances = torch.rand(B, T, P, P)
output = gcn(x, distances)
```

## Performance Considerations

### Efficiency Tips

1. **Use k-NN for large numbers of people**: K-nearest neighbors approach is more efficient than fully connected graphs
2. **Choose appropriate k**: For most applications, k=2 or k=3 provides good results
3. **Number of GCN layers**: 2-3 layers are usually sufficient

### Memory Usage

- Memory usage scales as O(B × T × P²) for adjacency matrices
- Use smaller k values for k-NN to reduce memory footprint
- Consider processing in smaller batches for very large P

## Troubleshooting

### CUDA out of memory

**Solution**: Reduce batch size or use k-NN with smaller k value.

```python
config.k_neighbors = 2  # Reduce from 3
```

### Single person doesn't work

**Solution**: Set k_neighbors to 0 or None for single person scenarios.

```python
config.k_neighbors = 0  # For single person
```

### Backward compatibility issues

**Solution**: Ensure `use_gcn=False` or omit the parameter entirely.

```python
config.use_gcn = False  # Use original implementation
```

## Model Files Supported

The GCN module is available in:
- `src/models_dual_inter_traj_big/`
- `src/models_dual_inter_traj_pips/`

Both implementations are identical and can be used interchangeably.
