# Dynamic GCN for Variable Person Count Support

This module implements Graph Convolutional Networks (GCN) to remove the fixed number of persons (n_p) limitation in multi-person motion prediction.

## Overview

The current model architecture has limitations:
1. **Fixed person count (n_p)**: Input and output dimensions are fixed
2. **Limited person interaction modeling**: Cannot adapt to varying numbers of people
3. **Inefficient batch processing**: Cannot handle samples with different person counts in the same batch

The GCN-based solution addresses these issues by:
1. Using dynamic graph structures that adapt to any number of persons
2. Modeling person interactions through graph convolutions
3. Supporting variable-sized batches with efficient padding and masking

## Architecture

### Core Components

#### 1. GraphConvolution Layer (`gcn.py`)
Lightweight graph convolution that operates on dynamic graphs:
```python
from models_dual_inter_traj.gcn import GraphConvolution

gcn = GraphConvolution(in_features=256, out_features=256)
output = gcn(node_features, adjacency_matrices)
```

#### 2. DynamicGCNModel (`dynamic_gcn_model.py`)
Complete model with dynamic person count support:
```python
from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel

model = DynamicGCNModel(config)
output = model(motion_input, num_persons=[2, 3, 5, 4])
```

#### 3. Dynamic Adjacency Matrix Construction
Two strategies for building person interaction graphs:

**Feature-based (similarity)**:
```python
from models_dual_inter_traj.gcn import build_dynamic_adjacency_matrix

adj_matrices = build_dynamic_adjacency_matrix(features, num_persons)
```

**Distance-based (spatial proximity)**:
```python
from models_dual_inter_traj.gcn import build_distance_based_adjacency_matrix

adj_matrices = build_distance_based_adjacency_matrix(distances, num_persons)
```

## Usage Examples

### Basic Usage

```python
import torch
from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel

# Initialize model with config
model = DynamicGCNModel(config)

# Batch with variable person counts
batch_size = 4
max_persons = 5
seq_len = 15
input_dim = 39

# Different number of persons in each sample
num_persons = [2, 3, 5, 4]

# Create padded input (pad to max_persons)
motion_input = torch.randn(batch_size, max_persons, seq_len, input_dim)

# Zero out padded positions
for b in range(batch_size):
    motion_input[b, num_persons[b]:, :, :] = 0

# Forward pass
output = model(motion_input, num_persons=num_persons)
print(f"Output shape: {output.shape}")  # [4, 5, 15, 39]
```

### With Distance Information

```python
# Create distance matrix
distances = torch.zeros(batch_size, seq_len, max_persons, max_persons)

# Fill with actual distances between persons
for b in range(batch_size):
    for t in range(seq_len):
        for i in range(num_persons[b]):
            for j in range(num_persons[b]):
                distances[b, t, i, j] = compute_distance(person_i, person_j)

# Forward pass with distances
output = model(motion_input, num_persons=num_persons, distances=distances)
```

### Batch Processing with DataLoader

```python
from torch.utils.data import DataLoader
from models_dual_inter_traj.dynamic_data_utils import (
    DynamicPersonDataset, collate_variable_persons
)

# Create dataset
dataset = DynamicPersonDataset(data_list, num_persons_list)

# Create DataLoader with custom collate function
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_variable_persons,
    shuffle=True
)

# Training loop
for batch in dataloader:
    # batch is a DynamicPersonBatch object
    output = model(batch.motion_data, num_persons=batch.num_persons)
    
    # Compute loss only on valid persons
    loss = compute_loss_with_mask(output, targets, batch.mask, criterion)
    loss.backward()
```

### Training with Dynamic Person Counts

```python
from models_dual_inter_traj.dynamic_data_utils import compute_loss_with_mask

model = DynamicGCNModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for motion_input, motion_target, num_persons, mask in dataloader:
        # Forward pass
        output = model(motion_input, num_persons=num_persons)
        
        # Compute loss only on valid (non-padded) persons
        loss = compute_loss_with_mask(output, motion_target, mask, criterion)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Key Features

### 1. Dynamic Graph Construction
- **Adaptive to person count**: Graphs are built on-the-fly based on actual number of persons
- **Two construction modes**:
  - Similarity-based: Uses feature similarity to determine connections
  - Distance-based: Uses spatial proximity from trajectory data
- **Efficient**: Only processes valid persons, ignoring padding

### 2. Flexible Batch Processing
- **Variable person counts**: Each sample can have different number of persons
- **Automatic padding**: Utilities provided for padding to maximum batch size
- **Masked loss computation**: Only valid persons contribute to loss

### 3. Integration with Existing Architecture
- **Compatible with MLP**: Can be combined with existing MLP layers
- **HybridGCNMLP**: Combines GCN for person interaction + MLP for temporal modeling
- **Drop-in replacement**: Can replace fixed n_p models with minimal changes

## Configuration

Example configuration for DynamicGCNModel:

```python
config = {
    'motion': {
        'dim': 39,  # 13 joints × 3 coords
        'h36m_input_length_dct': 15
    },
    'motion_fc_in': {'temporal_fc': False},
    'motion_fc_out': {'temporal_fc': False},
    'hidden_dim': 256,
    'num_gcn_layers': 2,
    'use_distance_based_adj': True,
    'adj_temperature': 1.0,
    'motion_mlp': {
        'hidden_dim': 256,
        'seq_len': 15,
        'with_normalization': True,
        'spatial_fc_only': False,
        'num_layers': 8,
        'norm_axis': 'spatial',
        'interaction_interval': 2,
        'n_p': 3,  # Not used in dynamic mode
        'use_distance': True
    }
}
```

## API Reference

### GraphConvolution
```python
class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, x: Tensor, adj_matrices: List[Tensor]) -> Tensor
```

### DynamicGCNModel
```python
class DynamicGCNModel(nn.Module):
    def __init__(self, config)
    def forward(self, 
                motion_input: Tensor,      # [B, P, T, D]
                num_persons: List[int],    # [B]
                distances: Tensor = None   # [B, T, P, P]
               ) -> Tensor                 # [B, P, T, D]
```

### Utility Functions
```python
# Graph construction
build_dynamic_adjacency_matrix(features, num_persons, temperature=1.0)
build_distance_based_adjacency_matrix(distances, num_persons, k=None)

# Data processing
pad_features_to_max_persons(features_list, max_persons)
create_person_mask(num_persons_list, max_persons, device=None)
process_batch_dynamic(batch_data, num_persons_list)
unpad_batch_dynamic(padded_data, num_persons_list)

# Loss computation
compute_loss_with_mask(predictions, targets, mask, criterion)
```

## Benefits

### 1. Removes Fixed Person Count Limitation
- ✅ Handle 1 to N persons in the same model
- ✅ No need for separate models for different person counts
- ✅ More flexible deployment scenarios

### 2. Improved Person Interaction Modeling
- ✅ Explicit modeling of person-person relationships via graphs
- ✅ Adaptive connections based on similarity or proximity
- ✅ Natural representation of multi-person scenes

### 3. Computational Efficiency
- ✅ Only processes valid persons (no wasted computation on padding)
- ✅ Dynamic graph construction is efficient
- ✅ Compatible with batched processing

### 4. Easy Integration
- ✅ Compatible with existing model components
- ✅ Minimal changes to training pipeline
- ✅ Can be used alongside or replace existing modules

## Testing

Run the example file to see demonstrations:

```bash
cd src/models_dual_inter_traj
python dynamic_gcn_example.py
```

This will run:
1. Basic usage with variable person counts
2. Distance-based adjacency matrix construction
3. Batch processing with dynamic padding
4. Training loop simulation

## Performance Considerations

### Memory Usage
- Memory scales with `max_persons` in batch, not fixed n_p
- Use smaller batch sizes if samples have many persons
- Consider gradient accumulation for large person counts

### Computation
- GCN computation is O(P²) where P is number of persons
- Distance-based adjacency is more expensive but often more accurate
- For very large P, consider k-nearest neighbor graphs

### Tips for Optimization
1. **Batch similar sizes together**: Group samples with similar person counts
2. **Use k-NN graphs**: For large person counts, limit connections to k nearest neighbors
3. **Adjust GCN layers**: Start with 2 layers, add more if needed
4. **Temperature tuning**: Adjust adjacency matrix temperature (0.5-2.0) for best results

## Migration from Fixed n_p

To migrate from fixed n_p model to dynamic GCN:

1. **Replace model instantiation**:
```python
# Old
from models_dual_inter_traj.model import siMLPe
model = siMLPe(config)

# New
from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel
model = DynamicGCNModel(config)
```

2. **Update data processing**:
```python
# Old: fixed batch size
motion_input = torch.randn(batch_size, n_p, seq_len, dim)

# New: variable person count
num_persons = [2, 3, 5, 4]  # varies per sample
max_persons = max(num_persons)
motion_input = torch.randn(batch_size, max_persons, seq_len, dim)
```

3. **Update forward pass**:
```python
# Old
output = model(motion_input, traj)

# New
output = model(motion_input, num_persons=num_persons, distances=distances)
```

4. **Update loss computation**:
```python
# Old: loss over all positions
loss = criterion(output, target)

# New: loss only over valid persons
loss = compute_loss_with_mask(output, target, mask, criterion)
```

## Future Improvements

Potential enhancements:
- [ ] Attention-based graph construction
- [ ] Temporal graph convolution (edges across time)
- [ ] Hierarchical GCN for joint-level and person-level graphs
- [ ] Pre-training on large-scale multi-person datasets
- [ ] Integration with transformer architectures

## References

- Original GCN paper: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
- Multi-person pose forecasting: Adeli et al., "Socially and Contextually Aware Human Motion and Pose Forecasting", IEEE RA-L 2020

## License

Same as parent repository.
