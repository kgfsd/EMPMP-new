# Quick Start Guide: Dynamic GCN for Variable Person Count

This guide provides a quick introduction to using the Dynamic GCN implementation for handling variable numbers of persons in multi-person motion prediction.

## Problem

The original model has a **fixed person count (n_p)**, which means:
- ❌ Cannot handle scenes with varying numbers of people
- ❌ Must train separate models for different person counts
- ❌ Inefficient when some samples have fewer than n_p persons

## Solution

The **Dynamic GCN** removes this limitation by:
- ✅ Supporting any number of persons (1 to N)
- ✅ Using graph convolutions to model person interactions
- ✅ Efficiently handling variable-sized batches with padding

---

## Installation

The implementation is already included in the `models_dual_inter_traj` package:

```python
from models_dual_inter_traj import (
    DynamicGCNModel,
    HybridGCNMLP,
    collate_variable_persons,
    compute_loss_with_mask,
)
```

**Requirements:**
- PyTorch >= 1.11.0
- einops
- (existing repository dependencies)

---

## Basic Usage (3 Steps)

### Step 1: Create Model

```python
from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel

# Use your existing config
model = DynamicGCNModel(config)
```

### Step 2: Prepare Data

```python
import torch

# Variable number of persons per sample
num_persons = [2, 3, 5, 4]  # Different for each batch sample
max_persons = max(num_persons)  # 5

# Create padded input
batch_size = 4
seq_len = 15
input_dim = 39
motion_input = torch.randn(batch_size, max_persons, seq_len, input_dim)

# Zero out padded positions
for b in range(batch_size):
    motion_input[b, num_persons[b]:, :, :] = 0
```

### Step 3: Forward Pass

```python
# Forward pass with variable person counts
output = model(motion_input, num_persons=num_persons)

# Output has same shape as input
print(output.shape)  # [4, 5, 15, 39]
```

---

## Training Loop

```python
from models_dual_inter_traj.dynamic_data_utils import compute_loss_with_mask

# Setup
model = DynamicGCNModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Training
for epoch in range(num_epochs):
    for motion_input, motion_target, num_persons, mask in dataloader:
        # Forward
        output = model(motion_input, num_persons=num_persons)
        
        # Loss only on valid persons
        loss = compute_loss_with_mask(output, motion_target, mask, criterion)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## DataLoader Setup

```python
from torch.utils.data import DataLoader
from models_dual_inter_traj.dynamic_data_utils import (
    DynamicPersonDataset,
    collate_variable_persons
)

# Create dataset with variable person counts
dataset = DynamicPersonDataset(data_list, num_persons_list)

# Use custom collate function
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_variable_persons,
    shuffle=True
)
```

---

## With Distance Information

If you have distance/trajectory data:

```python
# Compute distances between persons
distances = compute_distances(trajectories)  # [B, T, P, P]

# Forward pass with distances
output = model(
    motion_input, 
    num_persons=num_persons,
    distances=distances  # Used for adjacency matrix
)
```

---

## Configuration

Extend your existing config with GCN settings:

```python
config.hidden_dim = 256              # GCN hidden dimension
config.num_gcn_layers = 2            # Number of GCN layers
config.use_distance_based_adj = True # Use distances for graph
config.adj_temperature = 1.0         # Adjacency softmax temperature
```

---

## Examples

Run the comprehensive examples:

```bash
cd src/models_dual_inter_traj
python dynamic_gcn_example.py
```

This demonstrates:
1. ✅ Basic usage with variable person counts
2. ✅ Distance-based adjacency construction
3. ✅ Batch processing with padding
4. ✅ Training loop simulation

---

## Key Differences from Original Model

| Aspect | Original (siMLPe) | Dynamic GCN |
|--------|------------------|-------------|
| Person count | Fixed (n_p) | Variable (1 to N) |
| Forward pass | `model(x, traj)` | `model(x, num_persons, distances)` |
| Batch structure | Fixed shape | Padded with mask |
| Person interaction | Implicit (concatenation) | Explicit (graph convolution) |
| Loss computation | All positions | Only valid persons |

---

## Common Patterns

### 1. Single Forward Pass
```python
output = model(motion_input, num_persons=[2, 3, 4])
```

### 2. With Distances
```python
output = model(motion_input, num_persons=[2, 3, 4], distances=dist_matrix)
```

### 3. Extract Valid Outputs
```python
for b in range(batch_size):
    valid_output = output[b, :num_persons[b], :, :]
    # Process only valid persons
```

### 4. Masked Loss
```python
loss = compute_loss_with_mask(pred, target, mask, criterion)
```

---

## Tips

### Memory Efficiency
- Batch samples with similar person counts together
- Use smaller batch sizes for scenes with many persons
- Consider gradient accumulation for large person counts

### Graph Construction
- Use **distance-based** adjacency for trajectory data
- Use **similarity-based** for feature-based scenes
- Adjust `adj_temperature` (0.5-2.0) for connection strength

### Performance
- Start with 2 GCN layers, add more if needed
- For very large person counts, use k-nearest neighbor graphs
- Profile your specific use case to optimize

---

## Troubleshooting

### Shape Mismatch Error
**Problem:** Input shape doesn't match expected format  
**Solution:** Ensure input is `[batch_size, max_persons, seq_len, dim]`

### Memory Error
**Problem:** Out of memory with large person counts  
**Solution:** Reduce batch size or use gradient accumulation

### No Improvement in Training
**Problem:** Loss not decreasing  
**Solution:** Check learning rate, verify mask is correct, try different adjacency strategy

---

## Next Steps

1. **Read Full Documentation**: See `GCN_README.md` for detailed API reference
2. **Run Examples**: Execute `dynamic_gcn_example.py` to see all features
3. **Adapt to Your Data**: Modify example code for your specific dataset
4. **Experiment**: Try different configurations and graph construction strategies

---

## Support

- 📖 Full documentation: `GCN_README.md`
- 💡 Examples: `dynamic_gcn_example.py`
- 🧪 Tests: `test_dynamic_gcn.py`
- 📝 Implementation details: See individual module docstrings

---

## Citation

If you use this Dynamic GCN implementation in your research, please cite the original repository and mention the GCN extension.

---

**Happy coding!** 🚀
