# Variable Person Count Support (9-15 People)

This document describes the implementation of variable person count support in the EMPMP model architecture, enabling dynamic handling of 9-15 people while maintaining a lightweight model.

## Overview

The model now supports variable person counts through a padding and masking mechanism, allowing:
- **Fixed maximum persons**: Up to 15 people
- **Dynamic person counts**: 9-15 people per scene
- **Variable persons within dataset**: People can enter/exit scenes
- **Lightweight architecture**: Parameters are shared across person counts

## Key Changes

### 1. Model Architecture Updates

#### `StylizationBlock` (in `mlp.py`)
- Added `max_p` parameter for maximum person support
- Modified linear layers to use `max_p` instead of fixed `num_p`
- Added `padding_mask` parameter to forward method
- Implements padding for inputs with fewer than `max_p` persons
- Applies mask to zero out padded persons in computations

```python
# Example usage
block = StylizationBlock(time_dim=30, num_p=3, dim=45, max_p=15)
```

#### `TransMLP` (in `mlp.py`)
- Added `max_p` parameter for maximum person support
- Global MLPs use `seq*max_p` dimension to accommodate variable persons
- Forward method pads inputs to `max_p` and applies masking
- Only processes valid persons, extracts original person count in output

```python
# Example usage
mlp = TransMLP(dim=45, seq=30, use_norm=True, use_spatial_fc=False,
               num_layers=48, layernorm_axis='spatial',
               interaction_interval=4, p=3, max_p=15)
```

#### `siMLPe` Model (in `model.py`)
- Added `padding_mask` parameter to forward method
- Passes mask through to the MLP layers

```python
# Example usage
output = model(motion_input, traj, padding_mask=padding_mask)
```

### 2. Dataset Updates

#### `DATA` class (in `dataset_mocap.py`)
- Added `max_p` parameter for maximum person support
- Added `variable_persons` flag to enable variable person mode
- Stores both `n_p` (typical count) and `max_p` (maximum count)

```python
# Example usage
dataset = DATA(mode="train", t_his=30, t_pred=30, n_p=3, max_p=15, variable_persons=True)
```

### 3. Configuration Updates

To enable variable person support, add the following to your config:

```python
config.motion_mlp.n_p = 9  # Minimum/typical person count
config.motion_mlp.max_p = 15  # Maximum person count
```

## Usage Examples

### Training with Variable Persons

```python
import torch
from src.models_dual_inter_traj_big.model import siMLPe

# Configure model
config.motion_mlp.n_p = 9
config.motion_mlp.max_p = 15

model = siMLPe(config)

# Prepare batch with variable persons (e.g., 10 and 15 persons)
batch_size = 2
max_persons = 15

# Sample 1: 10 persons
motion_input_1 = torch.randn(1, 10, 30, 45)
traj_1 = torch.randn(1, 10, 30, 15, 3)
# Pad to max_persons
motion_input_1 = torch.cat([motion_input_1, torch.zeros(1, 5, 30, 45)], dim=1)
traj_1 = torch.cat([traj_1, torch.zeros(1, 5, 30, 15, 3)], dim=1)
padding_mask_1 = torch.cat([torch.ones(1, 10), torch.zeros(1, 5)], dim=1)

# Sample 2: 15 persons
motion_input_2 = torch.randn(1, 15, 30, 45)
traj_2 = torch.randn(1, 15, 30, 15, 3)
padding_mask_2 = torch.ones(1, 15)

# Combine into batch
motion_input = torch.cat([motion_input_1, motion_input_2], dim=0)
traj = torch.cat([traj_1, traj_2], dim=0)
padding_mask = torch.cat([padding_mask_1, padding_mask_2], dim=0)

# Forward pass
output = model(motion_input, traj, padding_mask=padding_mask)
# output shape: [2, 15, 30, 45] - but only valid persons are computed
```

### Testing Different Person Counts

```python
# Test with different person counts
for n_persons in [9, 10, 12, 15]:
    motion_input = torch.randn(1, n_persons, 30, 45)
    traj = torch.randn(1, n_persons, 30, 15, 3)
    padding_mask = torch.ones(1, n_persons)
    
    output = model(motion_input, traj, padding_mask=padding_mask)
    print(f"Processed {n_persons} persons successfully")
```

## Implementation Details

### Padding Strategy
- All inputs are padded to `max_p` (15 persons)
- Padded positions are filled with zeros
- Padding mask indicates valid persons (1) vs padded (0)

### Masking Mechanism
- Applied after each transformation in `StylizationBlock`
- Ensures padded persons don't contribute to gradients
- Masked positions remain zero throughout computation

### Lightweight Design
- Model parameters are shared across all person counts
- No additional parameters needed for different person counts
- Global features dimension adapts to `max_p * seq_len`
- Efficient memory usage through padding only when needed

### Backward Compatibility
- If `max_p` is not specified, defaults to `n_p` (original behavior)
- Existing code continues to work without modification
- `padding_mask` is optional; if not provided, assumes all persons are valid

## Benefits

1. **Flexibility**: Handle 9-15 people dynamically
2. **Efficiency**: Shared parameters keep model lightweight
3. **Robustness**: Handles variable person counts in same dataset/batch
4. **Scalability**: Easy to adjust max_p for different requirements

## Performance Considerations

- Computational cost scales with `max_p`, not actual person count
- For best performance, batch samples with similar person counts
- Padding overhead is minimal for counts close to `max_p`
- Memory usage is proportional to `max_p * batch_size`

## Testing

Run the test script to verify variable person support:

```bash
python test_variable_persons.py
```

This will test the model with:
- Different fixed person counts (9, 12, 15)
- Variable persons in the same batch
- Padding mask functionality

## Notes

- The implementation focuses on `models_dual_inter_traj_big` and `models_dual_inter_traj_pips` variants
- Other model variants can be updated similarly if needed
- Dataset loaders should be modified to provide padding masks when using variable person counts
