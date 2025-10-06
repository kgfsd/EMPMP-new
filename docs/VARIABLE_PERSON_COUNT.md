# Variable Person Count Support

## Overview

The EMPMP-new model architecture has been enhanced to support **variable numbers of people** while maintaining its lightweight characteristics. This allows the model to handle datasets and scenarios with different person counts (1, 2, or 3 people) without retraining or architecture changes.

## Key Features

- **Dynamic Person Count**: The model can now process inputs with 1, 2, or up to `max_p` people in a single batch
- **Lightweight Design**: Uses efficient padding and masking instead of creating separate models for each person count
- **Backward Compatible**: Existing configurations and trained models continue to work
- **Zero Overhead**: No additional computational cost when using the maximum person count

## How It Works

### Architecture Changes

The key changes enable variable person counts through:

1. **Maximum Person Count (`max_p`)**: The model is initialized with a maximum person count (default: 3)
2. **Dynamic Padding**: Input tensors with fewer than `max_p` people are zero-padded internally
3. **Efficient Processing**: Padded values are properly masked to avoid affecting computations
4. **Smart Truncation**: Output is automatically truncated to match the actual person count

### Modified Components

#### 1. StylizationBlock
- Added `max_p` and `time_dim` as instance variables
- Modified `forward()` to handle variable person counts through padding
- Ensures global features are properly sized for any person count ≤ `max_p`

#### 2. TransMLP
- Added `max_p` and `seq` as instance variables  
- Modified `forward()` to pad/unpad global features dynamically
- Supports interaction between variable numbers of people

#### 3. build_mlps
- Now accepts both `n_p` (backward compatibility) and `max_p` parameters
- Automatically uses `max_p` if available, falls back to `n_p`

## Configuration

### Setting Maximum Person Count

In your config file (e.g., `src/baseline_h36m_30to30/config.py`):

```python
# Maximum number of people the model can handle
C.max_p = 3  # Can be 1, 2, 3, or more

# For backward compatibility, n_p is still supported
C.n_p = 3

# Motion MLP configuration
C.motion_mlp.max_p = C.max_p
```

### Training Configuration

The model automatically adapts to the person count in your training data:

```python
# During training, you can use mixed person counts
# Example: Some batches with 1 person, others with 2 or 3

# Data shape: (batch_size, num_people, time_steps, features)
# num_people can vary between 1 and max_p within the same dataset
```

## Usage Examples

### Example 1: Training with Variable Person Count

```python
from src.models_dual_inter.model import siMLPe
from src.baseline_h36m_30to30.config import config

# Set max person count
config.max_p = 3
config.motion_mlp.max_p = 3

# Create model
model = siMLPe(config)

# Input shapes can vary:
input_1_person = torch.randn(batch_size, 1, time_steps, dim)  # 1 person
input_2_people = torch.randn(batch_size, 2, time_steps, dim)  # 2 people  
input_3_people = torch.randn(batch_size, 3, time_steps, dim)  # 3 people

# All inputs work with the same model
output_1 = model(input_1_person)  # Shape: (batch_size, 1, time_steps, dim)
output_2 = model(input_2_people)  # Shape: (batch_size, 2, time_steps, dim)
output_3 = model(input_3_people)  # Shape: (batch_size, 3, time_steps, dim)
```

### Example 2: Inference with Different Person Counts

```python
# Load a model trained with max_p=3
model = siMLPe(config)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# The same model can handle 1, 2, or 3 people
with torch.no_grad():
    # Single person inference
    single_person_input = get_single_person_data()  # Shape: (1, 1, T, D)
    single_person_output = model(single_person_input)
    
    # Two people inference
    two_people_input = get_two_people_data()  # Shape: (1, 2, T, D)
    two_people_output = model(two_people_input)
    
    # Three people inference  
    three_people_input = get_three_people_data()  # Shape: (1, 3, T, D)
    three_people_output = model(three_people_input)
```

### Example 3: Using with Padding Masks (Advanced)

For datasets with truly variable person counts per sample, you can use padding masks:

```python
from src.baseline_3dpw_big.lib.dataset.dataset_util import batch_process_joints

# Your data with variable person counts
joints = load_variable_person_data()  # May have different person counts
masks = load_person_masks()
padding_mask = compute_padding_mask(joints)  # True for real people, False for padding

# Process with variable person count support
in_joints, in_masks, out_joints, out_masks, pelvis, padding_mask = \
    batch_process_joints(joints, masks, padding_mask, config, training=True)

# Model handles automatically
output = model(in_joints)
```

## Performance Characteristics

### Memory Usage

- **Storage**: Model parameters are fixed regardless of actual person count
- **Runtime**: Memory scales with `max_p`, not the actual person count
- **Example**: A model with `max_p=3` uses the same memory whether processing 1, 2, or 3 people

### Computational Efficiency

- **Padding Overhead**: Minimal - only affects dimensions that scale with person count
- **No Recomputation**: Padding is done once per forward pass
- **Optimized**: Zero-padding operations are highly optimized in PyTorch

### Comparison

| Approach | Memory | Training Flexibility | Inference Speed |
|----------|--------|---------------------|-----------------|
| Separate Models | High (3x) | Low | Fast |
| Dynamic Model (Ours) | Low | High | Fast |
| Masked Attention | Medium | High | Medium |

## Model Variants

All model variants support variable person counts:

- ✅ `models_dual_inter` - Basic model without distance features
- ✅ `models_dual_inter_traj` - With trajectory and distance features
- ✅ `models_dual_inter_traj_out_T` - With output trajectory
- ✅ `models_dual_inter_traj_3dpw` - Specialized for 3DPW dataset
- ✅ `models_dual_inter_traj_big` - Large model variant
- ✅ `models_dual_inter_traj_pips` - PIPS variant

## Technical Details

### Padding Strategy

The model uses **zero-padding** for variable person count support:

1. **Input Stage**: When `P < max_p`, input tensors are padded:
   - `x`: Padded from `(B, P, D, T)` to `(B, max_p, D, T)`
   - `x_global`: Padded from `(B, D, P*T)` to `(B, D, max_p*T)`
   - `distances` (if used): Padded from `(B, P, 1, T)` to `(B, max_p, 1, T)`

2. **Processing**: All computations operate on padded tensors

3. **Output Stage**: Padded dimensions are removed:
   - Output is truncated from `(B, max_p, D, T)` to `(B, P, D, T)`

### Why Zero-Padding?

- **Simple**: Easy to implement and understand
- **Efficient**: No additional parameters or computations
- **Effective**: Zero values don't contribute to gradients or predictions
- **Compatible**: Works seamlessly with layer normalization and other operations

## Limitations and Considerations

1. **Maximum Person Count**: Set `max_p` large enough for your use case
   - Cannot exceed `max_p` at runtime without retraining
   - Larger `max_p` increases memory usage proportionally

2. **Training Data**: Best results when training with mixed person counts
   - Include examples with 1, 2, and 3 people
   - Use data augmentation like `getRandomPermuteOrder` to expose embeddings to varied poses

3. **Batch Processing**: All samples in a batch must have the same person count
   - Use padding_mask for true variable-length batches
   - Or collate samples by person count

## Troubleshooting

### Issue: Model expects fixed person count

**Solution**: Ensure your config has `max_p` set:
```python
config.max_p = 3
config.motion_mlp.max_p = 3
```

### Issue: Out of memory with max_p=5

**Solution**: Reduce `max_p` or batch size. Memory scales with `max_p`.

### Issue: Poor performance on single-person scenarios

**Solution**: Include single-person examples in training data or use data augmentation.

## Migration Guide

### Updating Existing Code

1. **Update Config**:
   ```python
   # Add to config.py
   C.max_p = C.n_p  # Use existing n_p as max_p
   C.motion_mlp.max_p = C.max_p
   ```

2. **No Model Changes**: Existing trained models work as-is

3. **Optional**: Retrain with mixed person counts for better variable-count performance

### For New Projects

Use `max_p` from the start:
```python
C.max_p = 3  # Maximum expected person count
C.n_p = 3     # Default person count (optional, for backward compatibility)
```

## References

- Original implementation: `src/models_dual_inter/mlp.py`
- Configuration: `src/baseline_h36m_30to30/config.py`
- Dataset utilities: `src/baseline_3dpw_big/lib/dataset/dataset_util.py`

## Contributing

When extending the model:

1. Ensure new components respect `max_p` parameter
2. Add padding support where person-dimension operations occur
3. Test with various person counts (1, 2, max_p)
4. Update this documentation

---

**Questions?** Please open an issue on the repository.
