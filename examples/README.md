# Examples

This directory contains example scripts demonstrating the usage of the GCN-based global flow module.

## gcn_example.py

Comprehensive example script showing:

1. **Basic GCN Usage**: How to use the DynamicGCN module directly
2. **Configuration-Based Usage**: How to enable GCN through config
3. **Comparison**: Side-by-side comparison with original TransMLP
4. **Variable-Sized Inputs**: Handling different numbers of people
5. **Graph Construction Strategies**: Different ways to construct graphs

### Running the Examples

```bash
cd /path/to/EMPMP-new
python examples/gcn_example.py
```

**Note**: Requires PyTorch and other dependencies from `requirements.txt`.

### Example Output

```
============================================================
GCN-Based Global Flow Module - Usage Examples
============================================================

============================================================
Example 1: Basic GCN Module Usage
============================================================
Input shape: torch.Size([2, 3, 64, 15])
Distance shape: torch.Size([2, 15, 3, 3])
Output shape: torch.Size([2, 3, 64, 15])
✓ Basic GCN usage successful!

... (more examples)

============================================================
All examples completed successfully! ✓
============================================================
```

## Quick Start Snippets

### Enable GCN in Your Model

```python
# In your config file
config.motion_mlp.use_gcn = True
config.motion_mlp.k_neighbors = 2
config.motion_mlp.gcn_layers = 2

# Build model
from src.models_dual_inter_traj_big.mlp import build_mlps
mlp = build_mlps(config.motion_mlp)
```

### Use GCN Directly

```python
from src.models_dual_inter_traj_big.gcn import DynamicGCN
import torch

# Create GCN
gcn = DynamicGCN(dim=64, num_layers=2, k_neighbors=2)

# Forward pass
x = torch.randn(2, 3, 64, 15)  # [B, P, D, T]
distances = torch.rand(2, 15, 3, 3)  # [B, T, P, P]
output = gcn(x, distances)
```

## Additional Resources

- [Usage Guide](../docs/GCN_MODULE_USAGE.md)
- [Implementation Details](../docs/GCN_IMPLEMENTATION_DETAILS.md)
