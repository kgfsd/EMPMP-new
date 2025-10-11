# EMPMP-new

## Enhanced Multi-Person Motion Prediction with GCN

This repository contains implementations for multi-person motion prediction with an enhanced GCN-based global flow module.

## Recent Updates

### GCN-Based Global Flow Module (New!)

We've added a Graph Convolutional Network (GCN) based global flow module that improves handling of variable numbers of people in the scene.

**Key Features:**
- Dynamic graph construction based on Euclidean distances
- Support for variable-sized inputs (different numbers of people)
- Multiple graph construction strategies (k-NN, threshold, Gaussian kernel)
- Backward compatible with existing code
- Efficient handling of edge cases (single person, many people)

**Quick Start:**

```python
# Enable GCN in your config
config.motion_mlp.use_gcn = True
config.motion_mlp.k_neighbors = 2  # Optional
config.motion_mlp.gcn_layers = 2   # Optional

# Build model (automatically uses GCN)
from src.models_dual_inter_traj_big.mlp import build_mlps
mlp = build_mlps(config)
```

**Documentation:**
- [Usage Guide](docs/GCN_MODULE_USAGE.md) - How to use the GCN module
- [Implementation Details](docs/GCN_IMPLEMENTATION_DETAILS.md) - Technical details

## Requirements

See `requirements.txt` for dependencies.

## License

See LICENSE file for details.