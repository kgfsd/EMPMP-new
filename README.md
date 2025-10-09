# EMPMP-new

Multi-Person Motion Prediction with Enhanced Models

## Features

### ⭐ NEW: Dynamic GCN for Variable Person Count

We now support **variable number of persons** through Graph Convolutional Networks (GCN)! 

**Key Benefits:**
- ✅ Handle 1 to N persons in the same model (no more fixed n_p!)
- ✅ Explicit person interaction modeling via graph structures
- ✅ Efficient batching with dynamic padding
- ✅ Easy integration with existing models

**Quick Start:**
```python
from models_dual_inter_traj import DynamicGCNModel

model = DynamicGCNModel(config)
output = model(motion_input, num_persons=[2, 3, 5, 4])  # Variable counts!
```

**Documentation:**
- 📖 [Full Documentation](src/models_dual_inter_traj/GCN_README.md)
- 🚀 [Quick Start Guide](src/models_dual_inter_traj/QUICK_START.md)
- 💡 [Examples](src/models_dual_inter_traj/dynamic_gcn_example.py)

## Repository Structure

```
src/
├── models_dual_inter_traj/         # Main model implementations
│   ├── gcn.py                      # GCN layer implementation
│   ├── dynamic_gcn_model.py        # Dynamic GCN models
│   ├── dynamic_data_utils.py       # Data utilities for variable person count
│   ├── GCN_README.md               # Comprehensive GCN documentation
│   └── QUICK_START.md              # Quick start guide
├── models_dual_inter/              # Alternative model variants
├── models_dual_inter_traj_pips/    # PIPS variant
├── models_dual_inter_traj_big/     # Large model variant
├── models_dual_inter_traj_out_T/   # Output temporal variant
├── baseline_3dpw/                  # 3DPW baseline
└── baseline_h36m_*/                # H36M baselines
```

## Installation

```bash
# Clone repository
git clone https://github.com/kgfsd/EMPMP-new.git
cd EMPMP-new

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Original Fixed Person Count Models

```python
from src.models_dual_inter_traj.model import siMLPe

model = siMLPe(config)
output = model(motion_input, traj)
```

### New Dynamic GCN Models

```python
from src.models_dual_inter_traj import DynamicGCNModel

model = DynamicGCNModel(config)

# Handle variable person counts in batch
num_persons = [2, 3, 5, 4]  # Different for each sample
output = model(motion_input, num_persons=num_persons)
```

See [QUICK_START.md](src/models_dual_inter_traj/QUICK_START.md) for detailed usage examples.

## What's New

### Dynamic Person Count Support (Latest)

- **GraphConvolution layer**: Lightweight GCN for person interaction
- **DynamicGCNModel**: Full model with variable person count
- **HybridGCNMLP**: Combines GCN with original MLP structure
- **Dynamic data utilities**: Tools for handling variable-sized batches
- **Comprehensive documentation**: README, Quick Start, and examples

### Key Improvements

1. **No Fixed n_p**: Models can now handle any number of persons (1 to N)
2. **Graph-based Interaction**: Explicit modeling of person-person relationships
3. **Flexible Batching**: Support for different person counts in same batch
4. **Efficient Processing**: Only valid persons are processed (no wasted computation)
5. **Easy Integration**: Compatible with existing model components

## Examples

Run the comprehensive examples:

```bash
cd src/models_dual_inter_traj
python dynamic_gcn_example.py
```

This demonstrates:
- Basic usage with variable person counts
- Distance-based adjacency matrix construction
- Batch processing with dynamic padding
- Training loop with variable person counts

## License

[Add your license information here]

## Citation

[Add citation information here]

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md] for guidelines.

## Contact

[Add contact information here]