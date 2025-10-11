# GCN Implementation Details

## Technical Overview

This document provides implementation details for the Graph Convolutional Network (GCN) based global flow module.

## Key Implementation Points

### 1. Dynamic Graph Construction

The `DynamicGraphConstruction` class creates adjacency matrices dynamically based on pairwise distances:

```python
class DynamicGraphConstruction(nn.Module):
    def __init__(self, k_neighbors=None, distance_threshold=None, self_loop=True):
        # k_neighbors: Use k-NN approach
        # distance_threshold: Use distance-based thresholding
        # self_loop: Add self-connections
```

**Graph Construction Methods:**

1. **K-Nearest Neighbors**: For each person, connects to k closest neighbors
   - Time complexity: O(P² log P) per frame due to sorting
   - Space complexity: O(B × T × P²)
   - Implementation uses `torch.topk` for efficiency

2. **Distance Threshold**: Creates edges when distance < threshold
   - Time complexity: O(P²) per frame
   - Space complexity: O(B × T × P²)
   - Simple comparison operation

3. **Gaussian Kernel**: Fully connected with distance-weighted edges
   - Time complexity: O(P²) per frame
   - Space complexity: O(B × T × P²)
   - Uses `exp(-d² / 2σ²)` for smooth weighting

**Normalization:**

Symmetric normalization is applied: `D^(-1/2) A D^(-1/2)`

This ensures:
- Stable gradient flow
- Consistent scale across different numbers of people
- Standard GCN formulation

### 2. GCN Layer Architecture

```python
class GCNLayer(nn.Module):
    def forward(self, x, adjacency):
        # x: [B, P, D, T]
        # adjacency: [B, T, P, P]
        
        # Rearrange for matrix multiplication
        x = x.permute(0, 3, 1, 2)  # [B, T, P, D]
        
        # Apply linear transformation
        x_transformed = x @ self.weight  # [B, T, P, D_out]
        
        # Graph convolution
        out = adjacency @ x_transformed  # [B, T, P, D_out]
        
        # Rearrange back
        out = out.permute(0, 2, 3, 1)  # [B, P, D_out, T]
```

**Key Design Choices:**

1. **Dimension Ordering**: Uses `[B, P, D, T]` to maintain compatibility with existing code
2. **Permutations**: Minimal permutations for efficiency
3. **Batch Processing**: All operations are fully batched
4. **Residual Connections**: Each GCN block includes residual connections

### 3. Integration with Existing Architecture

The GCN is integrated through `GCNStylizationBlock`:

```python
class GCNStylizationBlock(nn.Module):
    def forward(self, x, x_global, distances):
        # 1. Apply GCN to capture spatial relationships
        x_gcn = self.gcn(x, distances)
        
        # 2. Apply original stylization logic
        x_global_clone = self.emb_layers(x_global)
        scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
        x = x * (1 + scale) + shift
        
        # 3. Integrate GCN output
        x = x + x_gcn
        
        # 4. Update global features
        x_global = x_global + self.global_emb_layers(x_clone)
```

This design:
- Preserves original global-local interaction
- Adds spatial relationship modeling via GCN
- Maintains backward compatibility
- Uses residual connections throughout

### 4. Variable-Sized Input Handling

**Challenge**: Handling different numbers of people (P) across batches or over time.

**Solution**: Dynamic operations that adapt to input size:

```python
def forward(self, x, distances):
    B, P, D, T = x.shape  # P can vary
    
    # All operations work with variable P:
    adjacency = self.graph_constructor(distances)  # Adapts to P
    for gcn_block in self.gcn_blocks:
        x = gcn_block(x, adjacency)  # Matrix ops scale with P
    
    return x
```

**Key Points:**
- No fixed-size assumptions in code
- Matrix operations naturally scale with P
- Memory allocation is dynamic
- Works for P = 1 (single person) to large P

### 5. Efficiency Optimizations

1. **Sparse Graph Construction (k-NN)**:
   - Only computes k edges per node instead of all P edges
   - Reduces memory and computation for large P

2. **Batch Matrix Operations**:
   - All operations use PyTorch's optimized batch matrix multiplication
   - Leverages GPU parallelism

3. **Minimal Data Movement**:
   - Reduces permutations between operations
   - Keeps data in GPU memory

4. **Lazy Initialization**:
   - Graph construction happens during forward pass
   - Adapts to actual input sizes

### 6. Gradient Flow

The architecture ensures good gradient flow through:

1. **Residual Connections**: At every GCN block
2. **Layer Normalization**: Stabilizes training
3. **Xavier Initialization**: Proper weight initialization (gain=1e-8 for output layers)
4. **Skip Connections**: Between local and global features

### 7. Edge Cases

**Single Person (P=1)**:
```python
# Distance matrix is 1×1
distances = torch.zeros(B, T, 1, 1)

# Adjacency becomes identity (with self-loop)
adjacency[b, t] = [[1.0]]

# GCN reduces to identity + MLP
out = self.weight(x)  # No graph aggregation needed
```

**Many People (P >> 1)**:
```python
# Use k-NN to limit connections
graph_constructor = DynamicGraphConstruction(k_neighbors=3)

# Memory scales as O(k × P) instead of O(P²)
# Still captures local spatial structure
```

### 8. Backward Compatibility

The implementation maintains backward compatibility through:

```python
def build_mlps(args):
    use_gcn = getattr(args, 'use_gcn', False)  # Default to False
    
    if use_gcn:
        return TransMLPWithGCN(...)  # New GCN-based
    else:
        return TransMLP(...)  # Original implementation
```

This allows:
- Existing configs work without modification
- Gradual adoption of GCN features
- A/B testing between approaches
- Loading old checkpoints

## Performance Characteristics

### Time Complexity

- **Graph Construction**: O(B × T × P² log P) for k-NN, O(B × T × P²) for threshold/Gaussian
- **GCN Layer**: O(B × T × P² × D) per layer
- **Overall**: O(B × T × P² × D × L) where L is number of GCN layers

### Space Complexity

- **Adjacency Matrix**: O(B × T × P²)
- **Features**: O(B × P × D × T)
- **Total**: O(B × T × P² + B × P × D × T)

### Scaling Behavior

For P people:
- **Memory**: Quadratic in P (due to adjacency matrix)
- **Computation**: Quadratic in P (due to graph operations)
- **Recommended limits**: P < 20 for standard GPU memory

**Mitigation strategies for large P**:
1. Use k-NN with small k (e.g., k=2 or k=3)
2. Process in smaller spatial regions
3. Use sparse matrix representations (future work)

## Implementation Files

```
src/
├── models_dual_inter_traj_big/
│   ├── gcn.py          # GCN module implementation
│   ├── mlp.py          # Updated with GCN integration
│   └── model.py        # Main model (uses gcn through mlp)
└── models_dual_inter_traj_pips/
    ├── gcn.py          # Copy of GCN module
    ├── mlp.py          # Updated with GCN integration
    └── model.py        # Main model (uses gcn through mlp)
```

## Testing

Basic functionality can be tested with:

```python
import torch
from src.models_dual_inter_traj_big.gcn import DynamicGCN

# Test variable sizes
for P in [1, 3, 5, 10]:
    x = torch.randn(2, P, 64, 15)
    distances = torch.rand(2, 15, P, P)
    
    gcn = DynamicGCN(dim=64, num_layers=2, k_neighbors=min(2, P-1))
    out = gcn(x, distances)
    
    assert out.shape == x.shape, f"Failed for P={P}"
    print(f"✓ P={P} works correctly")
```

## Future Enhancements

Potential improvements:

1. **Temporal Graphs**: Add edges across time steps
2. **Attention Mechanisms**: Learn edge weights instead of distance-based
3. **Heterogeneous Graphs**: Different edge types for different interactions
4. **Sparse Implementations**: For very large P
5. **Learned Graph Construction**: Parameterize graph construction
6. **Multi-scale Graphs**: Hierarchical graph structures

## Debugging Tips

1. **Check adjacency matrix**: Print `adjacency[0, 0]` to verify graph structure
2. **Monitor gradient norms**: Ensure gradients flow properly through GCN
3. **Compare with baseline**: Test with `use_gcn=False` for comparison
4. **Visualize graphs**: Plot adjacency matrices to understand connectivity
5. **Profile memory**: Use PyTorch profiler to find memory bottlenecks
