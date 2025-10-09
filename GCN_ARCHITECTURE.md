# GCN Architecture for Dynamic Person Count

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EMPMP with Dynamic GCN                           │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Input: Variable Number of Persons                             │  │
│  │ [Batch Sample 1: 2 persons]                                   │  │
│  │ [Batch Sample 2: 3 persons]                                   │  │
│  │ [Batch Sample 3: 5 persons]                                   │  │
│  │ [Batch Sample 4: 4 persons]                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Dynamic Padding to Max Persons (5)                            │  │
│  │ Shape: [4, 5, seq_len, feature_dim]                           │  │
│  │ • Valid persons marked by mask                                │  │
│  │ • Padded positions set to 0                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Node Feature Encoding (Linear Layer)                          │  │
│  │ Input dim → Hidden dim (e.g., 39 → 256)                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Dynamic Adjacency Matrix Construction                         │  │
│  │ • Per-sample graph based on actual person count               │  │
│  │ • Option 1: Feature similarity                                │  │
│  │ • Option 2: Spatial distance                                  │  │
│  │ Output: List of adjacency matrices [A₁, A₂, A₃, A₄]           │  │
│  │   where Aᵢ has shape [nᵢ, nᵢ] (nᵢ = persons in sample i)     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Graph Convolution Layers                                      │  │
│  │ For each GCN layer:                                           │  │
│  │   1. Linear transformation: X' = XW                           │  │
│  │   2. Graph aggregation: Y = AX'                               │  │
│  │   3. Activation: Y = ReLU(Y)                                  │  │
│  │ • Processes only valid persons (guided by adjacency matrix)   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Optional: MLP Layers (Temporal Processing)                    │  │
│  │ • Can be integrated for temporal modeling                     │  │
│  │ • HybridGCNMLP: GCN for person interaction + MLP for time     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Output Projection                                             │  │
│  │ Hidden dim → Output dim (e.g., 256 → 39)                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Output: Predictions for All Persons (with padding)            │  │
│  │ Shape: [4, 5, seq_len, feature_dim]                           │  │
│  │ • Loss computed only on valid persons using mask              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Input Processing

```
Original Data (Variable Size):
Sample 1: [2 persons, 15 frames, 39 dims]
Sample 2: [3 persons, 15 frames, 39 dims]
Sample 3: [5 persons, 15 frames, 39 dims]
Sample 4: [4 persons, 15 frames, 39 dims]

↓ Padding ↓

Padded Batch:
[4 samples, 5 max_persons, 15 frames, 39 dims]

Mask:
[4, 5] boolean tensor
[[True, True, False, False, False],   # Sample 1: 2 valid persons
 [True, True, True, False, False],    # Sample 2: 3 valid persons
 [True, True, True, True, True],      # Sample 3: 5 valid persons
 [True, True, True, True, False]]     # Sample 4: 4 valid persons
```

### 2. Graph Construction

#### Feature-Based (Similarity)
```
For each sample independently:
  1. Extract features: F = [n_persons, feature_dim]
  2. Normalize: F_norm = F / ||F||
  3. Compute similarity: S = F_norm @ F_norm^T
  4. Apply softmax: A = softmax(S / temperature)
  
Result: Adjacency matrix A with shape [n_persons, n_persons]
```

#### Distance-Based (Spatial)
```
For each sample independently:
  1. Get distances: D = [time, n_persons, n_persons]
  2. Average over time: D_avg = mean(D, dim=0)
  3. Convert to similarity: S = 1 / (D_avg + ε)
  4. Normalize: A = softmax(S)
  
Result: Adjacency matrix A with shape [n_persons, n_persons]
```

### 3. Graph Convolution

```
For each sample in batch:
  Input: X = [n_persons, hidden_dim, seq_len]
  Adjacency: A = [n_persons, n_persons]
  
  For each timestep:
    1. Linear: H = X @ W  (learnable weight W)
    2. Aggregate: Y = A @ H  (neighbor aggregation)
    3. Activate: Y = ReLU(Y)
  
  Output: Y = [n_persons, hidden_dim, seq_len]
```

### 4. Loss Computation with Mask

```
Prediction: [batch_size, max_persons, seq_len, dim]
Target:     [batch_size, max_persons, seq_len, dim]
Mask:       [batch_size, max_persons]

For each sample b:
  valid_pred = prediction[b, mask[b]]    # Only valid persons
  valid_target = target[b, mask[b]]      # Only valid persons
  loss += criterion(valid_pred, valid_target)

Final loss = total_loss / batch_size
```

## Data Flow Example

### Example: Batch with [2, 3, 5, 4] persons

```
Step 1: Input (Variable Size)
─────────────────────────────
Sample 0: [2, 15, 39]
Sample 1: [3, 15, 39]
Sample 2: [5, 15, 39]
Sample 3: [4, 15, 39]

Step 2: After Padding
─────────────────────────────
Batch: [4, 5, 15, 39]
  Sample 0: [2 valid + 3 padded]
  Sample 1: [3 valid + 2 padded]
  Sample 2: [5 valid + 0 padded]
  Sample 3: [4 valid + 1 padded]

Step 3: Node Encoding
─────────────────────────────
Batch: [4, 5, 15, 256]  (39→256)

Step 4: Graph Construction
─────────────────────────────
Adjacency matrices (per sample):
  A₀: [2, 2]  - connects 2 persons
  A₁: [3, 3]  - connects 3 persons
  A₂: [5, 5]  - connects 5 persons
  A₃: [4, 4]  - connects 4 persons

Step 5: Graph Convolution
─────────────────────────────
For each sample:
  X₀: [2, 256, 15] → GCN(X₀, A₀) → Y₀: [2, 256, 15]
  X₁: [3, 256, 15] → GCN(X₁, A₁) → Y₁: [3, 256, 15]
  X₂: [5, 256, 15] → GCN(X₂, A₂) → Y₂: [5, 256, 15]
  X₃: [4, 256, 15] → GCN(X₃, A₃) → Y₃: [4, 256, 15]

Stack back: [4, 5, 256, 15] (with padding preserved)

Step 6: Output Projection
─────────────────────────────
Batch: [4, 5, 15, 39]  (256→39)

Step 7: Loss Computation
─────────────────────────────
Only compute loss on valid persons:
  Sample 0: persons 0-1 (2 valid)
  Sample 1: persons 0-2 (3 valid)
  Sample 2: persons 0-4 (5 valid)
  Sample 3: persons 0-3 (4 valid)
```

## Comparison: Fixed vs Dynamic

### Fixed n_p Model (Original)

```
┌──────────────────────────────────┐
│ Always process n_p persons       │
│ Example: n_p = 5                 │
│                                  │
│ Scene with 2 persons:            │
│   • Input: [2 real + 3 padding]  │
│   • Process: All 5 (wasteful!)   │
│   • Loss: All 5 positions        │
│                                  │
│ Scene with 7 persons:            │
│   • Cannot handle! Model fails.  │
└──────────────────────────────────┘
```

### Dynamic GCN Model (New)

```
┌──────────────────────────────────┐
│ Adapt to any number of persons   │
│ Example: Handles 1 to N persons  │
│                                  │
│ Scene with 2 persons:            │
│   • Input: [2 real + padding]    │
│   • Process: Only 2 (efficient!) │
│   • Loss: Only 2 positions       │
│                                  │
│ Scene with 7 persons:            │
│   • Input: [7 real]              │
│   • Process: All 7 (adaptive!)   │
│   • Loss: All 7 positions        │
└──────────────────────────────────┘
```

## Advantages Visualization

```
Metric               Fixed n_p    Dynamic GCN
─────────────────────────────────────────────
Person Count         ❌ Fixed     ✅ Variable
Wasted Computation   ❌ High      ✅ None
Batch Flexibility    ❌ None      ✅ Full
Interaction Model    ⚠️ Implicit  ✅ Explicit
Integration          ✅ N/A       ✅ Seamless
Documentation        ✅ Existing  ✅ Complete
```

## Module Organization

```
models_dual_inter_traj/
│
├── Core GCN Components
│   ├── gcn.py
│   │   ├── GraphConvolution          (GCN layer)
│   │   ├── build_dynamic_adjacency   (Similarity-based)
│   │   └── build_distance_adjacency  (Distance-based)
│   │
│   ├── dynamic_gcn_model.py
│   │   ├── DynamicGCNModel           (Full dynamic model)
│   │   └── HybridGCNMLP              (GCN + MLP hybrid)
│   │
│   └── dynamic_data_utils.py
│       ├── DynamicPersonBatch        (Batch container)
│       ├── collate_variable_persons  (DataLoader collate)
│       ├── compute_loss_with_mask    (Masked loss)
│       └── DynamicPersonDataset      (Dataset wrapper)
│
├── Documentation
│   ├── GCN_README.md                 (Full documentation)
│   ├── QUICK_START.md                (Quick guide)
│   └── GCN_ARCHITECTURE.md           (This file)
│
├── Examples & Tests
│   ├── dynamic_gcn_example.py        (Usage examples)
│   ├── test_dynamic_gcn.py           (Test suite)
│   └── test_syntax.py                (Syntax validation)
│
└── Integration
    └── __init__.py                   (Package interface)
```

## Performance Characteristics

### Computational Complexity

```
Fixed n_p Model:
  • Always: O(n_p × T × D)
  • Wasted when actual persons < n_p

Dynamic GCN Model:
  • Per sample: O(n_actual × T × D + n_actual²)
  • Graph construction: O(n_actual²)
  • GCN forward: O(n_actual² × D)
  • No wasted computation!
```

### Memory Usage

```
Fixed n_p Model:
  • Batch memory: B × n_p × T × D
  • Always allocates for max persons

Dynamic GCN Model:
  • Batch memory: B × max(n_actual) × T × D
  • Adapts to actual batch requirements
  • More efficient when batching similar sizes
```

## Use Cases

### Ideal Scenarios
1. ✅ Scenes with varying person counts
2. ✅ Datasets with mixed occupancy
3. ✅ Real-time applications (don't know person count in advance)
4. ✅ Research requiring person interaction modeling
5. ✅ Limited computational resources (avoid wasted computation)

### When to Use Original Model
1. Fixed n_p known and constant
2. Simple baseline comparisons
3. When graph construction overhead not justified

## Summary

The Dynamic GCN architecture provides:
- **Flexibility**: Handle any number of persons
- **Efficiency**: No wasted computation
- **Explicitness**: Clear person interaction modeling
- **Simplicity**: Easy to use and integrate

All while maintaining compatibility with existing codebase and providing comprehensive documentation.
