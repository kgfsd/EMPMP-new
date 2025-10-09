"""
Multi-Person Motion Prediction Models with Dynamic Person Count Support.

This package provides both fixed and dynamic person count models:
- siMLPe: Original model with fixed n_p
- DynamicGCNModel: GCN-based model with variable person count support
- HybridGCNMLP: Hybrid model combining GCN and MLP
"""

from .model import siMLPe
from .gcn import (
    GraphConvolution,
    build_dynamic_adjacency_matrix,
    build_distance_based_adjacency_matrix,
    pad_features_to_max_persons,
    create_person_mask,
)
from .dynamic_gcn_model import DynamicGCNModel, HybridGCNMLP
from .dynamic_data_utils import (
    DynamicPersonBatch,
    collate_variable_persons,
    process_batch_dynamic,
    unpad_batch_dynamic,
    compute_loss_with_mask,
    DynamicPersonDataset,
    create_random_person_count_batch,
    validate_batch_consistency,
    augment_person_count,
)

__all__ = [
    # Original model
    'siMLPe',
    
    # GCN components
    'GraphConvolution',
    'build_dynamic_adjacency_matrix',
    'build_distance_based_adjacency_matrix',
    'pad_features_to_max_persons',
    'create_person_mask',
    
    # Dynamic models
    'DynamicGCNModel',
    'HybridGCNMLP',
    
    # Data utilities
    'DynamicPersonBatch',
    'collate_variable_persons',
    'process_batch_dynamic',
    'unpad_batch_dynamic',
    'compute_loss_with_mask',
    'DynamicPersonDataset',
    'create_random_person_count_batch',
    'validate_batch_consistency',
    'augment_person_count',
]
