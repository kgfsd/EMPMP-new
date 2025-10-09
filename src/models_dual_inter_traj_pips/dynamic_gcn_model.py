"""
Dynamic GCN Model for Multi-Person Motion Prediction with Variable Person Count.

This module provides a GCN-based model that removes the fixed n_p limitation
by supporting dynamic number of persons in each batch sample.
"""

import copy
import torch
from torch import nn
from einops.layers.torch import Rearrange

from .gcn import (
    GraphConvolution,
    build_dynamic_adjacency_matrix,
    build_distance_based_adjacency_matrix,
)
from .mlp import build_mlps


class DynamicGCNModel(nn.Module):
    """
    Dynamic GCN model that supports variable number of persons.
    
    This model replaces fixed person count (n_p) with dynamic graph construction
    that adapts to varying numbers of persons in each batch sample.
    
    Args:
        config: Configuration object with model parameters
    """
    
    def __init__(self, config):
        super(DynamicGCNModel, self).__init__()
        self.config = copy.deepcopy(config)
        
        # 1. Node feature encoder (replaces motion_fc_in)
        self.node_encoder = nn.Linear(
            config.motion.dim if hasattr(config.motion, 'dim') else config.motion.h36m_input_length_dct,
            config.hidden_dim if hasattr(config, 'hidden_dim') else 256
        )
        
        # 2. Graph convolution layers
        hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 256
        num_gcn_layers = getattr(config, 'num_gcn_layers', 2)
        
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        
        # 3. Optional: Keep original MLP structure for processing
        if hasattr(config, 'motion_mlp'):
            self.motion_mlp = build_mlps(config.motion_mlp)
        else:
            self.motion_mlp = None
        
        # 4. Output projection
        output_dim = config.motion.dim if hasattr(config.motion, 'dim') else config.motion.h36m_input_length_dct
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Configuration for adjacency matrix construction
        self.use_distance_based_adj = getattr(config, 'use_distance_based_adj', True)
        self.adj_temperature = getattr(config, 'adj_temperature', 1.0)
        
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.node_encoder.weight)
        nn.init.zeros_(self.node_encoder.bias)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=1e-8)
        nn.init.zeros_(self.output_projection.bias)
    
    def build_dynamic_graph(self, features, num_persons, distances=None):
        """
        Build dynamic adjacency matrices for each sample in the batch.
        
        Args:
            features: Node features [batch_size, max_persons, hidden_dim, seq_len]
            num_persons: Number of persons per batch sample [batch_size] or list
            distances: Optional distance matrix [batch_size, time, max_persons, max_persons]
        
        Returns:
            List of adjacency matrices
        """
        if self.use_distance_based_adj and distances is not None:
            return build_distance_based_adjacency_matrix(distances, num_persons)
        else:
            return build_dynamic_adjacency_matrix(features, num_persons, self.adj_temperature)
    
    def forward(self, motion_input, num_persons=None, distances=None):
        """
        Forward pass with dynamic person count support.
        
        Args:
            motion_input: Input motion features [batch_size, max_persons, seq_len, dim]
                         or [batch_size, max_persons, dim, seq_len] depending on config
            num_persons: Number of actual persons per batch sample [batch_size]
                        If None, assumes all positions are valid (no padding)
            distances: Optional pre-computed distances [batch_size, time, max_persons, max_persons]
        
        Returns:
            Predicted motion features [batch_size, max_persons, seq_len, dim]
        """
        batch_size = motion_input.size(0)
        
        # Handle case where num_persons is not provided
        if num_persons is None:
            max_persons = motion_input.size(1)
            num_persons = [max_persons] * batch_size
        
        # Ensure correct input format: [batch_size, max_persons, seq_len, dim]
        if motion_input.size(-1) != (self.config.motion.dim if hasattr(self.config.motion, 'dim') 
                                      else self.config.motion.h36m_input_length_dct):
            motion_input = motion_input.transpose(2, 3)
        
        # 1. Node feature encoding
        # Reshape for encoding: [batch_size * max_persons * seq_len, dim]
        b, p, t, d = motion_input.shape
        motion_flat = motion_input.reshape(b * p * t, d)
        node_features_flat = self.node_encoder(motion_flat)  # [b*p*t, hidden_dim]
        
        # Reshape to [batch_size, max_persons, seq_len, hidden_dim]
        hidden_dim = node_features_flat.size(-1)
        node_features = node_features_flat.reshape(b, p, t, hidden_dim)
        
        # Rearrange to [batch_size, max_persons, hidden_dim, seq_len]
        node_features = self.arr0(node_features)
        
        # 2. Build dynamic adjacency matrices
        adj_matrices = self.build_dynamic_graph(node_features, num_persons, distances)
        
        # 3. Apply GCN layers
        graph_features = node_features
        for gcn_layer in self.gcn_layers:
            graph_features = gcn_layer(graph_features, adj_matrices)
            graph_features = torch.relu(graph_features)
        
        # 4. Optional: Apply MLP processing
        if self.motion_mlp is not None:
            # The MLP expects [B, P, D, T] format
            if distances is not None:
                graph_features = self.motion_mlp(graph_features, distances)
            else:
                # If MLP doesn't support distances, try without
                try:
                    graph_features = self.motion_mlp(graph_features, distances)
                except TypeError:
                    # MLP doesn't take distances parameter
                    graph_features = self.motion_mlp(graph_features)
        
        # 5. Output projection
        # Rearrange back to [batch_size, max_persons, seq_len, hidden_dim]
        graph_features = self.arr1(graph_features)
        
        # Project to output dimension
        output = graph_features.reshape(b * p * t, hidden_dim)
        output = self.output_projection(output)
        output = output.reshape(b, p, t, -1)
        
        return output


class HybridGCNMLP(nn.Module):
    """
    Hybrid model combining GCN for person interaction with MLP for temporal modeling.
    
    This provides an alternative architecture that uses GCN specifically for
    modeling inter-person relationships while keeping temporal modeling in MLP.
    """
    
    def __init__(self, config):
        super(HybridGCNMLP, self).__init__()
        self.config = copy.deepcopy(config)
        
        # Input processing
        self.temporal_fc_in = getattr(config.motion_fc_in, 'temporal_fc', False)
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                config.motion.h36m_input_length_dct,
                config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_in = nn.Linear(config.motion.dim, config.motion.dim)
        
        # GCN for person interaction
        hidden_dim = getattr(config, 'hidden_dim', 256)
        self.person_gcn = GraphConvolution(config.motion.dim, hidden_dim)
        
        # MLP for temporal processing (keep original structure)
        self.motion_mlp = build_mlps(config.motion_mlp)
        
        # Output processing
        self.temporal_fc_out = getattr(config.motion_fc_out, 'temporal_fc', False)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                config.motion.h36m_input_length_dct,
                config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_out = nn.Linear(hidden_dim, config.motion.dim)
        
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.zeros_(self.motion_fc_out.bias)
    
    def forward(self, motion_input, num_persons=None, distances=None):
        """
        Forward pass combining GCN and MLP.
        
        Args:
            motion_input: Input features [batch_size, max_persons, seq_len, dim]
            num_persons: Number of persons per sample [batch_size]
            distances: Distance matrix [batch_size, time, max_persons, max_persons]
        
        Returns:
            Output features [batch_size, max_persons, seq_len, dim]
        """
        batch_size = motion_input.size(0)
        
        if num_persons is None:
            max_persons = motion_input.size(1)
            num_persons = [max_persons] * batch_size
        
        # Input transformation
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)
        
        # Build adjacency matrix
        if distances is not None:
            adj_matrices = build_distance_based_adjacency_matrix(distances, num_persons)
        else:
            adj_matrices = build_dynamic_adjacency_matrix(motion_feats, num_persons)
        
        # Apply GCN for person interaction
        person_features = self.person_gcn(motion_feats, adj_matrices)
        person_features = torch.relu(person_features)
        
        # Apply MLP for temporal modeling
        if distances is not None:
            try:
                motion_feats = self.motion_mlp(person_features, distances)
            except TypeError:
                motion_feats = self.motion_mlp(person_features)
        else:
            motion_feats = self.motion_mlp(person_features)
        
        # Output transformation
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)
        
        return motion_feats
