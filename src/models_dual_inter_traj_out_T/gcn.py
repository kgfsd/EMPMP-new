"""
Graph Convolutional Network (GCN) implementation for dynamic person count handling.
This module provides GCN layers and utilities to support variable number of persons
in multi-person motion prediction, removing the fixed n_p limitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Lightweight graph convolution layer that supports dynamic graphs.
    
    Args:
        in_features: Dimension of input node features
        out_features: Dimension of output node features
        bias: Whether to use bias term
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_matrices):
        """
        Forward pass of graph convolution.
        
        Args:
            x: Node features [batch_size, num_persons, feature_dim, seq_len]
            adj_matrices: List of adjacency matrices, one per batch sample
                         Each matrix has shape [num_persons_i, num_persons_i]
        
        Returns:
            Graph convolved features [batch_size, num_persons, feature_dim, seq_len]
        """
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            # Get features for this batch item: [num_persons, feature_dim, seq_len]
            features = x[b]
            adj = adj_matrices[b]
            
            # Apply linear transformation: [num_persons, feature_dim, seq_len]
            # Reshape for matrix multiplication
            num_persons, feat_dim, seq_len = features.shape
            features_flat = features.permute(0, 2, 1).reshape(num_persons * seq_len, feat_dim)
            
            # Linear transformation
            support = torch.matmul(features_flat, self.weight)  # [num_persons*seq_len, out_features]
            support = support.reshape(num_persons, seq_len, self.out_features).permute(0, 2, 1)
            
            # Graph convolution: aggregate from neighbors
            output = torch.matmul(adj, support)  # [num_persons, out_features, seq_len]
            
            if self.bias is not None:
                output = output + self.bias.unsqueeze(0).unsqueeze(-1)
            
            outputs.append(output)
        
        # Stack outputs: [batch_size, num_persons, out_features, seq_len]
        return torch.stack(outputs, dim=0)


def build_dynamic_adjacency_matrix(features, num_persons, temperature=1.0):
    """
    Build dynamic adjacency matrix based on feature similarity.
    
    Args:
        features: Node features [batch_size, max_persons, feature_dim, seq_len]
        num_persons: List or tensor of actual number of persons per batch [batch_size]
        temperature: Temperature for softmax normalization
    
    Returns:
        List of adjacency matrices, one per batch sample
    """
    batch_size = features.size(0)
    adj_matrices = []
    
    for b in range(batch_size):
        # Get valid features for this batch (only actual persons)
        n_persons = num_persons[b] if isinstance(num_persons, (list, tuple)) else num_persons[b].item()
        valid_features = features[b, :n_persons]  # [n_persons, feature_dim, seq_len]
        
        # Average over time dimension for similarity computation
        feat_avg = valid_features.mean(dim=-1)  # [n_persons, feature_dim]
        
        # Compute pairwise similarity (cosine similarity)
        feat_norm = F.normalize(feat_avg, p=2, dim=-1)
        similarity = torch.matmul(feat_norm, feat_norm.transpose(0, 1))  # [n_persons, n_persons]
        
        # Apply temperature and softmax to get adjacency matrix
        adj = F.softmax(similarity / temperature, dim=-1)
        
        adj_matrices.append(adj)
    
    return adj_matrices


def build_distance_based_adjacency_matrix(distances, num_persons, k=None):
    """
    Build adjacency matrix based on spatial distances between persons.
    
    Args:
        distances: Distance matrix [batch_size, time, max_persons, max_persons]
        num_persons: List or tensor of actual number of persons per batch [batch_size]
        k: Number of nearest neighbors to connect (None = fully connected)
    
    Returns:
        List of adjacency matrices, one per batch sample
    """
    batch_size = distances.size(0)
    adj_matrices = []
    
    for b in range(batch_size):
        n_persons = num_persons[b] if isinstance(num_persons, (list, tuple)) else num_persons[b].item()
        
        # Average distances over time: [max_persons, max_persons]
        dist_avg = distances[b].mean(dim=0)
        
        # Extract valid distances: [n_persons, n_persons]
        valid_dist = dist_avg[:n_persons, :n_persons]
        
        # Convert distances to similarities (inverse distance)
        similarity = 1.0 / (valid_dist + 1e-6)
        
        # Set diagonal to 0 (no self-loops initially)
        similarity = similarity - torch.diag(torch.diag(similarity))
        
        if k is not None and k < n_persons:
            # Keep only k-nearest neighbors
            _, indices = torch.topk(similarity, k, dim=-1, largest=True)
            mask = torch.zeros_like(similarity)
            mask.scatter_(1, indices, 1.0)
            similarity = similarity * mask
        
        # Normalize to get adjacency matrix
        adj = F.softmax(similarity, dim=-1)
        
        # Add self-connections
        adj = adj + torch.eye(n_persons, device=adj.device)
        adj = adj / adj.sum(dim=-1, keepdim=True)
        
        adj_matrices.append(adj)
    
    return adj_matrices


def pad_features_to_max_persons(features_list, max_persons):
    """
    Pad feature tensors to have the same number of persons.
    
    Args:
        features_list: List of feature tensors with shape [n_persons_i, feature_dim, seq_len]
        max_persons: Maximum number of persons to pad to
    
    Returns:
        Padded features [batch_size, max_persons, feature_dim, seq_len]
        Mask indicating valid persons [batch_size, max_persons]
    """
    batch_size = len(features_list)
    feat_dim = features_list[0].size(1)
    seq_len = features_list[0].size(2)
    
    # Initialize padded tensor
    padded_features = torch.zeros(batch_size, max_persons, feat_dim, seq_len, 
                                   device=features_list[0].device)
    mask = torch.zeros(batch_size, max_persons, dtype=torch.bool, 
                      device=features_list[0].device)
    
    for b, features in enumerate(features_list):
        n_persons = features.size(0)
        padded_features[b, :n_persons] = features
        mask[b, :n_persons] = True
    
    return padded_features, mask


def create_person_mask(num_persons_list, max_persons, device=None):
    """
    Create boolean mask for valid persons in batch.
    
    Args:
        num_persons_list: List or tensor of number of persons per batch
        max_persons: Maximum number of persons
        device: Target device
    
    Returns:
        Boolean mask [batch_size, max_persons]
    """
    batch_size = len(num_persons_list) if isinstance(num_persons_list, (list, tuple)) else num_persons_list.size(0)
    
    if device is None:
        device = num_persons_list[0].device if torch.is_tensor(num_persons_list[0]) else torch.device('cpu')
    
    mask = torch.zeros(batch_size, max_persons, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        n_persons = num_persons_list[b] if isinstance(num_persons_list, (list, tuple)) else num_persons_list[b].item()
        mask[b, :n_persons] = True
    
    return mask
