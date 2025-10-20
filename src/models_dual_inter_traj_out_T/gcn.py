import torch
from torch import nn
import torch.nn.functional as F


class DynamicGraphConstruction(nn.Module):
    """
    Dynamically constructs adjacency matrices based on Euclidean distances between people.
    Supports variable numbers of people across batches and frames.
    """
    def __init__(self, self_loop=True):
        """
        Args:
            k_neighbors: If set, create edges to k nearest neighbors
            distance_threshold: If set, create edges for distances below threshold
            self_loop: Whether to add self-loops to the graph
        """
        super().__init__()
        self.self_loop = self_loop
        
    def forward(self, distances, padding_mask=None):
        """
        Args:
            distances: [B, T, P, P] - pairwise Euclidean distances between people
            padding_mask: [B, P] - bool mask indicating real people (True) vs padded (False)
            
        Returns:
            adjacency: [B, T, P, P] - binary or weighted adjacency matrix
        """
        B, T, P, P_check = distances.shape
        assert P == P_check, "Distance matrix must be square"
        
        # Initialize adjacency matrix
        adjacency = torch.zeros_like(distances)
        
        # Create mask for valid connections (both people must be real)
        if padding_mask is not None:
            # padding_mask: [B, P] -> [B, 1, P, P]
            valid_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)  # [B, P, P]
            valid_mask = valid_mask.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, P, P]
        else:
            valid_mask = torch.ones_like(distances, dtype=torch.bool)
        
        
            # Fully connected graph with distance-based weights
            # Use Gaussian kernel to convert distances to edge weights
            # Only compute for valid connections
        valid_distances = distances * valid_mask.float()
        sigma = valid_distances[valid_distances > 0].std() + 1e-8
        adjacency = torch.exp(-distances ** 2 / (2 * sigma ** 2))
            # Apply valid mask
        adjacency = adjacency * valid_mask.float()
        
        # Add self-loops for valid people only
        if self.self_loop and padding_mask is not None:
            for b in range(B):
                actual_people = padding_mask[b].sum().item()
                eye = torch.eye(actual_people, device=distances.device)
                adjacency[b, :, :actual_people, :actual_people] += eye.unsqueeze(0)
        elif self.self_loop:
            eye = torch.eye(P, device=distances.device).unsqueeze(0).unsqueeze(0)
            adjacency = adjacency + eye
            
        # Normalize adjacency matrix (symmetric normalization)
        adjacency = self._normalize_adjacency(adjacency, padding_mask)
        
        return adjacency
    
    def _normalize_adjacency(self, adjacency, padding_mask=None):
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}
        Handles padding by ensuring padded nodes don't affect normalization
        """
        # Compute degree matrix
        degree = adjacency.sum(dim=-1, keepdim=True)
        degree = torch.clamp(degree, min=1e-8)  # Avoid division by zero
        
        # If we have padding, zero out degrees for padded people
        if padding_mask is not None:
            # padding_mask: [B, P] -> [B, 1, P, 1]
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, P, 1]
            degree = degree * mask_expanded.float()
            degree = torch.clamp(degree, min=1e-8)
        
        # D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        
        # D^{-1/2} A D^{-1/2}
        adjacency_normalized = degree_inv_sqrt * adjacency * degree_inv_sqrt.transpose(-2, -1)
        
        # Zero out connections to/from padded people
        if padding_mask is not None:
            valid_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)  # [B, P, P]
            valid_mask = valid_mask.unsqueeze(1).expand(-1, adjacency.size(1), -1, -1)  # [B, T, P, P]
            adjacency_normalized = adjacency_normalized * valid_mask.float()
        
        return adjacency_normalized


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer that operates on dynamic graphs.
    Handles variable numbers of people.
    """
    def __init__(self, in_features, out_features, bias=True, activation=None):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            activation: Activation function (e.g., 'relu', 'tanh')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1e-5)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
    def forward(self, x, adjacency, padding_mask=None):
        """
        Args:
            x: [B, P, D, T] - node features (people features)
            adjacency: [B, T, P, P] - adjacency matrix
            padding_mask: [B, P] - bool mask indicating real people (True) vs padded (False)
            
        Returns:
            out: [B, P, D_out, T] - transformed node features
        """
        B, P, D, T = x.shape
        
        # Rearrange to [B, T, P, D] for easier processing
        x = x.permute(0, 3, 1, 2)  # [B, T, P, D]
        
        # Apply linear transformation: X * W
        x_transformed = torch.matmul(x, self.weight)  # [B, T, P, D_out]
        
        # Graph convolution: A * X * W
        out = torch.matmul(adjacency, x_transformed)  # [B, T, P, D_out]
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        # Apply activation
        out = self.activation(out)
        
        # Zero out features for padded people
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, P, 1]
            mask_expanded = mask_expanded.expand(-1, T, -1, self.out_features)  # [B, T, P, D_out]
            out = out * mask_expanded.float()
        
        # Rearrange back to [B, P, D_out, T]
        out = out.permute(0, 2, 3, 1)  # [B, P, D_out, T]
        
        return out


class GCNBlock(nn.Module):
    """
    A GCN block with residual connection and layer normalization.
    """
    def __init__(self, dim, use_norm=True):
        super().__init__()
        self.gcn = GCNLayer(dim, dim, bias=True, activation='relu')
        self.use_norm = use_norm
        if use_norm:
            from .mlp import LN
            self.norm = LN(dim)
        else:
            self.norm = nn.Identity()
            
    def forward(self, x, adjacency, padding_mask=None):
        """
        Args:
            x: [B, P, D, T] - node features
            adjacency: [B, T, P, P] - adjacency matrix
            padding_mask: [B, P] - bool mask indicating real people
            
        Returns:
            out: [B, P, D, T] - output features
        """
        x_transformed = self.gcn(x, adjacency, padding_mask)
        x_transformed = self.norm(x_transformed)
        
        # Residual connection
        out = x + x_transformed
        
        return out


class DynamicGCN(nn.Module):
    """
    Dynamic GCN module that constructs graphs based on spatial distances
    and applies graph convolutions. Supports variable numbers of people.
    """
    def __init__(self, dim, num_layers=2):
        """
        Args:
            dim: Feature dimension
            num_layers: Number of GCN layers
            k_neighbors: Number of neighbors for k-NN graph construction
            distance_threshold: Distance threshold for edge creation
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Dynamic graph construction
        self.graph_constructor = DynamicGraphConstruction(
            self_loop=True
        ) 
        
        # GCN layers
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(dim, use_norm=True)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, distances, padding_mask=None):
        """
        Args:
            x: [B, P, D, T] - node features (people features)
            distances: [B, T, P, P] - pairwise Euclidean distances
            padding_mask: [B, P] - bool mask indicating real people
            
        Returns:
            out: [B, P, D, T] - output features after GCN
        """
        # Construct dynamic adjacency matrix
        adjacency = self.graph_constructor(distances, padding_mask)
        
        # Apply GCN layers
        for gcn_block in self.gcn_blocks:
            x = gcn_block(x, adjacency, padding_mask)
            
        return x
