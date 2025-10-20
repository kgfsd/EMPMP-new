import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionFeatureExtractor(nn.Module):
    """
    Extracts a fixed-size interaction feature vector F_ipm from a variable-size distance matrix.
    This implementation is permutation-invariant to the number of people.
    It uses a Deep Sets-like architecture.
    """
    def __init__(self, feature_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, distances, padding_mask=None):
        """
        distances: [B, T, P, P]
        padding_mask: [B, P]
        Returns: F_ipm [B, T, D_ipm]
        """
        B, T, P, _ = distances.shape

        # Create a mask for the PxP matrix
        if padding_mask is not None:
            # [B, 1, P, 1] & [B, 1, 1, P] -> [B, 1, P, P]
            p_mask_2d = padding_mask.unsqueeze(1).unsqueeze(3) & padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            p_mask_2d = torch.ones(B, 1, P, P, device=distances.device, dtype=torch.bool)

        # Process each distance value (element-wise)
        # [B, T, P, P, 1]
        dist_reshaped = distances.unsqueeze(-1)
        # [B, T, P, P, D_h]
        embedded_dists = self.mlp1(dist_reshaped)

        # Mask invalid entries before aggregation
        embedded_dists = embedded_dists * p_mask_2d.unsqueeze(-1).float()

        # Aggregate over the "others" dimension for each person
        # Sum over dim 3 (P_j) -> [B, T, P, D_h]
        person_centric_features = embedded_dists.sum(dim=3)

        # Aggregate over the "self" dimension for the scene
        # Sum over dim 2 (P_i) -> [B, T, D_h]
        scene_feature = person_centric_features.sum(dim=2)

        # Normalize by the number of valid pairs if a mask is provided
        if padding_mask is not None:
            num_valid_pairs = torch.clamp(p_mask_2d.sum(dim=[-1, -2]), min=1).float() # [B, 1]
            scene_feature = scene_feature / num_valid_pairs.unsqueeze(-1)

        # Final projection
        f_ipm = self.mlp2(scene_feature) # [B, T, D_ipm]
        return f_ipm


class IPLM(nn.Module):
    """
    Interaction Prior Learning Module (IPLM).
    Learns a codebook of interaction priors and uses them to refine features.
    """
    def __init__(self, knowledge_space_size, feature_dim, lr=0.01, ema_decay=0.99):
        super().__init__()
        self.knowledge_space_size = knowledge_space_size
        self.feature_dim = feature_dim
        self.lr = lr
        self.ema_decay = ema_decay

        # Initialize Interaction Knowledge Space K as a non-trainable buffer
        self.register_buffer('K', torch.randn(knowledge_space_size, feature_dim))
        # EMA averaged cluster size, for stable codebook learning
        self.register_buffer('ema_cluster_size', torch.zeros(knowledge_space_size))
        # EMA averaged codebook vectors
        self.register_buffer('ema_w', self.K.clone())


    def find_nearest_top2(self, f_ipm):
        """
        Finds the top 2 nearest neighbors for each feature vector.
        f_ipm: [N, D]
        Returns:
            d1, d2: distances [N]
            k1_indices, k2_indices: indices [N]
        """
        distances = torch.cdist(f_ipm, self.K)  # [N, S]
        sorted_dists, sorted_indices = torch.sort(distances, dim=-1)
        
        d1 = sorted_dists[..., 0]
        d2 = sorted_dists[..., 1]
        k1_indices = sorted_indices[..., 0]
        k2_indices = sorted_indices[..., 1]
        
        return d1, d2, k1_indices, k2_indices

    def forward(self, f_ipm):
        """
        f_ipm: [B, T, D_ipm]
        """
        original_shape = f_ipm.shape
        f_ipm_flat = f_ipm.view(-1, self.feature_dim)  # [B*T, D]

        if self.training:
            # --- Training Phase ---
            dists = torch.cdist(f_ipm_flat, self.K) # [N, S]
            k1_indices = torch.argmin(dists, dim=1) # [N]
            k1 = self.K[k1_indices]  # [N, D]

            # Calculate the MSE loss for the main training objective
            loss_lk = F.mse_loss(k1.detach(), f_ipm_flat)

            # Update the knowledge space K using EMA (more stable than the paper's formula)
            with torch.no_grad():
                # one-hot encoding of the nearest codebook vectors
                k_one_hot = F.one_hot(k1_indices, num_classes=self.knowledge_space_size).float() # [N, S]
                
                # Sum of vectors for each cluster
                dw = k_one_hot.T @ f_ipm_flat # [S, D]
                
                # Update EMA cluster size
                self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                        (1 - self.ema_decay) * k_one_hot.sum(0)
                
                # Laplace smoothing to avoid zero cluster size
                n = self.ema_cluster_size.sum()
                smoothed_cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.knowledge_space_size * 1e-5) * n
                
                # Update EMA codebook vectors
                self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw
                
                # Normalize and update the codebook
                self.K = self.ema_w / smoothed_cluster_size.unsqueeze(1)

            # Use the quantized vector for the output feature
            f_ipmlm_flat = k1
            f_ipmlm = f_ipmlm_flat.view(original_shape)
            return f_ipmlm, loss_lk

        else:
            # --- Prediction Phase ---
            d1, d2, k1_indices, _ = self.find_nearest_top2(f_ipm_flat)
            
            k1 = self.K[k1_indices]  # [N, D]
            
            # Calculate confidence C
            confidence = 1 - (d1 / (d2 + 1e-8))  # [N]
            
            # Reshape for broadcasting
            confidence = confidence.view(original_shape[0], original_shape[1], 1)
            k1 = k1.view(original_shape)
            
            # Return F_ipmlm = C * k1
            f_ipmlm = confidence * k1
            return f_ipmlm, None