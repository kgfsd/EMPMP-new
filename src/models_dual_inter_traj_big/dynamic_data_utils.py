"""
Data utilities for dynamic person count handling.

This module provides utilities for processing batches with variable number
of persons, including padding, masking, and collation functions.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


class DynamicPersonBatch:
    """
    Container for batch data with variable number of persons.
    
    Attributes:
        motion_data: Padded motion tensor [batch_size, max_persons, seq_len, dim]
        num_persons: List of actual person counts per sample
        mask: Boolean mask [batch_size, max_persons] indicating valid persons
        distances: Optional distance matrix [batch_size, time, max_persons, max_persons]
    """
    
    def __init__(self, motion_data, num_persons, mask=None, distances=None):
        self.motion_data = motion_data
        self.num_persons = num_persons
        self.mask = mask
        self.distances = distances
        self.batch_size = len(num_persons)
        self.max_persons = motion_data.size(1)
    
    def to(self, device):
        """Move batch to device."""
        self.motion_data = self.motion_data.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.distances is not None:
            self.distances = self.distances.to(device)
        return self
    
    def get_valid_outputs(self, outputs):
        """
        Extract only valid (non-padded) outputs from model predictions.
        
        Args:
            outputs: Model outputs [batch_size, max_persons, seq_len, dim]
        
        Returns:
            List of tensors with valid outputs per sample
        """
        valid_outputs = []
        for b in range(self.batch_size):
            n_persons = self.num_persons[b]
            valid_outputs.append(outputs[b, :n_persons, :, :])
        return valid_outputs


def collate_variable_persons(batch):
    """
    Collate function for DataLoader with variable person counts.
    
    Args:
        batch: List of tuples (motion_data, num_persons) or (motion_data, num_persons, distances)
               motion_data has shape [n_persons_i, seq_len, dim]
    
    Returns:
        DynamicPersonBatch object
    """
    # Check batch format
    has_distances = len(batch[0]) > 2
    
    # Extract components
    motion_list = [item[0] for item in batch]
    num_persons = [item[1] for item in batch]
    
    if has_distances:
        distances_list = [item[2] for item in batch]
    else:
        distances_list = None
    
    # Find max persons
    max_persons = max(num_persons)
    batch_size = len(batch)
    seq_len = motion_list[0].size(1)
    dim = motion_list[0].size(2)
    
    # Initialize padded tensors
    padded_motion = torch.zeros(batch_size, max_persons, seq_len, dim)
    mask = torch.zeros(batch_size, max_persons, dtype=torch.bool)
    
    # Fill with actual data
    for b, (motion, n_persons) in enumerate(zip(motion_list, num_persons)):
        padded_motion[b, :n_persons, :, :] = motion
        mask[b, :n_persons] = True
    
    # Handle distances if present
    if distances_list is not None:
        time_steps = distances_list[0].size(0)
        padded_distances = torch.zeros(batch_size, time_steps, max_persons, max_persons)
        for b, (distances, n_persons) in enumerate(zip(distances_list, num_persons)):
            padded_distances[b, :, :n_persons, :n_persons] = distances[:, :n_persons, :n_persons]
    else:
        padded_distances = None
    
    return DynamicPersonBatch(padded_motion, num_persons, mask, padded_distances)


def process_batch_dynamic(batch_data, num_persons_list):
    """
    Process batch data with dynamic person counts.
    
    This function pads sequences to the maximum number of persons in the batch
    and creates appropriate masks.
    
    Args:
        batch_data: List of motion tensors with shape [n_persons_i, seq_len, dim]
        num_persons_list: List of number of persons in each sample
    
    Returns:
        tuple: (padded_data, mask)
            - padded_data: [batch_size, max_persons, seq_len, dim]
            - mask: [batch_size, max_persons] boolean mask
    """
    max_persons = max(num_persons_list)
    batch_size = len(batch_data)
    seq_len = batch_data[0].size(1)
    dim = batch_data[0].size(2)
    
    # Initialize padded tensor
    padded_data = torch.zeros(batch_size, max_persons, seq_len, dim,
                              dtype=batch_data[0].dtype,
                              device=batch_data[0].device)
    
    # Create mask
    mask = torch.zeros(batch_size, max_persons, dtype=torch.bool,
                      device=batch_data[0].device)
    
    # Fill with actual data
    for b, (data, n_persons) in enumerate(zip(batch_data, num_persons_list)):
        padded_data[b, :n_persons, :, :] = data
        mask[b, :n_persons] = True
    
    return padded_data, mask


def unpad_batch_dynamic(padded_data, num_persons_list):
    """
    Remove padding from batch data to recover original variable-sized tensors.
    
    Args:
        padded_data: Padded tensor [batch_size, max_persons, seq_len, dim]
        num_persons_list: List of actual person counts per sample
    
    Returns:
        List of tensors with original sizes [n_persons_i, seq_len, dim]
    """
    batch_size = padded_data.size(0)
    unpadded_list = []
    
    for b in range(batch_size):
        n_persons = num_persons_list[b]
        unpadded_list.append(padded_data[b, :n_persons, :, :])
    
    return unpadded_list


def compute_loss_with_mask(predictions, targets, mask, criterion):
    """
    Compute loss only on valid (non-padded) positions.
    
    Args:
        predictions: Model predictions [batch_size, max_persons, seq_len, dim]
        targets: Ground truth [batch_size, max_persons, seq_len, dim]
        mask: Boolean mask [batch_size, max_persons]
        criterion: Loss function
    
    Returns:
        Average loss over valid positions
    """
    batch_size = predictions.size(0)
    total_loss = 0.0
    total_persons = 0
    
    for b in range(batch_size):
        valid_pred = predictions[b][mask[b]]
        valid_target = targets[b][mask[b]]
        
        if valid_pred.numel() > 0:
            loss = criterion(valid_pred, valid_target)
            total_loss += loss
            total_persons += mask[b].sum().item()
    
    # Average over all valid persons
    if total_persons > 0:
        return total_loss / batch_size
    else:
        return torch.tensor(0.0, device=predictions.device)


class DynamicPersonDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that supports variable number of persons.
    
    This dataset can be used with collate_variable_persons to handle
    batches with different person counts.
    """
    
    def __init__(self, data_list, num_persons_list, distances_list=None):
        """
        Args:
            data_list: List of motion tensors [n_persons_i, seq_len, dim]
            num_persons_list: List of person counts
            distances_list: Optional list of distance tensors
        """
        assert len(data_list) == len(num_persons_list)
        self.data_list = data_list
        self.num_persons_list = num_persons_list
        self.distances_list = distances_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.distances_list is not None:
            return (self.data_list[idx], 
                   self.num_persons_list[idx],
                   self.distances_list[idx])
        else:
            return (self.data_list[idx], 
                   self.num_persons_list[idx])


def create_random_person_count_batch(batch_size, max_persons, seq_len, dim, 
                                     min_persons=1, device='cpu'):
    """
    Create a random batch with variable person counts for testing.
    
    Args:
        batch_size: Number of samples in batch
        max_persons: Maximum number of persons
        seq_len: Sequence length
        dim: Feature dimension
        min_persons: Minimum number of persons per sample
        device: Target device
    
    Returns:
        tuple: (motion_data, num_persons)
    """
    num_persons = torch.randint(min_persons, max_persons + 1, 
                                (batch_size,), device=device).tolist()
    
    motion_data = torch.randn(batch_size, max_persons, seq_len, dim, device=device)
    
    # Zero out padded positions
    for b in range(batch_size):
        motion_data[b, num_persons[b]:, :, :] = 0
    
    return motion_data, num_persons


def validate_batch_consistency(motion_data, num_persons):
    """
    Validate that batch data is consistent with person counts.
    
    Args:
        motion_data: Motion tensor [batch_size, max_persons, seq_len, dim]
        num_persons: List of person counts
    
    Raises:
        AssertionError if validation fails
    """
    batch_size = motion_data.size(0)
    max_persons = motion_data.size(1)
    
    assert len(num_persons) == batch_size, \
        f"num_persons length {len(num_persons)} != batch_size {batch_size}"
    
    for b in range(batch_size):
        n_persons = num_persons[b]
        assert 0 < n_persons <= max_persons, \
            f"Invalid person count {n_persons} (max={max_persons})"
        
        # Check that padded positions are zero (optional, for debugging)
        if n_persons < max_persons:
            padded_region = motion_data[b, n_persons:, :, :]
            # Allow small numerical errors
            assert torch.allclose(padded_region, torch.zeros_like(padded_region), atol=1e-5), \
                f"Padded region not zero for sample {b}"


def augment_person_count(motion_data, num_persons, augment_prob=0.5):
    """
    Data augmentation: randomly drop some persons from samples.
    
    This can be used during training to make the model more robust to
    varying person counts.
    
    Args:
        motion_data: Motion tensor [batch_size, max_persons, seq_len, dim]
        num_persons: List of person counts
        augment_prob: Probability of augmenting each sample
    
    Returns:
        tuple: (augmented_motion_data, augmented_num_persons)
    """
    batch_size = motion_data.size(0)
    augmented_data = motion_data.clone()
    augmented_counts = num_persons.copy()
    
    for b in range(batch_size):
        if torch.rand(1).item() < augment_prob and num_persons[b] > 1:
            # Randomly drop 1 person
            new_count = num_persons[b] - 1
            
            # Choose which person to drop
            drop_idx = torch.randint(0, num_persons[b], (1,)).item()
            
            # Shift remaining persons
            if drop_idx < new_count:
                augmented_data[b, drop_idx:new_count] = augmented_data[b, drop_idx+1:num_persons[b]]
            
            # Zero out the last position
            augmented_data[b, new_count:] = 0
            augmented_counts[b] = new_count
    
    return augmented_data, augmented_counts
