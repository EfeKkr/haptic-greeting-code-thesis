# Dataset class for loading pre-extracted CNN or ViT features.
# Supports dynamic temporal truncation (100%, 60%, 30%) via use_ratio
# Returns (features, label) pairs.

import os
import numpy as np
from torch.utils.data import Dataset
import torch

class FeatureDataset(Dataset):
    def __init__(self, data_dir, label_type='action', file_list=None, use_ratio=1.0):
        self.data_dir = data_dir
        self.label_type = label_type
        self.use_ratio = use_ratio  # e.g., 0.6 for 60%
        self.data_files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.npy')
        ]) if file_list is None else file_list

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        path = os.path.join(self.data_dir, file_name)
        features = np.load(path).astype(np.float32)  # shape: [T, 512]

        # Truncate based on use_ratio
        num_frames = features.shape[0]
        truncated_len = max(1, int(num_frames * self.use_ratio))
        features = features[:truncated_len]

        # Normalize
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

        # Label extraction
        parts = file_name.split('-')
        scenario = int(parts[2][1:])
        action = int(parts[3].split('_')[0][1:])
        label = (action - 1) if self.label_type == 'action' else scenario

        return torch.tensor(features), torch.tensor(label)