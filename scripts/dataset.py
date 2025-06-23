#A custom PyTorch dataset to load raw data representing egocentric video recordings stored as .npy files. 
# Every .npy file is a video that has dimensions [T, H, W, C] with T being the video frames. Video files 
# are also modified in a way where they are padded to the length of the longest video to create batch processing. 
# This class primarily finds use prior to feature extraction (e.g. ofViT). Labels are either action type 
# or scenario depending on the file names used.

import torch
import os
import numpy as np
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, data_dir, label_type='action', transform=None, file_list=None):
        self.data_dir = data_dir
        self.label_type = label_type
        self.transform = transform

        self.data_files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.npy')
        ]) if file_list is None else file_list
        
        # Find the maximum number of frames across all videos (used for padding)
        self.max_frames = self._find_max_length()

    def _find_max_length(self):
        max_len = 0
        for file_name in self.data_files:
            path = os.path.join(self.data_dir, file_name)
            frames = np.load(path)
            max_len = max(max_len, frames.shape[0])
        return max_len

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path)  # [T, H, W, C]

        # Pad shorter videos with zeros to match max_frames
        T, H, W, C = data.shape
        if T < self.max_frames:
            pad_shape = (self.max_frames - T, H, W, C)
            padding = np.zeros(pad_shape, dtype=np.float32)
            data = np.concatenate([data, padding], axis=0)

        # Rearrange to [T, C, H, W] for compatibility with CNNs or ViTs
        data = np.transpose(data, (0, 3, 1, 2))  # [T, C, H, W]

        if self.transform:
            data = torch.stack([self.transform(frame) for frame in torch.tensor(data)], dim=0)

        parts = file_name.split('-')
        scenario = int(parts[2][1:])
        action = int(parts[3].split('_')[0][1:])
        label = (action - 1) if self.label_type == 'action' else scenario

        return torch.tensor(data), torch.tensor(label)