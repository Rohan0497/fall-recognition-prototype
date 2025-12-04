from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import read_split_file
from src.data.preprocessing import load_video_clip

class KTHClipDataset(Dataset):
    """
    Loads fixed-length RGB clips from KTH videos using split files.
    Returns:
      X: torch.float32 tensor (C, T, H, W)
      y: torch.long scalar
    """
    def __init__(self, split_file: str, num_frames: int = 32, size: int = 112):
        self.entries = read_split_file(Path(split_file))
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.entries[idx]
        clip = load_video_clip(e.video_path, num_frames=self.num_frames, size=self.size)  # (T,H,W,3)
        # Convert to (C,T,H,W) for 3D CNN in PyTorch
        clip = np.transpose(clip, (3, 0, 1, 2))  # (3,T,H,W)
        X = torch.from_numpy(clip).float()
        y = torch.tensor(e.label, dtype=torch.long)
        return X, y
