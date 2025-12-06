from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import read_split_file
from src.data.openpose_extract import (
    resample_time,
    fill_missing,
    skeleton_to_heatmaps,
)


@dataclass(frozen=True)
class SkeletonHeatmapConfig:
    skeleton_root: Path = Path("data/processed/openpose_skeletons_coco_r096")
    num_frames: int = 32
    size: int = 112
    sigma: float = 2.0
    conf_thresh: float = 0.05


def normalize_skeleton_generic(skel: np.ndarray) -> np.ndarray:
    """
    Generic normalization that works for COCO/BODY_25:
    - Center each frame at mean joint position (x,y)
    - Scale by per-frame RMS distance so sequences are comparable
    """
    out = skel.copy()
    xy = out[:, :, :2]  # (T,J,2)

    center = xy.mean(axis=1, keepdims=True)  # (T,1,2)
    xy = xy - center

    scale = np.sqrt((xy ** 2).sum(axis=2).mean(axis=1))  # (T,)
    scale = np.where(scale > 1e-6, scale, 1.0)
    xy = xy / scale[:, None, None]

    out[:, :, :2] = xy
    return out


class KTHOpenPoseHeatmapDataset(Dataset):
    """
    Loads OpenPose skeletons (.npy) and converts to heatmap "video" clips.

    Returns:
      X: float32 tensor of shape (1, T, H, W)
      y: int64 tensor scalar

    Notes:
      - Uses your existing splits/*.txt (person-independent).
      - Expects OpenPose skeletons at:
          data/processed/openpose_skeletons_coco/<class>/<video_stem>.npy
    """

    def __init__(
        self,
        split_file: str | Path,
        cfg: SkeletonHeatmapConfig = SkeletonHeatmapConfig(),
        strict: bool = True,
    ):
        self.split_file = Path(split_file)
        self.cfg = cfg
        self.strict = strict
        self.entries = read_split_file(self.split_file)

        if not self.cfg.skeleton_root.exists():
            raise FileNotFoundError(
                f"Missing skeleton root folder: {self.cfg.skeleton_root}\n"
                f"Run extraction first: python -m scripts.run_openpose_batch"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _skeleton_path(self, video_path: Path) -> Path:
        cls = video_path.parent.name
        stem = video_path.stem
        return self.cfg.skeleton_root / cls / f"{stem}.npy"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.entries[idx]
        npy_path = self._skeleton_path(e.video_path)

        if not npy_path.exists():
            msg = f"Missing skeleton file: {npy_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            # non-strict mode: return an all-zero clip (keeps training running)
            X = torch.zeros((1, self.cfg.num_frames, self.cfg.size, self.cfg.size), dtype=torch.float32)
            y = torch.tensor(e.label, dtype=torch.long)
            return X, y

        skel = np.load(npy_path).astype(np.float32)               # (T0,J,3)
        skel = resample_time(skel, target_T=self.cfg.num_frames)  # (T,J,3)
        skel = fill_missing(skel, conf_thresh=self.cfg.conf_thresh)
        skel = normalize_skeleton_generic(skel)

        heat = skeleton_to_heatmaps(
            skel,
            out_size=self.cfg.size,
            sigma=self.cfg.sigma,
            conf_thresh=self.cfg.conf_thresh,
        )  # (T,H,W) in [0,1]

        X = torch.from_numpy(heat).unsqueeze(0).float()  # (1,T,H,W)
        y = torch.tensor(e.label, dtype=torch.long)
        return X, y
