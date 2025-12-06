from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.data.openpose_extract import resample_time, fill_missing, skeleton_to_heatmaps

SKEL_ROOT = Path("data/processed/openpose_skeletons_coco_r096")


def normalize_skeleton_generic(skel: np.ndarray) -> np.ndarray:
    """
    Works for any joint layout (COCO/BODY_25).
    - Center each frame at mean joint position (x,y)
    - Scale by per-frame RMS distance (avoid division by 0)
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


def main():
    if not SKEL_ROOT.exists():
        raise FileNotFoundError(f"Missing folder: {SKEL_ROOT}")

    npy_files = list(SKEL_ROOT.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy found under {SKEL_ROOT}. Run extraction first.")

    fp = npy_files[0]
    skel = np.load(fp).astype(np.float32)  # (T0,J,3)
    print("Loaded:", fp)
    print("Raw shape:", skel.shape)

    skel = resample_time(skel, target_T=32)           # (32,J,3)
    skel = fill_missing(skel, conf_thresh=0.05)
    skel = normalize_skeleton_generic(skel)

    heat = skeleton_to_heatmaps(skel, out_size=112, sigma=2.0, conf_thresh=0.05)  # (32,112,112)

    X = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0).float()  # (1,1,32,112,112)
    print("Heatmap clip tensor X:", X.shape, X.dtype)
    print("min/max:", float(X.min()), float(X.max()))

    print(" sanity_openpose_dataset complete.")


if __name__ == "__main__":
    main()
