"""Run OpenPose and store per-frame skeleton keypoints."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ModelPose = Literal["COCO", "BODY_25"]


@dataclass(frozen=True)
class OpenPoseConfig:
    openpose_bin: Path          # path to OpenPoseDemo(.exe)
    model_folder: Path          # path to openpose/models
    model_pose: ModelPose = "COCO"
    number_people_max: int = 1  # keep 1 to avoid multi-person confusion
    net_resolution: str = "-1x256"  # faster, good enough for KTH


def run_openpose_on_video(video_path: Path, out_json_dir: Path, cfg: OpenPoseConfig) -> None:
    out_json_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cfg.openpose_bin),
        "--video", str(video_path),
        "--write_json", str(out_json_dir),
        "--display", "0",
        "--render_pose", "0",
        "--model_folder", str(cfg.model_folder),
        "--model_pose", cfg.model_pose,
        "--number_people_max", str(cfg.number_people_max),
        "--net_resolution", cfg.net_resolution,
    ]

    # Run and raise on failure
    subprocess.run(cmd, check=True)


def _best_person_keypoints(people: list[dict], key: str, num_joints: int) -> np.ndarray:
    """
    Select the person with highest total confidence for the frame.
    Returns (J,3) array [x,y,conf]. If no people -> zeros.
    """
    if not people:
        return np.zeros((num_joints, 3), dtype=np.float32)

    best = None
    best_score = -1.0
    for p in people:
        arr = p.get(key, [])
        if not arr:
            continue
        a = np.array(arr, dtype=np.float32).reshape(-1, 3)
        if a.shape[0] != num_joints:
            continue
        score = float(a[:, 2].sum())
        if score > best_score:
            best_score = score
            best = a

    if best is None:
        return np.zeros((num_joints, 3), dtype=np.float32)
    return best.astype(np.float32)


def parse_openpose_json_dir(json_dir: Path, model_pose: ModelPose = "COCO") -> np.ndarray:
    """
    Reads OpenPose per-frame JSON files and returns skeleton tensor (T,J,3).
    For COCO: J=18 using 'pose_keypoints_2d'
    For BODY_25: J=25 using 'pose_keypoints_2d'
    """
    if model_pose == "COCO":
        num_joints = 18
        key = "pose_keypoints_2d"
    else:
        num_joints = 25
        key = "pose_keypoints_2d"

    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    frames = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        people = data.get("people", [])
        kp = _best_person_keypoints(people, key=key, num_joints=num_joints)  # (J,3)
        frames.append(kp)

    return np.stack(frames, axis=0)  # (T,J,3)


def resample_time(skel: np.ndarray, target_T: int) -> np.ndarray:
    """
    Uniformly resample skeleton sequence to fixed length target_T.
    skel: (T,J,3)
    """
    T = skel.shape[0]
    if T == target_T:
        return skel
    idx = np.linspace(0, T - 1, target_T).astype(int)
    return skel[idx]


def fill_missing(skel: np.ndarray, conf_thresh: float = 0.05) -> np.ndarray:
    """
    Simple forward/back-fill for missing joints (low confidence).
    """
    out = skel.copy()
    T, J, _ = out.shape
    for j in range(J):
        conf = out[:, j, 2]
        mask = conf >= conf_thresh
        if mask.all():
            continue
        # forward fill
        last = None
        for t in range(T):
            if mask[t]:
                last = out[t, j, :2].copy()
            elif last is not None:
                out[t, j, :2] = last
        # backward fill for leading missing
        last = None
        for t in range(T - 1, -1, -1):
            if mask[t]:
                last = out[t, j, :2].copy()
            elif last is not None:
                out[t, j, :2] = last
    return out


def normalize_skeleton_coco(skel: np.ndarray) -> np.ndarray:
    """
    COCO joints index:
    1 neck, 2 RShoulder, 5 LShoulder, 8 RHip, 11 LHip
    Center at neck; scale by shoulder distance (fallback to hip distance).
    """
    out = skel.copy()
    neck = out[:, 1, :2]
    out[:, :, :2] -= neck[:, None, :]

    rsh, lsh = out[:, 2, :2], out[:, 5, :2]
    rhip, lhip = out[:, 8, :2], out[:, 11, :2]

    sh_dist = np.linalg.norm(lsh - rsh, axis=1)
    hip_dist = np.linalg.norm(lhip - rhip, axis=1)
    scale = np.where(sh_dist > 1e-3, sh_dist, hip_dist)
    scale = np.where(scale > 1e-3, scale, 1.0)

    out[:, :, :2] /= scale[:, None, None]
    return out


def skeleton_to_heatmaps(
    skel: np.ndarray,  # (T,J,3) normalized
    out_size: int = 112,
    sigma: float = 2.0,
    conf_thresh: float = 0.05,
) -> np.ndarray:
    """
    Convert skeleton (normalized coords) to a single-channel heatmap video:
    returns (T,H,W) float32 in [0,1] approximately.
    We map normalized coords into image space centered at (H/2, W/2).
    """
    T, J, _ = skel.shape
    H = W = out_size
    vid = np.zeros((T, H, W), dtype=np.float32)

    # coordinate mapping: normalized -> pixels
    cx, cy = W / 2.0, H / 2.0
    scale = min(H, W) / 3.0  # controls how “spread out” the body is on canvas

    # precompute gaussian kernel window size
    rad = int(3 * sigma)
    xs = np.arange(-rad, rad + 1)
    ys = np.arange(-rad, rad + 1)
    gx, gy = np.meshgrid(xs, ys)
    g = np.exp(-(gx**2 + gy**2) / (2 * sigma**2)).astype(np.float32)

    for t in range(T):
        for j in range(J):
            conf = float(skel[t, j, 2])
            if conf < conf_thresh:
                continue
            x_n, y_n = float(skel[t, j, 0]), float(skel[t, j, 1])
            x = int(round(cx + x_n * scale))
            y = int(round(cy + y_n * scale))

            x0, y0 = x - rad, y - rad
            x1, y1 = x + rad, y + rad

            # clip to boundaries
            gx0 = max(0, -x0)
            gy0 = max(0, -y0)
            gx1 = (2 * rad + 1) - max(0, x1 - (W - 1))
            gy1 = (2 * rad + 1) - max(0, y1 - (H - 1))

            ix0 = max(0, x0)
            iy0 = max(0, y0)
            ix1 = min(W, x1 + 1)
            iy1 = min(H, y1 + 1)

            vid[t, iy0:iy1, ix0:ix1] += conf * g[gy0:gy1, gx0:gx1]

    # normalize per-video for stability
    mx = float(vid.max())
    if mx > 1e-6:
        vid /= mx
    return vid

