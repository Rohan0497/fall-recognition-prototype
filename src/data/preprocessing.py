"""Video decoding + sampling into fixed-length clips."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def load_video_clip(
    video_path: Path,
    num_frames: int = 32,
    size: int = 112,
) -> np.ndarray:
    """
    Load a fixed-length clip from a video by uniform sampling.

    Returns:
        clip: float32 array of shape (T, H, W, 3) scaled to [0, 1]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # total frames (may be unreliable for some codecs, but KTH is usually fine)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fallback: read all frames to count (slower, but robust)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        total = len(frames)
        if total == 0:
            raise RuntimeError(f"No frames in video: {video_path}")
        frame_ids = np.linspace(0, total - 1, num_frames).astype(int)
        selected = [frames[i] for i in frame_ids]
    else:
        frame_ids = np.linspace(0, total - 1, num_frames).astype(int)
        selected = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok:
                break
            selected.append(frame)
        cap.release()

        if len(selected) == 0:
            raise RuntimeError(f"Failed to read frames from: {video_path}")

        # pad if we read fewer than num_frames
        while len(selected) < num_frames:
            selected.append(selected[-1])

    # preprocess
    clip = []
    for frame in selected[:num_frames]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        clip.append(frame)

    clip = np.stack(clip, axis=0).astype(np.float32) / 255.0
    return clip

