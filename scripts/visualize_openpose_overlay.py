from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# OpenPose COCO (18) joint order:
# 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist,
# 5 LShoulder, 6 LElbow, 7 LWrist, 8 RHip, 9 RKnee, 10 RAnkle,
# 11 LHip, 12 LKnee, 13 LAnkle, 14 REye, 15 LEye, 16 REar, 17 LEar
COCO_EDGES = [
    (1, 2), (2, 3), (3, 4),        # right arm
    (1, 5), (5, 6), (6, 7),        # left arm
    (1, 8), (8, 9), (9, 10),       # right leg
    (1, 11), (11, 12), (12, 13),   # left leg
    (1, 0),                        # neck-nose
    (0, 14), (14, 16),             # right eye/ear
    (0, 15), (15, 17),             # left eye/ear
]

def draw_skeleton(frame: np.ndarray, skel_xyc: np.ndarray, conf_thresh: float, draw_ids: bool = False) -> np.ndarray:
    """
    frame: HxWx3 BGR
    skel_xyc: (J,3) with x,y,conf in pixel coords
    """
    out = frame.copy()
    H, W = out.shape[:2]

    # Draw edges
    for a, b in COCO_EDGES:
        xa, ya, ca = skel_xyc[a]
        xb, yb, cb = skel_xyc[b]
        if ca < conf_thresh or cb < conf_thresh:
            continue
        xa_i, ya_i = int(round(xa)), int(round(ya))
        xb_i, yb_i = int(round(xb)), int(round(yb))
        if not (0 <= xa_i < W and 0 <= ya_i < H and 0 <= xb_i < W and 0 <= yb_i < H):
            continue
        cv2.line(out, (xa_i, ya_i), (xb_i, yb_i), (0, 255, 0), 2)

    # Draw joints
    for j, (x, y, c) in enumerate(skel_xyc):
        if c < conf_thresh:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(out, (xi, yi), 3, (0, 0, 255), -1)
            if draw_ids:
                cv2.putText(out, str(j), (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_root", default="data/processed/openpose_skeletons_coco_r096")
    ap.add_argument("--raw_root", default="data/raw/kth")
    ap.add_argument("--out_dir", default="outputs/visualizations/openpose_overlays")
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=180, help="Limit frames to export per video (0=no limit).")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf_thresh", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draw_ids", action="store_true")
    args = ap.parse_args()

    skel_root = Path(args.skeleton_root)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(skel_root.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy skeleton files found under: {skel_root}")

    random.seed(args.seed)
    chosen = random.sample(npy_files, k=min(args.num_samples, len(npy_files)))

    print(f"Skeleton root: {skel_root}")
    print(f"Found .npy files: {len(npy_files)}")
    print(f"Exporting samples: {len(chosen)} -> {out_dir}")

    for npy_path in chosen:
        cls = npy_path.parent.name
        stem = npy_path.stem
        video_path = raw_root / cls / f"{stem}.avi"

        if not video_path.exists():
            print(f" missing video for {npy_path.name}: {video_path}")
            continue

        skel = np.load(npy_path)  # (T,J,3)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f" cannot open video: {video_path}")
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prefer mp4; if codec fails on your machine, switch to 'XVID' and .avi
        out_path = out_dir / f"{cls}__{stem}__overlay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))

        t = 0
        max_t = skel.shape[0]
        limit = args.max_frames if args.max_frames and args.max_frames > 0 else max_t
        limit = min(limit, max_t)

        while t < limit:
            ok, frame = cap.read()
            if not ok:
                break

            frame_overlay = draw_skeleton(frame, skel[t], conf_thresh=args.conf_thresh, draw_ids=args.draw_ids)
            cv2.putText(
                frame_overlay,
                f"{cls} | {stem} | frame {t}/{limit}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(frame_overlay)
            t += 1

        cap.release()
        writer.release()
        print(f" wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
