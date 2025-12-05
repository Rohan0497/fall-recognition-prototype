from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.dataset import read_split_file
from src.data.openpose_extract import OpenPoseConfig, run_openpose_on_video, parse_openpose_json_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenPose on KTH videos and save skeleton .npy files."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to process (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N videos (0 = no limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute skeletons even if .npy already exists.",
    )
    parser.add_argument(
        "--short_tmp",
        action="store_true",
        help=r"Use short temp JSON path on Windows to avoid MAX_PATH (C:\op_json_tmp_coco).",
    )
    args = parser.parse_args()

    # OpenPose paths (repo-local)
    OPENPOSE_BIN = Path("openpose/bin/OpenPoseDemo.exe")
    MODEL_FOLDER = Path("openpose/models")

    if not OPENPOSE_BIN.exists():
        raise FileNotFoundError(f"OpenPose binary not found: {OPENPOSE_BIN}")
    if not MODEL_FOLDER.exists():
        raise FileNotFoundError(f"OpenPose model folder not found: {MODEL_FOLDER}")

    cfg = OpenPoseConfig(
        openpose_bin=OPENPOSE_BIN,
        model_folder=MODEL_FOLDER,
        model_pose="COCO",        # J=18
        number_people_max=1,
        net_resolution="-1x160",  # fits 4GB GPUs better
    )

    out_root = Path("data/processed/openpose_skeletons_coco")
    tmp_json_root = Path(r"C:\op_json_tmp_coco") if args.short_tmp else Path("data/processed/openpose_json_tmp_coco")

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_json_root.mkdir(parents=True, exist_ok=True)

    # Collect entries
    if args.split == "train":
        entries = read_split_file(Path("splits/train.txt"))
    elif args.split == "val":
        entries = read_split_file(Path("splits/val.txt"))
    elif args.split == "test":
        entries = read_split_file(Path("splits/test.txt"))
    else:
        entries = (
            read_split_file(Path("splits/train.txt"))
            + read_split_file(Path("splits/val.txt"))
            + read_split_file(Path("splits/test.txt"))
        )

    print(f"Split={args.split} | videos={len(entries)} | limit={args.limit if args.limit else 'none'}")
    print(f"Output npy: {out_root}")
    print(f"Temp json:  {tmp_json_root}")

    processed = 0
    skipped = 0
    failed = 0

    for e in entries:
        if args.limit and processed >= args.limit:
            break

        video_path = e.video_path
        cls = video_path.parent.name
        stem = video_path.stem

        out_npy = out_root / cls / f"{stem}.npy"
        if out_npy.exists() and not args.overwrite:
            skipped += 1
            continue

        json_dir = tmp_json_root / cls / stem
        try:
            run_openpose_on_video(video_path, json_dir, cfg)
            skel = parse_openpose_json_dir(json_dir, model_pose=cfg.model_pose)  # (T,J,3)
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, skel)
            print("✅", out_npy)
            processed += 1
        except Exception as ex:
            failed += 1
            print("❌ Failed:", video_path, "->", ex)

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
