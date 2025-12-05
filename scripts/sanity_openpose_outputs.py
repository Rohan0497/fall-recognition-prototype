from __future__ import annotations

from pathlib import Path
import numpy as np

SKEL_ROOT = Path("data/processed/openpose_skeletons_coco")

def main():
    if not SKEL_ROOT.exists():
        raise FileNotFoundError(f"Missing folder: {SKEL_ROOT}")

    npy_files = list(SKEL_ROOT.rglob("*.npy"))
    print("Skeleton root:", SKEL_ROOT)
    print("Found .npy files:", len(npy_files))

    if not npy_files:
        print(" No skeleton files found. Run: python -m scripts.run_openpose_batch")
        return

    # sample up to 5 files
    for fp in npy_files[:5]:
        arr = np.load(fp)
        print("\nFile:", fp)
        print(" shape:", arr.shape, "dtype:", arr.dtype)

        if arr.ndim != 3 or arr.shape[2] != 3:
            print(" Unexpected tensor shape; expected (T, J, 3)")
            continue

        T, J, C = arr.shape
        conf = arr[:, :, 2]
        print(f" T={T}, J={J}, conf min/max=({conf.min():.3f}, {conf.max():.3f}), "
              f"nonzero conf={int((conf>0).sum())}/{T*J}")

    print("\n sanity_openpose_outputs complete.")

if __name__ == "__main__":
    main()
