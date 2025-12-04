"""PyTorch Dataset that loads skeleton tensors + labels."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass(frozen=True)
class SplitEntry:
    video_path: Path
    label: int

def read_split_file(split_file: Path) -> List[SplitEntry]:
    entries: List[SplitEntry] = []
    with split_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in {split_file} at {i}: {line}")
            rel_path, y = parts[0], int(parts[1])
            p = Path(rel_path)
            if not p.exists():
                raise FileNotFoundError(f"Missing file referenced by split: {p}")
            entries.append(SplitEntry(video_path=p, label=y))
    return entries

if __name__ == "__main__":
    # Quick sanity check (run: python -m src.data.dataset)
    train = read_split_file(Path("splits/train.txt"))
    val = read_split_file(Path("splits/val.txt"))
    test = read_split_file(Path("splits/test.txt"))

    print("✅ Loaded splits")
    print("train", len(train), "val", len(val), "test", len(test))
    print("example:", train[0])

