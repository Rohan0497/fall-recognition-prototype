from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import read_split_file
from src.data.torch_skeleton_dataset import KTHOpenPoseHeatmapDataset, SkeletonHeatmapConfig


def count_available_skeletons(split_file: str, skeleton_root: Path) -> tuple[int, int]:
    entries = read_split_file(Path(split_file))
    total = len(entries)
    found = 0
    for e in entries:
        cls = e.video_path.parent.name
        stem = e.video_path.stem
        npy_path = skeleton_root / cls / f"{stem}.npy"
        if npy_path.exists():
            found += 1
    return found, total

# -------------------------
# Model (lightweight 3D CNN)
# -------------------------
class Simple3DCNN(nn.Module):
    """
    Lightweight 3D CNN that works well on (B, C=1, T=32, H=112, W=112) for 4GB GPU.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),  # keep time, downsample spatial

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # downsample time + spatial

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for b, (X, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    acc = correct / total if total else 0.0
    preds_np = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    targets_np = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    return acc, preds_np, targets_np


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix_png(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Skeleton Heatmaps)")

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_accuracy_plot(train_acc: list[float], val_acc: list[float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(train_acc) + 1), train_acc, label="train")
    ax.plot(range(1, len(val_acc) + 1), val_acc, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs Epoch (Skeleton Heatmaps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# Train
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # speed / smoke-run controls
    parser.add_argument("--max_train_batches", type=int, default=0, help="0=all, else cap batches/epoch")
    parser.add_argument("--max_val_batches", type=int, default=0, help="0=all, else cap val batches")
    parser.add_argument("--max_test_batches", type=int, default=0, help="0=all, else cap test batches")

    # where skeletons are
    parser.add_argument("--skeleton_root", type=str, default="data/processed/openpose_skeletons_coco_r096")
    parser.add_argument("--strict_train", action="store_true", help="If set, error if train skeleton missing. Default is non-strict.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset config
    cfg = SkeletonHeatmapConfig(
        skeleton_root=Path(args.skeleton_root),
        num_frames=32,
        size=112,
        sigma=2.0,
        conf_thresh=0.05,
    )
    for split_name, split_path in [("train", "splits/train.txt"), ("val", "splits/val.txt"), ("test", "splits/test.txt")]:
        found, total = count_available_skeletons(split_path, cfg.skeleton_root)
        pct = 100.0 * found / total if total else 0.0
        print(f"[skeletons] {split_name}: found {found}/{total} ({pct:.1f}%) in {cfg.skeleton_root}")

    # IMPORTANT:
    # - train can be non-strict to allow partial extraction
    # - val/test should be strict (so metrics are meaningful)
    train_ds = KTHOpenPoseHeatmapDataset("splits/train.txt", cfg=cfg, strict=args.strict_train)
    val_ds = KTHOpenPoseHeatmapDataset("splits/val.txt", cfg=cfg, strict=True)
    test_ds = KTHOpenPoseHeatmapDataset("splits/test.txt", cfg=cfg, strict=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = 6
    class_names = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

    model = Simple3DCNN(in_channels=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Outputs
    out_logs = Path("outputs/logs")
    out_figs = Path("outputs/figures")
    ckpt_dir = Path("checkpoints")
    out_logs.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_logs / "train_skeleton_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    best_val = -1.0
    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for b, (X, y) in enumerate(pbar):
            if args.max_train_batches and b >= args.max_train_batches:
                break

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = epoch_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        val_acc, _, _ = evaluate(model, val_loader, device, max_batches=args.max_val_batches)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_acc])

        if val_acc > best_val:
            best_val = val_acc
            ckpt_path = ckpt_dir / "best_skeleton.pt"
            torch.save({"model_state": model.state_dict(), "val_acc": best_val, "epoch": epoch}, ckpt_path)

    # plots
    save_accuracy_plot(train_acc_hist, val_acc_hist, out_figs / "accuracy_vs_epoch_skeleton.png")

    # test (once)
    ckpt_path = Path("checkpoints/best_skeleton.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print("Loaded best_skeleton.pt (epoch:", ckpt.get("epoch"), "val_acc:", ckpt.get("val_acc"), ")")

    test_acc, preds, targets = evaluate(model, test_loader, device, max_batches=args.max_test_batches)
    print("Test accuracy:", test_acc)

    cm = confusion_matrix(preds, targets, num_classes=num_classes)
    save_confusion_matrix_png(cm, class_names, out_figs / "confusion_matrix_skeleton.png")
    print("Saved confusion matrix:", out_figs / "confusion_matrix_skeleton.png")
    print("Best val acc:", best_val)


if __name__ == "__main__":
    main()
