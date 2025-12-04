"""Training entrypoint: trains model and saves checkpoints + accuracy curve."""

from __future__ import annotations
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.torch_dataset import KTHClipDataset
from src.modeling.model_3dcnn import Simple3DCNN
import matplotlib.pyplot as plt

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer=None, device="cpu",desc: str = "epoch",) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        if train_mode:
            loss.backward()
            optimizer.step()
        batch_acc = accuracy(logits.detach(), y)
        total_loss += float(loss.item())
        total_acc += float(batch_acc)
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def main():
    # ---- config ----
    num_classes = 6
    epochs = 15
    batch_size = 8
    lr = 1e-3
    num_frames = 32
    size = 112
    num_workers = 0  # keep 0 on Windows for reliability

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- paths ----
    outputs_fig = Path("outputs/figures")
    outputs_log = Path("outputs/logs")
    ckpt_dir = Path("checkpoints")
    outputs_fig.mkdir(parents=True, exist_ok=True)
    outputs_log.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_ds = KTHClipDataset("splits/train.txt", num_frames=num_frames, size=size)
    val_ds = KTHClipDataset("splits/val.txt", num_frames=num_frames, size=size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- model ----
    model = Simple3DCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_path = outputs_log / "train_log.csv"
    rows = []
    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device=device, desc=f"train {epoch}/{epochs}")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device, desc=f"val   {epoch}/{epochs}")

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "num_frames": num_frames,
                "size": size,
            }, best_path)

    # write csv log
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # plot accuracy vs epoch
   

    epochs_list = [r["epoch"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]

    plt.figure()
    plt.plot(epochs_list, train_acc, label="train_acc")
    plt.plot(epochs_list, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    fig_path = outputs_fig / "accuracy_vs_epoch.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(" Saved:")
    print(" -", log_path)
    print(" -", fig_path)
    print(" -", best_path)
    print("Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()

