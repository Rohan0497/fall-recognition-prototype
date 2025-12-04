"""Evaluation entrypoint: confusion matrix + metrics on test set."""

from __future__ import annotations

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import KTHClipDataset
from src.modeling.model_3dcnn import Simple3DCNN


def load_class_names(label_map_path: Path, num_classes: int) -> list[str]:
    """
    Expects splits/label_map.json like: {"boxing":0, "walking":5, ...}
    Returns class names ordered by class id.
    """
    if not label_map_path.exists():
        return [str(i) for i in range(num_classes)]

    mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_name = {int(v): str(k) for k, v in mapping.items()}
    return [id_to_name.get(i, str(i)) for i in range(num_classes)]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 6))
    num_frames = int(ckpt.get("num_frames", 32))
    size = int(ckpt.get("size", 112))

    # Data
    test_ds = KTHClipDataset("splits/test.txt", num_frames=num_frames, size=size)
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,  # Windows-safe
        pin_memory=(device == "cuda"),
    )

    # Model
    model = Simple3DCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().tolist()

            y_true.extend(y.tolist())
            y_pred.extend(preds)

    # Metrics + Confusion Matrix
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    class_names = load_class_names(Path("splits/label_map.json"), num_classes)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "confusion_matrix.png"

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (Test Acc: {acc:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_names)

    # annotate counts
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(" Test accuracy:", acc)
    print(" Saved confusion matrix:", fig_path)


if __name__ == "__main__":
    main()

