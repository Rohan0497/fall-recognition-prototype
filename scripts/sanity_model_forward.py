import torch
from torch.utils.data import DataLoader
from src.data.torch_dataset import KTHClipDataset
from src.modeling.model_3dcnn import Simple3DCNN

ds = KTHClipDataset("splits/train.txt", num_frames=32, size=112)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

X, y = next(iter(dl))
model = Simple3DCNN(num_classes=6)

with torch.no_grad():
    logits = model(X)

print("X:", X.shape)
print("logits:", logits.shape)  # expect (2, 6)
print("y:", y)
