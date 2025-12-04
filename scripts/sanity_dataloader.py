import torch
from torch.utils.data import DataLoader
from src.data.torch_dataset import KTHClipDataset

ds = KTHClipDataset("splits/train.txt", num_frames=32, size=112)
dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

X, y = next(iter(dl))
print("X:", X.shape, X.dtype)  # expect (4, 3, 32, 112, 112)
print("y:", y.shape, y.dtype, y[:4])
