from torch.utils.data import DataLoader
from src.data.torch_skeleton_dataset import KTHOpenPoseHeatmapDataset

def main():
    ds = KTHOpenPoseHeatmapDataset("splits/test.txt", strict=True)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    X, y = next(iter(dl))
    print("X:", X.shape, X.dtype)  # (2,1,32,112,112)
    print("y:", y)

if __name__ == "__main__":
    main()
