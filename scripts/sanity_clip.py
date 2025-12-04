from pathlib import Path
from src.data.preprocessing import load_video_clip

p = Path("data/raw/kth/boxing/person02_boxing_d1_uncomp.avi")
clip = load_video_clip(p, num_frames=32, size=112)
print("clip shape:", clip.shape, "dtype:", clip.dtype, "min/max:", clip.min(), clip.max())
