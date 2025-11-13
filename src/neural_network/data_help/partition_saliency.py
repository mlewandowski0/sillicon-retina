import os 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import json
from tqdm import tqdm

VAL_FRACTION = 0.2
RNG_SEED = 49210

def get_frame_count(path):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise IOError(f"Could not open the video file: {path}")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return num_frames

def copy_split(folders, indices, split_root):
    for i in tqdm(indices):
        src = Path(folders[i])
        dst = Path(split_root) / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        

def symlink_split(folders, indices, split_root):
    for i in indices:
        src = Path(folders[i])
        dst = Path(split_root) / src.name
        if dst.exists():
            dst.unlink()  # or rmtree if it's a directory; be careful!
        os.symlink(src, dst, target_is_directory=True)


data_root = os.path.join("src", "neural_network", "data")
output_root = os.path.join(data_root, "saliency_dataset_ready")
dataset_root_path = os.path.join(data_root, "saliency_dataset_upsampled")

output_root_train = os.path.join(output_root, "train")
output_root_val   = os.path.join(output_root, "val")

os.makedirs(output_root_train, exist_ok=True)
os.makedirs(output_root_val,   exist_ok=True)

folders = sorted(
    [os.path.join(dataset_root_path, el)
     for el in os.listdir(dataset_root_path)
     if os.path.isdir(os.path.join(dataset_root_path, el))]
)

lengths = []
for folder in tqdm(folders):
    video_path = os.path.join(folder, "video_upsampled_fps_eq_200.mp4")
    no_frames = get_frame_count(video_path)
    lengths.append(no_frames)

lengths = np.array(lengths)
total_frames = int(lengths.sum())
print(f"Total no frames: {total_frames}")

# Optional: global histogram
plt.hist(lengths, bins=40, edgecolor="black")
plt.title("Histogram of video lengths (in frames)")
plt.xlabel("No frames")
plt.ylabel("No videos")
plt.savefig("video_lengths_histogram.png")
plt.close()


# stratification 
rng = np.random.default_rng(RNG_SEED)

# use length percentiles to define bins (5 bins here)
percentiles = np.percentile(lengths, [0, 20, 40, 60, 80, 100])

# digitize assigns each video to a bin 0..4 based on its length 
# bins passed to np.digitize should be the inner edges (exclude min, max)
bin_edges = percentiles[1:-1]
bin_ids = np.digitize(lengths, bin_edges, right=True)

train_indices = []
val_indices = []

for b in range(bin_ids.max() + 1):
    idx_in_bin = np.where(bin_ids == b)[0]
    
    if len(idx_in_bin) == 0:
        continue
    
    rng.shuffle(idx_in_bin)
    split_point = int(len(idx_in_bin) * (1.0 - VAL_FRACTION))
    train_indices.extend(idx_in_bin[:split_point])
    val_indices.extend(idx_in_bin[split_point:])
    
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)

# shuffle final lists for good measure
rng.shuffle(train_indices)
rng.shuffle(val_indices)

print(f"Train videos: {len(train_indices)}, Val videos: {len(val_indices)}")
print(f"Train frames: {int(lengths[train_indices].sum())}, "
      f"Val frames: {int(lengths[val_indices].sum())}")
print(f"Val fraction (videos): {len(val_indices) / len(lengths):.3f}")
print(f"Val fraction (frames): {lengths[val_indices].sum() / lengths.sum():.3f}")

plt.hist(lengths[train_indices], bins=40, edgecolor="black", alpha=0.5, label="train")
plt.hist(lengths[val_indices],   bins=40, edgecolor="black", alpha=0.5, label="val")
plt.title("Length distribution: train vs val")
plt.xlabel("No frames")
plt.ylabel("No videos")
plt.legend()
plt.savefig("video_lengths_histogram_train_val.png")
plt.close()

copy_split(folders, train_indices, output_root_train)
copy_split(folders, val_indices,   output_root_val)

split_info = {
    "train": [Path(folders[i]).name for i in train_indices],
    "val":   [Path(folders[i]).name for i in val_indices],
}

with open(os.path.join(output_root, "split.json"), "w") as f:
    json.dump(split_info, f, indent=2)