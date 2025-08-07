import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import imageio.v3 as iio
import os

def _load_images_from_folder(folder, global_fps):
    files = sorted(os.listdir(folder))

    if "fps.txt" in files:
        fps = None
        with open("fps.txt", "r") as f:
            fps = int(f.readlines()[0])
    else:
        fps = global_fps

    images = []
    for f in files:
        if f.endswith(".jpg") or f.endswith(".png"):
            image = iio.imread(os.path.join(folder, f))
            images.append(image)

    return {"images": images, "fps": fps}



class VideoDatasetFromPictures(Dataset):

    def __init__(self, dirpath, split, global_fps, suffix = None):

        # we assume that the dataset follows structure like
        # - dataset
        #   - train
        #       - folder 1
        #       - folder 2
        #   - val
        #       - folder 1
        #       - folder 2

        self.root = os.path.join(dirpath, split)
        self.files = [os.path.join(self.root,x) for x in os.listdir(self.root)]
        self.fps = global_fps

        if suffix is None:
            self.suffix = ""
        else:
            self.suffix = suffix

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        return _load_images_from_folder(os.path.join(x, self.suffix), self.fps)


if __name__ == "__main__":
    print(os.getcwd())
    ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                  split="train",
                                  global_fps=27,
                                  suffix="frames")

    print(len(ds))
    val = ds[0]
    images, fps = val["images"], val["fps"]

    print(len(images))