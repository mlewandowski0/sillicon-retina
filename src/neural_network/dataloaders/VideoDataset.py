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

def load_frames_from_video(video_path):
    frames = []
    for frame in iio.imiter(video_path):
        frames.append(frame)
    return frames

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
        self.files = sorted([os.path.join(self.root,x) for x in os.listdir(self.root)])
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

class VideoDatasetFromVideoPairGlobal(Dataset):

    def __init__(self, dirpath, global_fps, input_video="input.AVI", output_video="maps.mp4", suffix = None):

        # we assume that the dataset follows structure like
        # - dataset
        #   - train
        #       - folder 1
        #       - folder 2
        #   - val
        #       - folder 1
        #       - folder 2

        self.root = dirpath
        self.files = sorted([os.path.join(self.root, x) for x in os.listdir(self.root)])
        self.fps = global_fps

        if suffix is None:
            self.suffix = ""
        else:
            self.suffix = suffix

        self.input_video = input_video
        self.output_video = output_video

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        x = os.path.join(path, self.input_video)
        y = os.path.join(path, self.output_video)
        
        x = load_frames_from_video(x)
        y = load_frames_from_video(y)

        return {"images" : x, "maps": y, "fps": self.fps}


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