import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import imageio.v3 as iio
import numpy as np
import os
import cv2

def load_frames_from_video(video_path, scale=0.5):
    frames = []
    for frame in iio.imiter(video_path):
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return frames


class VideoDatasetFromVideoPairGlobal(Dataset):

    def __init__(self, dirpath, global_fps, split=None, input_video="input.AVI", output_video="maps.mp4", suffix = None):

        # we assume that the dataset follows structure like
        # - dataset
        #   - train
        #       - folder 1
        #       - folder 2
        #   - val
        #       - folder 1
        #       - folder 2

        if split is not None:
            self.root = os.path.join(dirpath, split)
        else:
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

        return {"images" : np.array(x), "maps": y, "fps": self.fps}
    
    
    
if __name__ == "__main__":

    idx = 1
    dataset = VideoDatasetFromVideoPairGlobal(dirpath="src/neural_network/data/saliency_dataset_processed", global_fps=30)
    x, y = dataset[idx]["images"], dataset[idx]["maps"]
    print(x.shape, len(y), dataset[idx]["fps"])