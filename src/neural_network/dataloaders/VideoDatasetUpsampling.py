import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from .VideoDataset import VideoDatasetFromPictures


class VideoDatasetUpsampling(Dataset):

    def __init__(self,
                 videoDataset):

        super(VideoDatasetUpsampling, self).__init__()
        self.videoDataset = videoDataset


    def _process_single_image(self, x):


    def 


if __name__ == "__main__":

    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds)
