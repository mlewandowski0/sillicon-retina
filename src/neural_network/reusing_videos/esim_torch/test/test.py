import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

import esim_torch


if __name__ == "__main__":
    esim = esim_torch.ESIM(contrast_threshold_neg=0.2,
                                   contrast_threshold_pos=0.2,
                           refractory_period_ns=0)
    
  
    FPS = 30
    print("Loading images")
    image_files = sorted(glob.glob("frames/*.png"))
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
    timestamps_s = np.arange(0, len(images)) * 1.0/ FPS

    timestamps_ns = (timestamps_s * 1e9).astype("int64")
    log_images = np.log(images.astype("float32") / 255 + 1e-4)

    # generate torch tensors
    print("Loading data to GPU")
    device = "cuda:0"
    log_images = torch.from_numpy(log_images).to(device)
    timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

    # generate events with GPU support
    print("Generating events")

    print(log_images)
    print(timestamps_ns)
    events = esim.forward(log_images, timestamps_ns)

    print(events)