import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import imageio

from esim_torch import SiliconRetinaVerboseFloatCUDA , SiliconRetinaConfig


if __name__ == "__main__":
    # Load the model
    T =  300
    
    imgs = []
    frames = [os.path.join("frames", f) for f in sorted(os.listdir("frames")) if f.endswith('.png')]
    
    for i in range(T):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32)
        imgs.append(img / 255.0)

    imgs = torch.stack(imgs, dim=0).to("cuda:0")

    config = SiliconRetinaConfig()
    model = SiliconRetinaVerboseFloatCUDA(config, width=imgs.shape[1], height=imgs.shape[2], T=T)
    
    # Run the model
    fires = model.forward(imgs)


    with imageio.get_writer('test.mp4', fps=30) as writer:
        for frame in fires:
            writer.append_data(255 * frame.cpu().numpy())