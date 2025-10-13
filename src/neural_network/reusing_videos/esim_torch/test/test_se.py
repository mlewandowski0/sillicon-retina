import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import imageio

from esim_torch import SiliconRetinaVerboseFloatCUDA , SiliconRetinaConfig
from esim_torch.silicon_retina_cuda import BLUR_MODE

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

    config = SiliconRetinaConfig(H=imgs.shape[1], W=imgs.shape[2],
                                 blur2_mode=BLUR_MODE.BLUR_NONE,
                                 turn_off_feedback=False)
    model = SiliconRetinaVerboseFloatCUDA(config, T=T)
    
    # Run the model
    fires_on, fires_off = model.forward(imgs)
    print(fires_on.shape)

    with imageio.get_writer('test.mp4', fps=30) as writer:
        for frame in fires_on:
            writer.append_data(255 * frame.cpu().numpy())