import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import imageio

from esim_torch import SiliconRetinaFloatCUDAOnline , SiliconRetinaConfig

def make_vis(pos, neg):
    pos = pos.detach().cpu().numpy()
    neg = neg.detach().cpu().numpy()
    ret = np.zeros_like(pos)
    ret = np.repeat(ret[..., None], axis=-1, repeats=3)
    
    ret[..., 0] = pos * 255
    ret[..., 2] = neg  * 255
    
    return ret.astype(np.uint8)
    

if __name__ == "__main__":
    # Load the model
    T =  300
    
    imgs = []
    root = os.path.join("src", "neural_network", "prototyping")
    path = os.path.join(root, "frames")
    frames = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.png')]

    for i in range(T):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32)
        imgs.append(img / 255.0)

    imgs = torch.stack(imgs, dim=0).to("cuda:0")

    config = SiliconRetinaConfig()

    config.HPF2_mode = 1
    config.HPF2_param = 0.5

    model = SiliconRetinaFloatCUDAOnline(config, height=imgs.shape[1], width=imgs.shape[2])
    
    positive, negative = [], []
    
    model.reset()
    for t in range(1, T+1, 1):

        input = imgs[t-1][None]
        p, n = model(input, t)
        positive.append((p.clone()[0]))
        negative.append((n.clone()[0]))

    positive = torch.stack(positive, dim=0)
    negative = torch.stack(negative, dim=0)

    with imageio.get_writer(os.path.join(root,'test.mp4'), fps=4) as writer:
        for i in range(len(positive)):
            frame = make_vis(positive[i], negative[i])
            writer.append_data(frame)