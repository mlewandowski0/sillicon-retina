import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import imageio
import math

from esim_torch import SiliconRetinaVerboseFloatCUDA , SiliconRetinaConfig
from src.neural_network.utils.event_utils import events_to_voxels, render_events_video


def bitmaps_to_voxel_grid(fired_on, fired_off, ts_ns):
    
    T, H, W = fired_on.shape 
    HW = H * W
    # linearize once 
    on_lin = fired_on.view(T, -1).nonzero(as_tuple=False)
    off_lin = fired_off.view(T, -1).nonzero(as_tuple=False)

    def unpack(t_idx):
        t  = ts_ns.index_select(0, t_idx[:,0])
        i  = t_idx[:,1]
        y  = (i // W).to(torch.int32)
        x  = (i %  W).to(torch.int32)
        return x, y, t

    x_on,  y_on,  t_on  = unpack(on_lin)
    x_off, y_off, t_off = unpack(off_lin)

    p_on  = torch.ones_like(x_on,  dtype=torch.int8)
    p_off = -torch.ones_like(x_off, dtype=torch.int8)

    x = torch.cat([x_on,  x_off],  dim=0)
    y = torch.cat([y_on,  y_off],  dim=0)
    t = torch.cat([t_on,  t_off],  dim=0)
    p = torch.cat([p_on,  p_off],  dim=0)

    order = torch.argsort(t, stable=True)
    return x[order], y[order], t[order], p[order]


if __name__ == "__main__":
    # Load the model
    T =  300
    fps = 30 
    per_frame = 5
    
    imgs = []
    
    cur_dir = os.path.join("src", "neural_network", "prototyping")
    root_path = os.path.join(cur_dir, "frames")
    frames = [os.path.join(root_path, f) for f in sorted(os.listdir(root_path)) if f.endswith('.png')]
    
    for i in range(T):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32)
        imgs.append(img / 255.0)

    imgs = torch.stack(imgs, dim=0).to("cuda:0")
    
    T, H, W = imgs.shape

    config = SiliconRetinaConfig()
    model = SiliconRetinaVerboseFloatCUDA(config, height=imgs.shape[1], width=imgs.shape[2], T=T)
    
    # Run the model
    fires_on, fires_off = model.forward(imgs)  # expect bool-like masks per frame

    # Convert to numpy bool arrays with shape (T, H, W)
    def to_bool_numpy(x):
        # x can be: torch.Tensor [T,H,W], list of tensors [H,W], or numpy
        if isinstance(x, (list, tuple)):
            x = [xi.detach().cpu().numpy() if hasattr(xi, "detach") else np.asarray(xi) for xi in x]
            x = np.stack(x, axis=0)
        else:
            x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return x.astype(bool)

    ts = torch.arange(0, T).cpu()
    x,y,t, p = bitmaps_to_voxel_grid(fired_on=fires_on.cpu(), fired_off=fires_off.cpu(), ts_ns=ts)

    t = t / (fps * per_frame)
   
    video_time = T / (fps * per_frame) 
    linspaced = np.linspace(0, video_time, num=math.ceil(video_time * fps), endpoint=True)
    
    raw_frames = imgs[::per_frame].cpu().numpy()[:-1]

    t_bounds = []
    for i in range(len(linspaced) - 1):
        t_bounds.append([linspaced[i], linspaced[i + 1]])
    
    
    t_bounds = torch.Tensor(t_bounds)
    boundaries = torch.searchsorted(t, t_bounds, right=False)  # start indices

    grids = []

    for k in range(len(boundaries)):
        s, e = boundaries[k]
        s = int(s.item())
        e = int(e.item())

        # skip degenerate spans (no events or zero/negative duration)        
        t0, t1 = t_bounds[k]
        t0 = t0.item()
        t1 = t1.item()
        if e <= s or t1 <= t0:
            continue

        xk, yk, tk, pk = x[s:e], y[s:e], t[s:e], p[s:e]

        # (optional) hard safety: clamp x/y once if your ESIM can emit border values
        # H, W = frames.shape[-2], frames.shape[-1]
        # xk = xk.clamp_(0, W-1); yk = yk.clamp_(0, H-1)

        g = events_to_voxels(xk, yk, tk, pk, t0, t1, device=x.device, H=H, W=W)
        grids.append(g)

    event_grids = torch.stack(grids, dim=0)
    
    render_events_video({"frames" : raw_frames,
                         "grids" : event_grids},
                        output_path = os.path.join(cur_dir, "event_vis.mp4"),
                        flows_on=False)
    
    fires_on = to_bool_numpy(fires_on)
    fires_off = to_bool_numpy(fires_off)

    # Sanity: shapes must match
    assert fires_on.shape == fires_off.shape, f"Shape mismatch: {fires_on.shape} vs {fires_off.shape}"

    T, H, W = fires_on.shape

    # Allocate RGB frames
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)

    # Color ON events in red channel, OFF in blue channel
    # Use masked assignment on the specific channels
    frames[..., 0][fires_on] = 255   # R channel for ON
    frames[..., 2][fires_off] = 255  # B channel for OFF

    # Write the video
    with imageio.get_writer(os.path.join(cur_dir, "test.mp4"), fps=30) as writer:
        for f in frames:
            writer.append_data(f)   # f is already a uint8 numpy array