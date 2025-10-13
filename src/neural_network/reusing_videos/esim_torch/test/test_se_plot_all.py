import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import imageio

from esim_torch import SiliconRetinaVerboseFloatCUDA , SiliconRetinaConfig
from esim_torch.silicon_retina_cuda import BLUR_MODE
import esim_torch


import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
import os
from pathlib import Path

# ---------- Utils

retina_cfgs = [SiliconRetinaConfig(blur2_mode=BLUR_MODE.BLUR_NONE),
               SiliconRetinaConfig(blur2_mode=BLUR_MODE.BLUR_AVG),
               SiliconRetinaConfig(blur2_mode=BLUR_MODE.BLUR_AVG, turn_off_feedback=True)]


def bin_dvs_to_frames(x, y, p, t_ns, W, H, T, fps):
    """
    Bin asynchronous DVS events (x,y,p,t_ns) into T frames at 'fps'.
    Returns: (T, 2, H, W) float32 arrays with counts per frame (channel 0=pos, 1=neg)
    """
    frames = np.zeros((T, 2, H, W), dtype=np.float32)
    # map times to [0, T-1]
    frame_idx = np.clip((t_ns.astype(np.float64) * 1e-9 * fps).astype(np.int64), 0, T-1)
    pos_mask = (p > 0)
    neg_mask = ~pos_mask
    # accumulate
    np.add.at(frames[:,0], (frame_idx[pos_mask], y[pos_mask], x[pos_mask]), 1.0)
    np.add.at(frames[:,1], (frame_idx[neg_mask], y[neg_mask], x[neg_mask]), 1.0)
    return frames

def normalize01(a, eps=1e-8):
    a = a.astype(np.float32)
    m, M = np.min(a), np.max(a)
    if M - m < eps:
        return np.zeros_like(a)
    return (a - m) / (M - m)

def put_title(img, text):
    """Add a small title bar on top of a grayscale or 3-ch image for video frames."""
    if img.ndim == 2:
        img3 = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img3 = (img*255).astype(np.uint8)
    bar = np.ones((24, img3.shape[1], 3), dtype=np.uint8)*255
    img_out = np.vstack([bar, img3])
    cv2.putText(img_out, text, (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return img_out

def normalize01(a, eps=1e-8):
    a = a.astype(np.float32)
    m, M = np.min(a), np.max(a)
    if M - m < eps:
        return np.zeros_like(a)
    return (a - m) / (M - m)

def colorize_two_maps(pos, neg):
    """
    Make a simple dual-polarity heatmap:
      pos -> red channel (normalized)
      neg -> blue channel (normalized)
    """
    pr = normalize01(pos)
    nr = normalize01(neg)
    rgb = np.zeros((pos.shape[0], pos.shape[1], 3), dtype=np.float32)
    rgb[...,0] = pr      # R
    rgb[...,2] = nr      # B
    return np.clip(rgb, 0, 1)

def build_time_surface(events_xypt, H, W, tau_ms, t_now_ns):
    """
    events_xypt: (x,y,p,t) arrays (numpy) for a window up to t_now_ns
    Returns two time surfaces (H,W) for pos/neg.
    """
    x,y,p,t = events_xypt
    ts_pos = np.full((H,W), fill_value=-np.inf, dtype=np.float64)
    ts_neg = np.full((H,W), fill_value=-np.inf, dtype=np.float64)
    pos_mask = (p > 0)
    neg_mask = ~pos_mask
    if np.any(pos_mask):
        ts_pos[y[pos_mask], x[pos_mask]] = np.maximum(ts_pos[y[pos_mask], x[pos_mask]], t[pos_mask])
    if np.any(neg_mask):
        ts_neg[y[neg_mask], x[neg_mask]] = np.maximum(ts_neg[y[neg_mask], x[neg_mask]], t[neg_mask])
    decay = lambda last: np.exp(-np.maximum(0.0, (t_now_ns - last) * 1e-6) / tau_ms)  # ns->ms
    Spos = np.where(np.isfinite(ts_pos), decay(ts_pos), 0.0)
    Sneg = np.where(np.isfinite(ts_neg), decay(ts_neg), 0.0)
    return Spos.astype(np.float32), Sneg.astype(np.float32)

def canny_edges(img, low=50, high=150):
    """img float32 [0,1], returns binary edge map (H,W)."""
    g = (img*255).astype(np.uint8)
    e = cv2.Canny(g, low, high)
    return (e > 0).astype(np.uint8)

def precision_recall(pred, gt, eps=1e-8):
    tp = np.sum((pred==1) & (gt==1))
    fp = np.sum((pred==1) & (gt==0))
    fn = np.sum((pred==0) & (gt==1))
    prec = tp / (tp+fp+eps)
    rec  = tp / (tp+fn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    return prec, rec, f1


def visualize_all(imgs, results, dvs_events, fps=30, out_dir="viz_out"):
    
    
    dvs_out = bin_dvs_to_frames(dvs_events["x"], dvs_events["y"], dvs_events["p"], dvs_events["t"],
                                W=imgs.shape[2], H=imgs.shape[1], T=imgs.shape[0], fps=fps)

    with imageio.get_writer(os.path.join(out_dir, "all_viz.mp4"), fps=fps) as writer:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        for t in range(imgs.shape[0]):
            
            current_frame = imgs[t].cpu().numpy()
            current_frame = np.repeat(current_frame[:, :, None], 3, axis=2)
            dvs_frame = dvs_out[t]
            
            rows = []
            
            for i, sr_res in enumerate(results):
                pos_ev, neg_ev = sr_res
                assert pos_ev.shape[0] == imgs.shape[0]
                assert pos_ev.shape[1] == imgs.shape[1]
                assert pos_ev.shape[2] == imgs.shape[2]
                
                dvs_colorize = colorize_two_maps(dvs_frame[0], dvs_frame[1])
                retina_img = colorize_two_maps(pos_ev[t].cpu().numpy(), neg_ev[t].cpu().numpy())
                row = np.hstack([put_title(current_frame, "input"), put_title(retina_img, f"retina {i}"), put_title(dvs_colorize, "DVS output")])
                rows.append(row)
                
            
            
            writer.append_data(np.vstack(rows))                
                
        
if __name__ == "__main__":
    # Load the model
    T =  300
    FPS = 30
    
    root_folder = "/home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/test/frames"
    imgs = []
    frames = [os.path.join(root_folder, f) for f in sorted(os.listdir(root_folder)) if f.endswith('.png')]
    device = "cuda:0"
    
    for i in range(T):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32)
        imgs.append(img / 255.0)

    imgs = torch.stack(imgs, dim=0).to("cuda:0")

    results = []
    
    for cfg in retina_cfgs:
        
        cfg.H = imgs.shape[1]
        cfg.W = imgs.shape[2]
        model = SiliconRetinaVerboseFloatCUDA(cfg, T=T)

        with torch.no_grad():
            pos_ev, neg_ev = model(imgs)
        results.append((pos_ev, neg_ev))
    
    # add the DVS output for comparison
    esim = esim_torch.ESIM(contrast_threshold_neg=0.2,
                                   contrast_threshold_pos=0.2,
                           refractory_period_ns=0)
    
    timestamps_s = np.arange(0, len(imgs)) * 1.0/ FPS
    timestamps_ns = (timestamps_s * 1e9).astype("int64")

    log_images = torch.log(imgs + 1e-4)
    timestamps_ns = torch.from_numpy(timestamps_ns).to(device)
    dvs_out = esim.forward(log_images, timestamps_ns)
    dvs_out["x"] = dvs_out["x"].cpu().numpy()
    dvs_out["y"] = dvs_out["y"].cpu().numpy()
    dvs_out["p"] = dvs_out["p"].cpu().numpy()
    dvs_out["t"] = dvs_out["t"].cpu().numpy()
                
    visualize_all(imgs=imgs, 
                  results=results, 
                  dvs_events=dvs_out,
                  fps=30,
                  out_dir="viz_out")