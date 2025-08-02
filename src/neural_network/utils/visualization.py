import os 
import numpy as np 
import cv2 
import time 
import imageio 
import matplotlib.pyplot as plt
import matplotlib

def get_paths_from_folder(folder):
    paths = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".png") or f.endswith(".npy"):
            paths.append(os.path.join(folder, f))
    return paths

def get_imgs_from_folder(folder):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".png"):
            im = imageio.imread(os.path.join(folder, f))
            imgs.append(im[:,:])
        elif f.endswith(".npy"):
            im = np.load(os.path.join(folder, f))
            imgs.append(im[:,:])
    return np.array(imgs)

def render_as_video(imgs, output_path="video.mp4"):
    with imageio.get_writer(output_path,
                            format="FFMPEG",
                            fps=5) as vid:
        for im in imgs:
            vid.append_data(im)

def flow_to_rgb(flow):
    # convert the optical flow in shape 2 x H x W to RGB 
    _, W, H = flow.shape
    hsv = np.zeros((W, H, 3), dtype=np.float32)
    
    X = flow[0, ...]
    Y = flow[1, ...]
    
    mag_sq = X**2 + Y**2
    max = np.max(mag_sq) 
    angle = np.atan2(Y, X)
    
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = np.sqrt(mag_sq) / np.sqrt(max) if max > 0 else 0.0
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255.0).astype(np.uint8)

def render_as_video_stich(datapoint, output_path="video_stiched.mp4"):
    
    flow_folder, rgb_folder, of_folder = datapoint
    print(rgb_folder, flow_folder, of_folder)
    
    rgbs, flow_rgbs, events = [], [], []
    rgbs = get_imgs_from_folder(rgb_folder)
    flows = get_imgs_from_folder(flow_folder)
    print(flows.shape)
    
    flow_rgbs = [flow_to_rgb(el) for el in flows]
    events = get_imgs_from_folder(of_folder)
    
    i = 0    
    with imageio.get_writer(output_path,
                            format="FFMPEG",
                            fps=5) as vid:
        for ev_vox in events: 
            for evs in ev_vox:
                evs = (255 * (evs - np.min(evs)) / (np.max(evs) - np.min(evs))).astype(np.uint8) 
                evs = np.repeat(evs[..., None], 3, axis=2)
                rgb = rgbs[i][..., None]
                rgb = np.repeat(rgb, 3, axis=2)
                vid.append_data(np.concatenate((rgb, evs, flow_rgbs[i]), axis=1))
            i += 1
