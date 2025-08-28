import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
import esim_torch
import matplotlib
from src.neural_network.utils.event_utils import *
import hashlib
from src.neural_network.dataloaders.utils import _mk_cache_paths_SE, _memmap_load, _save_cached_npy
from esim_torch.silicon_retina_torch import SiliconRetinaConfig,SiliconRetinaFloatCUDAOnline


"""
        METADATA LOOKS LIKE 
        
        
        {
            "raw_images": "raw_images.npz",
            "fps": 27,
            "optical_flow_path": "optical_flow.npz",
            "new_images_per_pair_FF": [....]
            "timestamps_FF": [....]
            "new_images_per_pair_OF": [....]
            "timestamps_OF": [....]
        }
"""

def bitmaps_to_voxel_grid(fired_on, fired_off, ts_ns):
    
    T, H, W = fired_on.shape 
    HW = H * W
    # linearize once 
    on_lin = fired_on.view(T, -1).nonzero(as_tuple=False).cpu()
    off_lin = fired_off.view(T, -1).nonzero(as_tuple=False).cpu()

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

# Convert to numpy bool arrays with shape (T, H, W)
def to_bool_numpy(x):
    # x can be: torch.Tensor [T,H,W], list of tensors [H,W], or numpy
    if isinstance(x, (list, tuple)):
        x = [xi.detach().cpu().numpy() if hasattr(xi, "detach") else np.asarray(xi) for xi in x]
        x = np.stack(x, axis=0)
    else:
        x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return x.astype(bool)

class SiliconRetinaEventDatasetFromUpsampled(Dataset):

    def __init__(self, directory,
                       B = 5,
                       height=180,
                       width=240,
                       config = SiliconRetinaConfig(),
                       force_recomputing=False,
                       input_framerate=200,
                       log_scale=False,
                       device="cuda",
                        limit=-1,
                       grids_dtype=np.float16):


        self.directory = directory
        self.B = B
        self.height = height
        self.width = width
        self.force_recomputing = force_recomputing
        self.device = device
        self.grids_dtype = grids_dtype
        self.config = config
        self.input_framerate = input_framerate
        self.log_scale = log_scale

        # load glboal metadata once
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            self.read_metadata = json.load(f)

        # enumerate samples
        self.data_files = sorted([os.path.join(directory, el) for el in os.listdir(directory) if os.path.isdir(os.path.join(directory, el))])

        if limit is not None and limit > 0:
            self.data_files = self.data_files[:limit]

        # Pre-read per-sample params once (avoid JSON I/O in __getitem__)
        self.properties = []
        for folder in self.data_files:
            with open(os.path.join(folder, "params.json"), "r") as f:
                self.properties.append(json.load(f))

        # ESIM
        self.simulator = None


    def __getstate__(self):
        state = self.__dict__.copy()
        state['simulator'] = None
        return state

    def _ensure_simulator(self):
        if self.simulator is None:
            self.simulator = SiliconRetinaFloatCUDAOnline(silicon_retina_config=self.config,
                                 height=self.height,
                                 width=self.width)



    def __len__(self):
        return len(self.data_files)

    def _mk_paths(self, folder):
        return _mk_cache_paths_SE(folder, self.B)

    @staticmethod
    def _load_video_gray(path):
        # your original load, but explicit return dtype
        import imageio
        vid = imageio.v3.imread(path)  # (N,H,W,C) or (N,H,W)
        if vid.ndim == 4:
            vid = vid[..., 0]
        return vid.astype(np.uint8)

    def _read_events_or_create(self, folder, properties):

        paths = self._mk_paths(folder)

        have_cache = (os.path.exists(paths['grids'])
                      and os.path.exists(paths['frames'])
                      and os.path.exists(paths['x'])
                      and os.path.exists(paths['y'])
                      and os.path.exists(paths['t'])
                      and os.path.exists(paths['p']))

        if have_cache and not self.force_recomputing:
            # fast path: mmapped reads

            # (T, B, H, W), dtype=np.float16
            grids_np = _memmap_load(paths['grids'])

            # (T+1, H, W) uint8
            frames_np = _memmap_load(paths['frames'])

            x_np = _memmap_load(paths["x"])  # uint16
            y_np = _memmap_load(paths["y"])  # uint16
            t_np = _memmap_load(paths["t"])  # int64
            p_np = _memmap_load(paths["p"])  # int8


            # zero-copy views into tensors on CPU: later moved to CUDA
            grids_np = _memmap_load(paths['grids'])
            event_grids = torch.from_numpy(grids_np)
            event_grids = event_grids.clone()

            events = {
                "x": torch.as_tensor(x_np),
                "y": torch.as_tensor(y_np),
                "t": torch.as_tensor(t_np),
                "p": torch.as_tensor(p_np),
            }
            result = {
                "frames": frames_np['raw_images'],  # keep as numpy (cheap, contiguous)
                "events": events,
                "grids": event_grids,  # half precision on load
            }

            # optional: optical flow (unchanged from your logic)
            optical_flow_file = os.path.join(folder, "optical_flow.npz")
            if os.path.exists(optical_flow_file):
                optical_flow = np.load(optical_flow_file)["optical_flow"]
                result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

            return result

        # ensure simulator
        self._ensure_simulator()

        # slow path ( first time read and data cacheing)
        frames = self._load_video_gray(os.path.join(folder, f"video_upsampled_fps_eq_{self.input_framerate}.mp4"))  # (N, H, W), uint8
        timestamps = np.asarray(properties["timestamps_FF"], dtype=np.float64)
        images = properties["new_images_per_pair_FF"]  # list of counts per segment
        timestamps_ns = (timestamps * 1e9).astype("int64")

        # Log intensities on GPU
        if self.log_scale:
            frames = np.log(frames.astype("float32") + 1e-3) / np.log(255. + 1e-3)
            frames = frames.astype("float32")
        else:
            frames = frames.astype("float32") / 255.0 + 1e-4
        frames = torch.from_numpy(frames).to(self.device, non_blocking=True)
        timestamps_ns_t = torch.from_numpy(timestamps_ns).cpu()

        # load the raw_frames
        raw_frames = np.load(os.path.join(folder, "raw_images.npz"))['raw_images'] / 255.

        # reset + simulate events
        self.simulator.reset()

        positive, negative = [], []
        T, H, W = frames.shape
        
        self.simulator.reset()
        for t in range(1, T+1, 1):

            input = frames[t-1][None]
            p, n = self.simulator(input, t)
            positive.append((p.clone()[0]))
            negative.append((n.clone()[0]))

        positive = torch.stack(positive, dim=0)
        negative = torch.stack(negative, dim=0)
        
        x,y,t, p = bitmaps_to_voxel_grid(fired_on=positive.cpu(), fired_off=negative.cpu(), ts_ns=timestamps_ns_t)
    
        raw_frames_fps = properties["fps"]
        raw_frames_dt = 1 / raw_frames_fps
        t_bounds = []
        
        
        for i in range(len(raw_frames) - 1):
            t_bounds.append([i * raw_frames_dt * 1e9, (i+1) * raw_frames_dt * 1e9])
        
        
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
                grids.append(torch.zeros((self.B, H, W), dtype=torch.float16, device=x.device))
                continue

            xk, yk, tk, pk = x[s:e], y[s:e], t[s:e], p[s:e]

            # (optional) hard safety: clamp x/y once if your ESIM can emit border values
            # H, W = frames.shape[-2], frames.shape[-1]
            # xk = xk.clamp_(0, W-1); yk = yk.clamp_(0, H-1)

            g = events_to_voxels(xk, yk, tk, pk, t0, t1, device=x.device, H=H, W=W)
            grids.append(g)

        event_grids = torch.stack(grids, dim=0)

        # write caches
        # cast & move to CPU arrays only once, right before saving
        _save_cached_npy(paths["frames"], raw_frames.astype(np.uint8))
        _save_cached_npy(paths["grids"], event_grids.detach().to("cpu").to(torch.float16).numpy())

        # Compact events to smaller dtypes on disk
        _save_cached_npy(paths["x"], x.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["y"], y.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["t"], t.detach().to("cpu").to(torch.int64).numpy())
        _save_cached_npy(paths["p"], p.detach().to("cpu").to(torch.int8).numpy())

        # Return in the same shapes/types you expect
        result = {
            "frames": raw_frames,                                 # np.uint8
            "events": {
                "x": x.to("cpu"),
                "y": y.to("cpu"),
                "t": t.to("cpu"),
                "p": p.to("cpu"),
            },
            "grids": event_grids.to("cpu"),                       # torch.float32 -> we saved float16
        }

        optical_flow_file = os.path.join(folder, "optical_flow.npz")
        if os.path.exists(optical_flow_file):
            optical_flow = np.load(optical_flow_file)["optical_flow"]
            result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

        return result




    def __getitem__(self, idx):
        folder = self.data_files[idx]
        props = self.properties[idx]

        return self._read_events_or_create(folder, props)



    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    import time

    cfg = SiliconRetinaConfig()
    cfg.alpha1 = 0.125
    cfg.alpha2 = 0.125

    ds = SiliconRetinaEventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "upsampled_validation"),
                                   B=5,
                                   config=cfg,
                                   force_recomputing=True)


    datapoint = ds[0]

    imgs, events, grids = datapoint["frames"], datapoint["events"], datapoint["grids"]
    optical_flow = datapoint["optical_flows"]


    x,y, t, p = events['x'], events['y'], events['t'], events['p']

    # torch.Size([55, 180, 240])
    print(imgs.shape)

    # torch.Size([54, 5, 240, 180])
    print(grids.shape)

    # visualize a datapoint
    render_events_video(datapoint, output_path="events_vis_retina_1.mp4")
