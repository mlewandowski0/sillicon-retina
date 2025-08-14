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
from .utils import _mk_cache_paths, _memmap_load, _save_cached_npy


class EventDatasetFromUpsampled(Dataset):


    def __init__(self, directory,
                       mode = "event",
                       B = 5,
                       contrast_threshold_neg = 0.2,
                       contrast_threshold_pos = 0.2,
                       force_recomputing=False,
                       device="cuda",
                        limit=-1,
                       grids_dtype=np.float16):

        self.directory = directory
        self.mode = mode
        self.B = B
        self.force_recomputing = force_recomputing
        self.device = device
        self.grids_dtype = grids_dtype

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
        self.event_simulator = esim_torch.ESIM(
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            refractory_period_ns=0,
        )

        # Cache filename builder bound with thresholds
        self._mk_paths = lambda folder: _mk_cache_paths(
            folder, B, contrast_threshold_neg, contrast_threshold_pos
        )

    def __len__(self):
        return len(self.data_files)


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
            event_grids = torch.as_tensor(grids_np)
            events = {
                "x": torch.as_tensor(x_np),
                "y": torch.as_tensor(y_np),
                "t": torch.as_tensor(t_np),
                "p": torch.as_tensor(p_np),
            }
            result = {
                "frames": frames_np,  # keep as numpy (cheap, contiguous)
                "events": events,
                "grids": event_grids,  # half precision on load
            }

            # optional: optical flow (unchanged from your logic)
            optical_flow_file = os.path.join(folder, "optical_flow.npz")
            if os.path.exists(optical_flow_file):
                optical_flow = np.load(optical_flow_file)["optical_flow"]
                result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

            return result


        # slow path ( first time read and data cacheing)
        frames = self._load_video_gray(os.path.join(folder, "video.mp4"))  # (N, H, W), uint8
        timestamps = np.asarray(properties["timestamps"], dtype=np.float64)
        images = properties["images"]  # list of counts per segment
        timestamps_ns = (timestamps * 1e9).astype("int64")

        # Log intensities on GPU
        log_images = np.log(frames.astype("float32") / 255.0 + 1e-4)
        log_images = torch.from_numpy(log_images).to(self.device, non_blocking=True)
        timestamps_ns_t = torch.from_numpy(timestamps_ns).to(self.device, non_blocking=True)

        # reset + simulate events
        self.event_simulator.reset()
        events = self.event_simulator.forward(log_images, timestamps_ns_t)
        x, y, t, p = events["x"], events["y"], events["t"], events["p"]  # GPU tensors

        # images: list of counts per segment (as in your original code)
        imgs_np = np.asarray(images, dtype=np.int64)

        # cumulative frame counts; cum_full has length len(images)
        cum_full = np.cumsum(imgs_np, dtype=np.int64)  # [im0, im0+im1, ..., sum(images)]
        # boundaries used by your original loop: [0, im0, im0+im1, ..., im0+...+im_{K-2}]
        frame_idxs = np.concatenate(([0], cum_full[:-1]))  # len = len(images)

        # gather timestamp bounds at those frame indices -> contiguous int64, 1-D
        idxs = torch.from_numpy(frame_idxs).to(timestamps_ns_t.device)
        t_bounds = timestamps_ns_t.index_select(0, idxs).contiguous()

        # sanity: ensure monotonic nondecreasing; if not, fix or skip
        if not torch.all(t_bounds[1:] >= t_bounds[:-1]):
            # if the dataset can contain non-monotonic timestamps, sort them (rare)
            perm = torch.argsort(t_bounds)
            t_bounds = t_bounds[perm]

        # ensure event timestamps are 1-D, int64 and sorted
        assert t.dtype == torch.int64 and t.dim() == 1
        # (ESIM outputs sorted t; if unsure: t, order = torch.sort(t); x,y,p = x[order], y[order], p[order])

        # compute cuts on GPU; right=False == 'left' to mimic [t0, t1)
        # We want K = len(images)-1 intervals => use adjacent pairs from t_bounds
        lefts = torch.searchsorted(t, t_bounds[:-1], right=False)  # start indices
        rights = torch.searchsorted(t, t_bounds[1:], right=False)  # end indices

        grids = []
        raw_images = frames[frame_idxs]  # length len(images), same as your original

        for k in range(lefts.numel()):
            s = int(lefts[k].item())
            e = int(rights[k].item())

            # skip degenerate spans (no events or zero/negative duration)
            t0 = int(t_bounds[k].item())
            t1 = int(t_bounds[k + 1].item())
            if e <= s or t1 <= t0:
                continue

            xk, yk, tk, pk = x[s:e], y[s:e], t[s:e], p[s:e]

            # (optional) hard safety: clamp x/y once if your ESIM can emit border values
            # H, W = frames.shape[-2], frames.shape[-1]
            # xk = xk.clamp_(0, W-1); yk = yk.clamp_(0, H-1)

            g = events_to_voxels(xk, yk, tk, pk, t0, t1)
            grids.append(g)

        event_grids = torch.stack(grids, dim=0)

        # write caches
        # cast & move to CPU arrays only once, right before saving
        _save_cached_npy(paths["frames"], raw_images.astype(np.uint8))
        _save_cached_npy(paths["grids"], event_grids.detach().to("cpu").to(torch.float16).numpy())

        # Compact events to smaller dtypes on disk
        _save_cached_npy(paths["x"], x.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["y"], y.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["t"], t.detach().to("cpu").to(torch.int64).numpy())
        _save_cached_npy(paths["p"], p.detach().to("cpu").to(torch.int8).numpy())

        # Return in the same shapes/types you expect
        result = {
            "frames": raw_images,                                 # np.uint8
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

        if self.mode == "event":
            return self._read_events_or_create(folder, props)
        else:
            raise NotImplementedError("")


    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    import time
    ds = EventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "upsampled_train"),
                                   mode="event",
                                   B=5,
                                   force_recomputing=True)


    datapoint = ds[4]

    imgs, events, grids = datapoint["frames"], datapoint["events"], datapoint["grids"]
    optical_flow = datapoint["optical_flows"]


    x,y, t, p = events['x'], events['y'], events['t'], events['p']

    # torch.Size([55, 180, 240])
    print(imgs.shape)

    # torch.Size([54, 5, 240, 180])
    print(grids.shape)

    # visualize a datapoint
    render_events_video(datapoint)
