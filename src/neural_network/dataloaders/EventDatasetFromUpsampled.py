import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
import esim_torch

def events_to_voxels(x, y, t, p, t0, t1, H= 240, W = 180, B = 5, device="cuda"):

        dt = (t1 - t0).float()
        if dt == 0:
            return torch.zeros(B, H, W, device=device, dtype=torch.float32)

        # Normalised ‘bin coordinate’ in [0, B−1]
        t_star = (t.float() - t0.float()) / dt * (B - 1)

        b0 = torch.floor(t_star).long()  # lower bin index
        w1 = t_star - b0.float()  # weight to upper bin
        w0 = 1.0 - w1

        b1 = b0 + 1  # upper bin index
        valid1 = b1 < B  # clip the overflow events

        # Flatten spatial locations so we can use index_add_
        idx_flat = y * W + x  # H×W → [0, H·W)
        stride = H * W

        grid = torch.zeros(B * H * W, device=device, dtype=torch.float32)

        # lower bin accumulation
        grid.index_add_(0,
                        idx_flat + b0 * stride,
                        p.float() * w0)

        # upper bin accumulation (only where b1 is inside range)
        if valid1.any():
            grid.index_add_(0,
                            idx_flat[valid1] + b1[valid1] * stride,
                            p.float()[valid1] * w1[valid1])

        return grid.view(B, H, W)

def _load_video( path):
    reader = imageio.get_reader(path, "ffmpeg")
    frames = []

    for frame in reader:
        frames.append(frame)

    reader.close()
    return np.stack(frames, axis=0)


class EventDatasetFromUpsampled(Dataset):


    def __init__(self, directory,
                       mode = "event",
                       B = 5):

        self.directory = directory
        self.B = B

        metadata_file = os.path.join(self.directory, "metadata.json")
        self.read_metadata = {}
        with open(metadata_file, "r") as f:
            self.read_metadata = json.load(f)

        self.mode = mode
        self.data_files = sorted([os.path.join(directory, el) for el in os.listdir(directory) if os.path.isdir(os.path.join(directory, el))])

        self.event_simulator = esim_torch.ESIM(contrast_threshold_neg=0.2,
                                               contrast_threshold_pos=0.2,
                                               refractory_period_ns=0)


    def read_events_or_create(self, folder, properties):

        event_file = os.path.join(folder, f"events_B={self.B}.npz")



        frames = _load_video(os.path.join(folder, "video.mp4"))[..., 0]
        timestamps = np.array(properties["timestamps"])

        images = properties["images"]
        fps = properties["fps"]

        timestamps_ns = (timestamps * 1e9).astype("int64")
        log_images = np.log(frames.astype("float32") / 255 + 1e-4)

        # generate torch tensors
        print("Loading data to GPU")
        device = "cuda"
        log_images = torch.from_numpy(log_images).to(device)
        timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

        # generate events with GPU support
        print("Generating events")
        events = self.event_simulator.forward(log_images, timestamps_ns)
        x, y, t, p = events['x'], events['y'], events['t'], events['p']

        # build voxel grids
        prev = 0
        cnt = 0

        raw_images = [frames[0]]
        event_grids = []

        for k in range(len(images) - 1):
            # print(f"[{prev}, {prev + images[k]}]")
            # print(f"[{timestamps[prev], timestamps[prev + images[k]]}] s")
            next_idx = prev + images[k]
            t0 = timestamps_ns[prev]
            t1 = timestamps_ns[next_idx]

            raw_images.append(frames[next_idx])
            cnt += 1

            prev = next_idx

            mask = (t >= t0) & (t < t1)
            event_grids.append(events_to_voxels(x[mask], y[mask], t[mask], p[mask],
                                                t0, t1))

        raw_images = np.array(raw_images)


        return raw_images, event_grids

    def __getitem__(self, idx):

        folder = self.data_files[idx]

        properties = {}
        with open(os.path.join(folder, "params.json")) as f:
            properties = json.load(f)

        if self.mode == "event":
            return torch.Tensor(raw_images)/255., events, torch.stack(event_grids, dim=0)



    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    ds = EventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "upsampled_train"))

    imgs, events, grids = ds[0]

    x,y, t, p = events['x'], events['y'], events['t'], events['p']

    # torch.Size([55, 180, 240])
    print(imgs.shape)

    # torch.Size([54, 5, 240, 180])
    print(grids.shape)