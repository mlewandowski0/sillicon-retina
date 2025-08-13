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

class EventDatasetFromUpsampled(Dataset):


    def __init__(self, directory,
                       mode = "event",
                       B = 5,
                       contrast_threshold_neg = 0.2,
                       contrast_threshold_pos = 0.2,
                       force_recomputing=False,
                        limit=-1):

        self.directory = directory
        self.B = B
        self.force_recomputing = force_recomputing

        metadata_file = os.path.join(self.directory, "metadata.json")
        self.read_metadata = {}
        with open(metadata_file, "r") as f:
            self.read_metadata = json.load(f)

        self.mode = mode
        self.data_files = sorted([os.path.join(directory, el) for el in os.listdir(directory) if os.path.isdir(os.path.join(directory, el))])

        if limit is not None and limit > 0:
            self.data_files = self.data_files[:limit]

        self.event_simulator = esim_torch.ESIM(contrast_threshold_neg=contrast_threshold_neg,
                                               contrast_threshold_pos=contrast_threshold_pos,
                                               refractory_period_ns=0)


    def read_events_or_create(self, folder, properties):

        event_grid_file = os.path.join(folder, f"events_B={self.B}.npz")
        raw_frames_file = os.path.join(folder, f"raw_frames.npz")
        events_file     = os.path.join(folder, f"events.npz")

        event_grids_loaded = False
        timestamps = np.array(properties["timestamps"])
        images = properties["images"]
        fps = properties["fps"]

        if (os.path.exists(event_grid_file) and os.path.exists(raw_frames_file) and os.path.exists(events_file)
                and not self.force_recomputing):

            event_file_npy = np.load(event_grid_file)
            event_grids = torch.Tensor(event_file_npy["event_grids"])

            raw_images = np.load(raw_frames_file)["raw_images"]

            events = np.load(events_file)
            events = {"x" : torch.Tensor(events['x']),
                      "y" : torch.Tensor(events['y']),
                      't' : torch.Tensor(events['t']),
                      'p' : torch.Tensor(events['p'])}

            result = {"frames" : raw_images,
                      "events" : events,
                      "grids" : event_grids}
        else:
            frames = load_video(os.path.join(folder, "video.mp4"))[..., 0]

            event_grids = []

            timestamps_ns = (timestamps * 1e9).astype("int64")
            log_images = np.log(frames.astype("float32") / 255 + 1e-4)

            device = "cuda"
            log_images = torch.from_numpy(log_images).to(device)
            timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

            # reset the event simulation
            self.event_simulator.reset()

            # generate events with GPU support
            events = self.event_simulator.forward(log_images, timestamps_ns)
            x, y, t, p = events['x'], events['y'], events['t'], events['p']

            # build voxel grids
            prev = 0
            cnt = 0

            raw_images = [frames[0]]

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

                if not event_grids_loaded:
                    event_grids.append(events_to_voxels(x[mask], y[mask], t[mask], p[mask],
                                                    t0, t1))

            raw_images = np.array(raw_images)


            event_grids = torch.stack(event_grids, dim=0)
            np.savez_compressed(event_grid_file, event_grids=event_grids.cpu())
            np.savez_compressed(events_file, x=x.cpu(), y=y.cpu(), t=t.cpu(), p=p.cpu())
            np.savez_compressed(raw_frames_file,raw_images=raw_images)

            result = {"frames" : np.array(raw_images),
                      "events" : events,
                      "grids" : event_grids}

        optical_flow_file = os.path.join(folder, "optical_flow.npz")

        if os.path.exists(optical_flow_file):
            optical_flow = np.load(optical_flow_file)["optical_flow"]
            result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

        return result

    def __getitem__(self, idx):

        folder = self.data_files[idx]

        properties = {}
        with open(os.path.join(folder, "params.json")) as f:
            properties = json.load(f)

        datapoint = {}
        if self.mode == "event":
            datapoint = self.read_events_or_create(folder, properties)

        # if "frames" in datapoint:
        #     datapoint["frames"] = torch.Tensor(datapoint['frames']) / 255.

        return datapoint



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
