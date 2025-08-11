import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
import esim_torch
import matplotlib


def flow_to_rgb(flow):
    W, H, _ = flow.shape
    hsv = np.zeros((W, H, 3), dtype=np.float32)

    X = flow[..., 0]
    Y = flow[..., 1]
    mag_sq = X ** 2 + Y ** 2
    max = np.max(mag_sq)
    angle = np.atan2(Y, X)

    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = np.sqrt(mag_sq) / np.sqrt(max) if max > 0 else 0.0
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255.0).astype(np.uint8)


import numpy as np
import imageio
import cv2

def render_events_video(
    datapoint,
    output_path="events_vis.mp4",
    fps=10,
    alpha=0.65,            # event overlay opacity
    on_color=(0,0,255),    # ON events (B,G,R)
    off_color=(255,0,0), # OFF events (B,G,R)
    quiver_step=16,        # stride for arrows
    quiver_scale=1.5       # arrow length scale
):
    """
    Expects datapoint with:
      images:       [T,H,W] or [T,H,W,1] in [0,1] or uint8
      event_grid:   [T,B,H,W]   (signed counts) OR
                    [T,2,B,H,W] (pos/neg separated) OR
                    [T,H,W]     (signed counts, already collapsed)
      optical_flow: [T,H,W,2]   (u,v)
    """
    # grab tensors
    rgbs   = np.array(datapoint["images"].cpu())
    events = np.array(datapoint["event_grid"].cpu())
    flows  = np.array(datapoint["optical_flow"].cpu())

    T = rgbs.shape[0]
    H = rgbs.shape[-3] if rgbs.ndim == 4 else rgbs.shape[-2]
    W = rgbs.shape[-2] if rgbs.ndim == 4 else rgbs.shape[-1]

    # make base frames uint8 BGR
    def to_bgr_u8(x):
        x = x.squeeze()  # ok if [H,W,1]
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1) * 255.0
        x = x.astype(np.uint8)
        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        elif x.shape[-1] == 1:
            x = cv2.cvtColor(x[...,0], cv2.COLOR_GRAY2BGR)
        return x

    # event helpers ----------------------------------------------------
    # unify to [T,2,B,H,W] or [T,1,B,H,W]
    if events.ndim == 4 and events.shape[1] not in (1,2):  # [T,B,H,W] -> [T,1,B,H,W]
        events = events[:,None,...]
    elif events.ndim == 3:  # [T,H,W] -> [T,1,1,H,W]
        events = events[:,None,None,...]

    has_polarity = (events.shape[1] == 2)
    B = events.shape[2] if events.ndim == 5 else 1

    # global robust min/max to avoid flicker
    ev_vals = events.reshape(-1)
    vmin = np.percentile(ev_vals, 1)
    vmax = np.percentile(ev_vals, 99)
    if vmax <= vmin: vmax = vmin + 1.0

    # simple increasing weights across bins (newer bins brighter)
    if B > 1:
        w = (np.arange(B, dtype=np.float32) + 1.0)
        w = w / w.sum()
    else:
        w = np.array([1.0], dtype=np.float32)

    def collapse_bins(ev_slice):  # ev_slice: [C,B,H,W] or [1,B,H,W]
        # weighted sum across bins
        if has_polarity:
            pos = np.tensordot(ev_slice[0], w, axes=(0,0))  # [H,W]
            neg = np.tensordot(ev_slice[1], w, axes=(0,0))
        else:
            g = np.tensordot(ev_slice[0], w, axes=(0,0))
            pos = np.clip(g, 0, None)
            neg = np.clip(-g, 0, None)
        return pos, neg

    def events_to_overlay(pos, neg):
        # normalize with global vmin/vmax, then clamp [0,1]
        pos_n = pos.copy()
        pos_n[pos != 0] = np.clip((pos[pos != 0] - vmin) / (vmax - vmin), 0, 1)

        neg_n = neg.copy()
        neg_n[neg != 0] = np.clip((neg[neg != 0] - vmin) / (vmax - vmin), 0, 1)

        # build sparse overlay + mask (only where events exist)
        mag = pos_n + neg_n
        mask = mag > 1e-6

        overlay = np.zeros((H,W,3), dtype=np.uint8)
        # ON
        if mask.any():
            on = (pos_n / (mag + 1e-12))  # per-pixel share
            off = (neg_n / (mag + 1e-12))
            # mix two colors proportionally
            overlay[mask] = np.clip(
                on[mask,None]*np.array(on_color) +
                off[mask,None]*np.array(off_color),
                0, 255
            ).astype(np.uint8)
        return overlay, mask

    # flow helpers ------------------------------------------------------
    def flow_to_rgb_hsv(flow):
        u, v = flow[...,0], flow[...,1]
        mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)
        hsv = np.zeros((H,W,3), dtype=np.uint8)
        hsv[...,0] = np.uint8((ang * 180/np.pi) / 2.0)  # 0..180
        hsv[...,1] = 255
        m = np.percentile(mag, 99) + 1e-6
        hsv[...,2] = np.uint8(np.clip(255 * (mag / m), 0, 255))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def draw_quiver(img_bgr, flow, step=16, scale=0.8):
        out = img_bgr.copy()
        ys = range(step//2, H, step)
        xs = range(step//2, W, step)
        for y in ys:
            for x in xs:
                u, v = flow[y, x]
                x2 = int(round(x + scale * u))
                y2 = int(round(y + scale * v))
                cv2.arrowedLine(out, (x, y), (x2, y2), (255,255,255), 1, tipLength=0.3)
        return out

    writer = imageio.get_writer(output_path, format="FFMPEG", fps=fps, codec="libx264", quality=7)
    for t in range(T-1):
        base = to_bgr_u8(rgbs[t+1])
        ev_slice = events[t] if events.ndim == 5 else events[t][:,None,...]  # [C,B,H,W]
        pos, neg = collapse_bins(ev_slice)
        ev_overlay, ev_mask = events_to_overlay(pos, neg)

        # overlay ONLY on event pixels (no global tint)
        merged = base.copy()
        merged[ev_mask] = cv2.addWeighted(base[ev_mask], 1.0 - alpha, ev_overlay[ev_mask], alpha, 0)

        flow = flows[t]
        flow_rgb = flow_to_rgb_hsv(flow)
        quiv = draw_quiver(base, flow, step=quiver_step, scale=quiver_scale)

        top = np.concatenate([base, merged], axis=1)
        bottom = np.concatenate([flow_rgb, quiv], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        # light labels
        cv2.putText(grid, "Image", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(grid, "Image + Events", (10+base.shape[1],20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(grid, "Flow (HSV)", (10, 20+top.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(grid, "Quiver", (10+flow_rgb.shape[1], 20+top.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    writer.close()

def events_to_voxels(x, y, t, p, t0, t1, H= 180, W = 240, B = 5, device="cuda"):

        dt = (t1 - t0).float()
        if dt == 0:
            return torch.zeros(B, H, W, device=device, dtype=torch.float32)

        t_star = (t.float() - t0.float()) / dt * (B - 1)

        b0 = torch.floor(t_star).long()
        w1 = t_star - b0.float()
        w0 = 1.0 - w1

        b1 = b0 + 1
        valid1 = b1 < B

        idx_flat = y * W + x  # H×W → [0, H·W)
        stride = H * W

        grid = torch.zeros(B * H * W, device=device, dtype=torch.float32)

        grid.index_add_(0,
                        idx_flat + b0 * stride,
                        p.float() * w0)

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
                       B = 5,
                       force_recomputing=False):

        self.directory = directory
        self.B = B
        self.force_recomputing = force_recomputing

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
        event_grids_loaded = False

        frames = _load_video(os.path.join(folder, "video.mp4"))[..., 0]
        timestamps = np.array(properties["timestamps"])
        images = properties["images"]
        fps = properties["fps"]

        if os.path.exists(event_file) and not self.force_recomputing:
            event_file_npy = np.load(event_file)
            event_grids = torch.Tensor(event_file_npy["event_grids"])

            raw_images = [frames[0]]
            prev = 0

            for k in range(len(images) - 1):
                next_idx = prev + images[k]
                raw_images.append(frames[next_idx])

            events = {"x" : torch.Tensor(event_file_npy['x']),
                      "y" : torch.Tensor(event_file_npy['y']),
                      't' : torch.Tensor(event_file_npy['t']),
                      'p' : torch.Tensor(event_file_npy['p'])}

            result = {"images" : np.array(raw_images),
                      "events" : events,
                      "event_grid" : event_grids}
        else:
            event_grids = []

            timestamps_ns = (timestamps * 1e9).astype("int64")
            log_images = np.log(frames.astype("float32") / 255 + 1e-4)

            device = "cuda"
            log_images = torch.from_numpy(log_images).to(device)
            timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

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
            np.savez_compressed(event_file, event_grids=event_grids.cpu(), x=x.cpu(), y=y.cpu(), t=t.cpu(), p=p.cpu())

            result = {"images" : np.array(raw_images),
                      "events" : events,
                      "event_grid" : event_grids}

        optical_flow_file = os.path.join(folder, "optical_flow.npz")

        if os.path.exists(optical_flow_file):
            optical_flow = np.load(optical_flow_file)["optical_flow"]
            result["optical_flow"] = torch.Tensor(optical_flow)

        return result

    def __getitem__(self, idx):

        folder = self.data_files[idx]

        properties = {}
        with open(os.path.join(folder, "params.json")) as f:
            properties = json.load(f)

        datapoint = {}
        if self.mode == "event":
            datapoint = self.read_events_or_create(folder, properties)




        if "images" in datapoint:
            datapoint["images"] = torch.Tensor(datapoint['images']) / 255.


        return datapoint



    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    ds = EventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "upsampled_train"),
                                   force_recomputing=True)

    datapoint = ds[0]
    imgs, events, grids = datapoint["images"], datapoint["events"], datapoint["event_grid"]
    optical_flow = datapoint["optical_flow"]


    x,y, t, p = events['x'], events['y'], events['t'], events['p']

    # torch.Size([55, 180, 240])
    print(imgs.shape)

    # torch.Size([54, 5, 240, 180])
    print(grids.shape)

    # visualize a datapoint
    render_events_video(datapoint)
