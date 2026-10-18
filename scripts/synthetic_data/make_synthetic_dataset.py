#!/usr/bin/env python
"""
Generate a tiny, format-correct synthetic dataset for testing the silicon-retina /
DVS training scripts without downloading DHF1K or ECOCO
(used by run_synthetic_pipeline.sh).

It mimics exactly what VideoDatasetUpsamplingCacheToFolder writes (once with
upsample_to_events=True -> *_OF, once with new_framerate=200 -> *_FF, both into
the same folder, which is how both loaders' inputs end up side by side), but
renders the "upsampled" frames analytically instead of running FILM.

Layout produced (default --layout ready):

  <root>/saliency_dataset_ready/{train,val}/
      metadata.json
      video_00000/
          params.json
          raw_images.npz                         key raw_images  (T,H,W,3) uint8, at `fps`
          outputs.npz                            key outputs     (T,H,W,3) uint8 saliency maps, at `fps`
          video_upsampled_fps_eq_200.mp4         len(timestamps_FF) RGB frames (retina loader default)
          video_upsampled_using_optical_flow.mp4 len(timestamps_OF) RGB frames (DVS loader)
          [optical_flow.npz]                     key optical_flow (T-1,H,W,2) float32, only with --with-flow

With --layout upsampled, all videos go flat into <root>/saliency_dataset_upsampled/
(the input of data_help/partition_saliency.py).

The scene: horizontally panning blocky texture + one bright disc moving on a
Lissajous path; the ground-truth saliency map is a Gaussian around the blob.
"""
import argparse
import json
import os

import imageio
import numpy as np


def render(t, H, W, seed):
    """Render RGB uint8 frame at continuous time t (seconds) + blob centre."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Blocky (sharp-edged) texture panning horizontally = camera motion.  The silicon
    # retina's spatial high-pass ignores smooth sinusoids, so edges are needed to
    # reach the ~1-10% grid occupancy that check_dataset (min 1%) requires.
    blk = 8
    tex = np.kron(rng.random((H // blk + 1, (W + 1000) // blk + 1)), np.ones((blk, blk)))[:H]
    vx = rng.uniform(60, 120)  # px / s
    off = int(vx * t)
    bg = 0.5 + 0.6 * (tex[:, off:off + W] - 0.5)
    # moving blob (sharp-edged bright disc)
    ax, ay = rng.uniform(0.25, 0.4, size=2)
    wx, wy = rng.uniform(2.0, 4.0, size=2) * 2 * np.pi / 2.0
    ph = rng.uniform(0, 2 * np.pi)
    cx = W * (0.5 + ax * np.sin(wx * t + ph))
    cy = H * (0.5 + ay * np.sin(wy * t))
    r = 0.12 * min(H, W)
    blob = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(np.float32)
    img = np.clip(bg * (1 - blob) + 0.95 * blob, 0, 1)
    rgb = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    return (rgb * 255 + 0.5).astype(np.uint8), (cx, cy)


def saliency(center, H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = center
    sig = 0.1 * min(H, W)
    m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    m = (255 * m / m.max()).astype(np.uint8)
    return np.repeat(m[..., None], 3, axis=-1)  # maps.mp4 frames are read as RGB


def ff_counts(T, fps, new_fps):
    """Replicates VideoDatasetUpsampling._upsample_to_fixed_framerate bookkeeping."""
    new_dt, img_dt = 1.0 / new_fps, 1.0 / fps
    times = np.arange(0, img_dt * T, new_dt)
    i, prev, deltas = 0, 0, []
    for time in times:
        if time >= (i + 1) * img_dt:
            i += 1
            deltas.append(prev)
            prev = 0
        prev += 1
    deltas.append(prev)
    return times, np.array(deltas)


def of_schedule(T, fps, centers):
    """Replicates _upsample_optical_flow: per raw pair, 2**k frames (k bisections
    from max flow magnitude, capped at 4) starting at the pair's first frame."""
    dt = 1.0 / fps
    ts, counts = [], []
    for i in range(T - 1):
        flow = np.hypot(centers[i + 1][0] - centers[i][0], centers[i + 1][1] - centers[i][1])
        k = 0 if flow < 1.0 else min(int(np.ceil(np.log2(flow))), 4)
        n = 2 ** k if k > 0 else 2  # k==0 -> [I0, midpoint]
        t0 = i * dt
        ts.extend(t0 + j * dt / n for j in range(n))
        counts.append(n)
    return np.array(ts), np.array(counts)


def write_mp4(frames, path, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    for f in frames:
        w.append_data(f)
    w.close()


def make_video(folder, seed, T, fps, H, W, new_fps, with_flow, gray=False):
    os.makedirs(folder, exist_ok=True)
    raw, centers = zip(*[render(i / fps, H, W, seed) for i in range(T)])
    raw = np.stack(raw)
    if gray:  # ECOCO: grayscale PNG frames (T,H,W); writer stores outputs = raw_images
        raw = raw[..., 0].copy()
        outputs = raw
    else:     # saliency: RGB frames + RGB-decoded maps.mp4 saliency maps
        outputs = np.stack([saliency(c, H, W) for c in centers])

    ts_ff, cnt_ff = ff_counts(T, fps, new_fps)
    ts_of, cnt_of = of_schedule(T, fps, centers)
    assert cnt_ff.sum() == len(ts_ff) and cnt_of.sum() == len(ts_of)

    write_mp4([render(t, H, W, seed)[0] for t in ts_ff], os.path.join(folder, f"video_upsampled_fps_eq_{new_fps}.mp4"), fps)
    write_mp4([render(t, H, W, seed)[0] for t in ts_of], os.path.join(folder, "video_upsampled_using_optical_flow.mp4"), fps)
    np.savez_compressed(os.path.join(folder, "raw_images.npz"), raw_images=raw)
    np.savez_compressed(os.path.join(folder, "outputs.npz"), outputs=outputs)
    if with_flow:
        fl = np.zeros((T - 1, H, W, 2), np.float32)
        for i in range(T - 1):
            fl[i, ..., 0] = centers[i + 1][0] - centers[i][0]
            fl[i, ..., 1] = centers[i + 1][1] - centers[i][1]
        np.savez_compressed(os.path.join(folder, "optical_flow.npz"), optical_flow=fl)

    params = {
        "raw_images": "raw_images.npz",
        "fps": fps,
        "optical_flow_path": "optical_flow.npz",
        "new_images_per_pair_OF": cnt_of.tolist(),
        "timestamps_OF": ts_of.tolist(),
        "new_images_per_pair_FF": cnt_ff.tolist(),
        "timestamps_FF": ts_ff.tolist(),
    }
    with open(os.path.join(folder, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    return len(ts_ff), len(ts_of)


def write_metadata(split_dir, n, new_fps):
    meta = {
        "number_of_videos": n,
        "bit_depth": 8,
        "model_path": "models/film_net/Style/saved_model",
        "optical_flow_saved": True,
        "optical_flow_name": "optical_flow.npz",
        "upsampling_OF_saved": True,
        "upsampling_to_fixed_fps": True,
        "new_framerate": new_fps,
    }
    with open(os.path.join(split_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="src/neural_network/data", help="repo data dir")
    ap.add_argument("--layout", choices=["ready", "upsampled", "ecoco"], default="ready",
                    help="ready: saliency_dataset_ready/{train,val}; upsampled: flat saliency_dataset_upsampled; "
                         "ecoco: upsampled_train + upsampled_validation (intensity reconstruction, use --H 180 --W 240 --fps 27)")
    ap.add_argument("--n-train", type=int, default=2)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--n-videos", type=int, default=10, help="for --layout upsampled")
    ap.add_argument("--frames", type=int, default=16, help="raw frames per video")
    ap.add_argument("--fps", type=int, default=30, help="raw fps (DHF1K pipeline uses 30)")
    ap.add_argument("--H", type=int, default=180)
    ap.add_argument("--W", type=int, default=320)
    ap.add_argument("--new-fps", type=int, default=200)
    ap.add_argument("--with-flow", action="store_true",
                    help="write optical_flow.npz (implied for --layout ecoco: ECOCODatasetWrapper needs it)")
    a = ap.parse_args()
    if a.layout == "ecoco":
        a.with_flow = True

    if a.layout == "ready":
        splits = {"train": a.n_train, "val": a.n_val}
        base = os.path.join(a.data_root, "saliency_dataset_ready")
    elif a.layout == "ecoco":
        splits = {"upsampled_train": a.n_train, "upsampled_validation": a.n_val}
        base = a.data_root
    else:
        splits = {"": a.n_videos}
        base = os.path.join(a.data_root, "saliency_dataset_upsampled")
    seed = 0
    for split, n in splits.items():
        d = os.path.join(base, split)
        os.makedirs(d, exist_ok=True)
        write_metadata(d, n, a.new_fps)
        for i in range(n):
            T = a.frames + 2 * (i % 3)
            nff, nof = make_video(os.path.join(d, f"video_{seed:05d}"), seed, T, a.fps, a.H, a.W, a.new_fps, a.with_flow,
                                  gray=(a.layout == "ecoco"))
            print(f"{d}/video_{seed:05d}: raw={T} FF={nff} OF={nof}")
            seed += 1
