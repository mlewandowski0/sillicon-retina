"""
End-to-end smoke test that needs no dataset download.

It exercises the same code paths as the paper's saliency experiments on a short
synthetic clip (a bright disc moving over a textured background):

  1. DVS events      - esim_torch.ESIM                      (baseline)
  2. Retina events   - esim_torch.SiliconRetinaFloatCUDAOnline (this work)
  3. Encoding        - src.neural_network.dataloaders.Encodings (voxel grids)
  4. Network         - FireNet-style FIRE_network + SaliencyLoss, one
                       forward/backward step for each event type.

Run from the repository root (after scripts/setup/install.sh):

    python -m scripts.smoke_test            # or: python scripts/smoke_test.py
    python -m scripts.smoke_test --save-figure media/readme/smoke_test.png
"""
import argparse
import os
import sys

import numpy as np
import torch

# allow `python scripts/smoke_test.py` as well as `python -m scripts.smoke_test`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esim_torch import ESIM, SiliconRetinaConfig, SiliconRetinaFloatCUDAOnline  # noqa: E402
from src.neural_network.dataloaders.Encodings import bitmaps_to_events, events_to_voxels  # noqa: E402
from src.neural_network.networks.FIRE_network import FIRE_network  # noqa: E402
from src.neural_network.networks.SaliencyLoss import SaliencyLoss  # noqa: E402


def make_clip(T=60, H=128, W=160, seed=0):
    """Moving disc over a static random texture, values in [0, 1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    background = 0.3 + 0.1 * rng.random((H, W))
    frames, saliency = [], []
    for t in range(T):
        cx, cy = 20 + (W - 40) * t / (T - 1), H / 2 + 20 * np.sin(t / 8)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        frames.append(np.where(d2 < 15 ** 2, 0.9, background))
        saliency.append(np.exp(-d2 / (2 * 12 ** 2)))  # "fixation map" on the disc
    return np.stack(frames).astype(np.float32), np.stack(saliency).astype(np.float32)


def to_voxel_grids(x, y, t, p, ts_ns, H, W, B=5):
    """One B-bin voxel grid per inter-frame interval (as in the dataloaders)."""
    grids = []
    for k in range(len(ts_ns) - 1):
        t0, t1 = int(ts_ns[k]), int(ts_ns[k + 1])
        m = (t >= t0) & (t < t1)
        if m.sum() == 0:
            grids.append(torch.zeros((B, H, W), dtype=torch.float16, device=x.device))
        else:
            grids.append(events_to_voxels(x[m], y[m], t[m], p[m], t0, t1, H=H, W=W, B=B, device=x.device))
    return torch.stack(grids).float()


def dvs_events(frames, ts_ns, threshold=0.2):
    esim = ESIM(contrast_threshold_neg=threshold, contrast_threshold_pos=threshold)
    ev = esim.forward(torch.log(torch.from_numpy(frames).cuda() + 1e-4), ts_ns.cuda())
    return ev["x"].long(), ev["y"].long(), ev["t"].long(), ev["p"].long()


def retina_events(frames, ts_ns):
    T, H, W = frames.shape
    retina = SiliconRetinaFloatCUDAOnline(SiliconRetinaConfig(H=H, W=W))
    retina.reset()
    pos, neg = [], []
    with torch.no_grad():
        for t in range(1, T + 1):
            on, off = retina(torch.from_numpy(frames[t - 1][None]).cuda(), t)
            pos.append(on.clone()[0])
            neg.append(off.clone()[0])
    pos, neg = torch.stack(pos), torch.stack(neg)
    x, y, t, p = bitmaps_to_events(pos=pos.cpu(), neg=neg.cpu(), ts_ns=ts_ns)
    return x.long().cuda(), y.long().cuda(), t.long().cuda(), p.long().cuda(), pos, neg


def train_step(grids, saliency, seq=20):
    torch.manual_seed(0)
    B, H, W = grids.shape[1:]
    net = FIRE_network(in_ch=B, hid1=16, ks1=3, add_bn=True, out_mode="density").cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = SaliencyLoss()
    net.reset(input_shape=(H, W), batch_size=1)
    loss = 0.0
    for k in range(seq):
        pred = net(grids[k][None])
        loss = loss + loss_fn(pred, saliency[k + 1][None, None])[0]  # (total, kl, cc)
    loss = loss / seq
    opt.zero_grad()
    loss.backward()
    opt.step()
    n_params = sum(p.numel() for p in net.parameters())
    return loss.item(), n_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-figure", default=None, help="optional PNG with input / DVS / retina panels")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "A CUDA GPU is required (esim_torch kernels)."
    frames, saliency = make_clip()
    T, H, W = frames.shape
    ts_ns = torch.arange(T, dtype=torch.int64) * 5_000_000  # 200 FPS, as in the paper's fixed-rate data
    sal = torch.from_numpy(saliency).cuda()

    dx, dy, dt, dp = dvs_events(frames, ts_ns)
    rx, ry, rt, rp, pos, neg = retina_events(frames, ts_ns)
    n_pix = (T - 1) * H * W
    print(f"clip: {T} frames of {H}x{W}")
    print(f"DVS    events: {dx.numel():7d}  ({100 * dx.numel() / n_pix:.2f}% of pixel-frames)")
    print(f"Retina events: {rx.numel():7d}  ({100 * rx.numel() / n_pix:.2f}% of pixel-frames)")
    assert dx.numel() > 0 and rx.numel() > 0, "a simulator produced no events"

    dvs_grids = to_voxel_grids(dx, dy, dt, dp, ts_ns.cuda(), H, W)
    ret_grids = to_voxel_grids(rx, ry, rt, rp, ts_ns.cuda(), H, W)
    for name, grids in [("DVS", dvs_grids), ("Retina", ret_grids)]:
        loss, n_params = train_step(grids, sal)
        assert np.isfinite(loss)
        print(f"FireNet ({n_params} params) one training step on {name:6s} voxel grids: loss = {loss:.4f}")

    if args.save_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = T // 2
        fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
        ax[0].imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("input frame")
        ax[1].imshow(dvs_grids[k - 1].sum(0).cpu(), cmap="bwr", vmin=-2, vmax=2)
        ax[1].set_title("DVS events")
        ax[2].imshow((pos[k].float() - neg[k].float()).cpu(), cmap="bwr", vmin=-1, vmax=1)
        ax[2].set_title("silicon retina events")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.save_figure) or ".", exist_ok=True)
        fig.savefig(args.save_figure, dpi=120)
        print("saved", args.save_figure)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
