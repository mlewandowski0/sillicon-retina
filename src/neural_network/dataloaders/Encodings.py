import numpy as np 
import torch 
import imageio
import os 
import enum 

class EventRepresentation(enum.Enum):
    EVENTS = 1
    EVENTS_TIME_PLUS_POS = 2
    

def events_to_voxels(x, y, t, p, t0, t1, H= 180, W = 240, B = 5, device="cuda"):
        dt = float(t1 - t0)
        if dt == 0:
            return torch.zeros(B, H, W, device=device, dtype=torch.float32).to(x.device)

        t_star = (t - t0) / dt * (B - 1)

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


def get_grids(x, y, t, p, fps, device, raw_frames, H, W, B = 5): 
 
    raw_frames_dt = 1 / fps
    t_bounds = []
                
    for i in range(len(raw_frames) - 1):
        t_bounds.append([i * raw_frames_dt * 1e9, (i+1) * raw_frames_dt * 1e9])
        
    t_bounds = torch.Tensor(t_bounds).to(x.device)
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
            grids.append(torch.zeros((B, H, W), dtype=torch.float16, device=x.device))
            continue

        xk, yk, tk, pk = x[s:e], y[s:e], t[s:e], p[s:e]


        g = events_to_voxels(xk, yk, tk, pk, t0, t1, device=x.device, H=H, W=W)
        grids.append(g)

    event_grids = torch.stack(grids, dim=0)
    return event_grids

def hash_jit(lin, jitter_ns):
    z = lin.clone()
    z ^= (z << 13) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 7)
    z ^= (z << 17) & 0xFFFFFFFFFFFFFFFF
    z = (z & 0x7FFFFFFFFFFFFFFF) % int(jitter_ns)
    return z.to(torch.int64)

def bitmaps_to_events(pos: torch.Tensor, 
                      neg: torch.Tensor, 
                      ts_ns: torch.Tensor,
                      c_ns = None,                      # sub-tick in ns (per + 1 linear index stack)
                      jitter_ns = None,                 # int max random jitter in ns (uniform [0, jitter_ns])
                      hashed_jitter : bool = False,     # if True, use deterministic hash jitter instead of random
                      clamp_within_frame : bool = True, # if True, clamp jittered times to within original frame bounds
                      stable_sort: bool = True):
    """
    pos/neg: (T,H,W) byte/bool/float {0,1}; ts_ns: (T,) int64 nanoseconds (monotonic)
    Returns x,y,t,p (all 1D tensors, sorted by t). p in {+1,-1} int8.
    """
    T,H,W = pos.shape
    HW = H * W

    # work on single device 
    dev = pos.device
    pos_lin = pos.view(T, -1).nonzero(as_tuple=False).to(dev)
    neg_lin = neg.view(T, -1).nonzero(as_tuple=False).to(dev)
     
    
    def unpack(ti):
        # ti : (N, 2) with columns, (t_idx, linear_idx)
        t = ts_ns.index_select(0, ti[:,0].long())

        i = ti[:, 1].long()
        y = torch.div(i, W, rounding_mode='floor').to(torch.int32)
        x = (i - (y.to(i.dtype) * W)).to(torch.int32)
        lin = i
        
        return x, y, t, lin


    x_on,y_on,t_on, lin_on      = unpack(pos_lin)
    x_off,y_off,t_off, lin_off  = unpack(neg_lin)
    
    # comptue a safe default c_ns from median frame spacing 
    if c_ns is None and (hashed_jitter or jitter_ns is not None):
        dt = torch.median(ts_ns[1:] - ts_ns[:-1]).item() if T > 1 else 0
        if dt > 0:
            c_ns = max(1, dt // (HW + 1))
            
    # positional micro-time staggering 
    if c_ns is not None and c_ns > 0:
        t_on  = t_on  + (lin_on * int(c_ns)).to(t_on.dtype)
        t_off = t_off + (lin_off * int(c_ns)).to(t_off.dtype)   

    # Optional extra jitter (either random per event or hashed per-pixel deterministically)
    if jitter_ns is not None and jitter_ns > 0:
        if hashed_jitter:
            # deterministic hash-based jitter in [0, jitter_ns)
            t_on = t_on + hash_jit(lin_on, jitter_ns)
            t_off = t_off + hash_jit(lin_off, jitter_ns)
        else:
            t_on = t_on + torch.randint(0, int(jitter_ns), (t_on.shape[0],), device=dev, dtype=torch.int64)
            t_off = t_off + torch.randint(0, int(jitter_ns), (t_off.shape[0],), device=dev, dtype=torch.int64)

    # (Optional) clam within the frame's [t_k, t_{k+1}]            
    if clamp_within_frame and T > 1:
        # Compute frame start/end for each event from its t-index (the first column of on_lin/off_lin)
        t_idx_on  = pos_lin[:, 0].long()
        t_idx_off = neg_lin[:, 0].long()
        t_start = ts_ns
        t_end   = torch.empty_like(ts_ns)
        t_end[:-1] = ts_ns[1:]
        t_end[-1]  = ts_ns[-1] + (ts_ns[-1] - ts_ns[-2] if T > 1 else 1)  # last frame margin

        t_on  = torch.minimum(torch.maximum(t_on,  t_start.index_select(0, t_idx_on)),  t_end.index_select(0, t_idx_on)  - 1)
        t_off = torch.minimum(torch.maximum(t_off, t_start.index_select(0, t_idx_off)), t_end.index_select(0, t_idx_off) - 1)

    p_on  = torch.ones_like(x_on,  dtype=torch.int8)
    p_off = -torch.ones_like(x_off, dtype=torch.int8)

    x = torch.cat([x_on,  x_off],  dim=0)
    y = torch.cat([y_on,  y_off],  dim=0)
    t = torch.cat([t_on,  t_off],  dim=0)
    p = torch.cat([p_on,  p_off],  dim=0)

    order = torch.argsort(t, stable=True)
    return x[order], y[order], t[order], p[order]

