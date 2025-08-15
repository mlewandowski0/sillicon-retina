import os 
import numpy as np 


def _save_cached_npy(path, arr):
    np.save(path, arr)

def _memmap_load(path, dtype=None):
    #mmap for zero-copy
    return np.load(path, mmap_mode='r')

def _mk_cache_paths_events(folder, B, ctn, ctp, version='v2'):
    def f(x): return f"{float(x):.4f}".replace(".", "p")

    base = f"cache_B={B}_ctn={f(ctn)}_ctp={f(ctp)}_{version}"
    return {
        "grids": os.path.join(folder, base + "_grids.npy"),
        "frames": os.path.join(folder, base + "_frames.npy"),
        "x": os.path.join(folder, base + "_x.npy"),
        "y": os.path.join(folder, base + "_y.npy"),
        "t": os.path.join(folder, base + "_t.npy"),
        "p": os.path.join(folder, base + "_p.npy"),
        "lock": os.path.join(folder, base + ".lock"),
    }

def _mk_cache_paths_SE(folder, B):
    def f(x): return f"{float(x):.4f}".replace(".", "p")

    base = f"cache_B={B}_retina"
    return {
        "grids": os.path.join(folder, base + "_grids.npy"),
        "frames": os.path.join(folder, base + "_frames.npy"),
        "x": os.path.join(folder, base + "_x.npy"),
        "y": os.path.join(folder, base + "_y.npy"),
        "t": os.path.join(folder, base + "_t.npy"),
        "p": os.path.join(folder, base + "_p.npy"),
        "lock": os.path.join(folder, base + ".lock"),
    }
