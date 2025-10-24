# upsample_worker_gpu.py
import os, sys, site

def _prepend_cuda_wheels_to_ld_library_path():
    libroots = []
    for pkg, sub in [
        ("nvidia/cuda_runtime", "lib"),
        ("nvidia/cublas",       "lib"),
        ("nvidia/cudnn",        "lib"),
        ("nvidia/cufft",        "lib"),
        ("nvidia/cusolver",     "lib"),
        ("nvidia/cusparse",     "lib"),
        ("nvidia/nvjitlink",    "lib"),
    ]:
        for sp in site.getsitepackages():
            p = os.path.join(sp, pkg, sub)
            if os.path.isdir(p):
                libroots.append(p)
    if libroots:
        os.environ["LD_LIBRARY_PATH"] = ":".join(libroots + [os.environ.get("LD_LIBRARY_PATH","")])

def _set_qol_env():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    # if you rely on imageio’s bundled ffmpeg, this helps avoid PATH issues
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", get_ffmpeg_exe())
    except Exception:
        pass

# --- set environment BEFORE importing tensorflow ---
_prepend_cuda_wheels_to_ld_library_path()
_set_qol_env()

import tensorflow as tf

# enable growth (programmatic)
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# optional mixed precision (safe for FILM/FLIP infer; comment out if undesired)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# print a one-liner sanity so you can see in logs
print("TF build:", tf.__version__, tf.sysconfig.get_build_info())
print("TF GPUs:", tf.config.list_physical_devices('GPU'))

# ---- your worker logic ----
from src.neural_network.dataloaders.VideoDatasetUpsampling import (
    VideoDatasetUpsampling, VideoDatasetUpsamplingCacheToFolder
)
from src.neural_network.dataloaders.VideoDatasetFromPairs import VideoDatasetFromVideoPairGlobal

def main():
    OUT = "src/neural_network/data/saliency_dataset_upsampled"
    try:
        start_index = int(os.environ.get("START_IDX", "0"))
    except Exception:
        start_index = 0

    video_val = VideoDatasetFromVideoPairGlobal(
        dirpath=os.path.join("src","neural_network","data","saliency_dataset_truncated"),
        split=None, global_fps=30,
        input_video="input.AVI", output_video="maps.mp4", suffix=None
    )

    upsampling_ds_val = VideoDatasetUpsampling(
        video_val,
        model_path=os.path.join("models","film_net","Style","saved_model"),
        upsample_to_events=True,
        new_framerate=False,
        debug=False
    )

    # start_index lets you resume; override=False skips already-complete items
    VideoDatasetUpsamplingCacheToFolder(
        upsamplingDataset=upsampling_ds_val,
        output_path=OUT,
        start_index=start_index,
        override=False
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
