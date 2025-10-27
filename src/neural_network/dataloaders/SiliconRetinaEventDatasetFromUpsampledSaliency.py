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
from src.neural_network.dataloaders.Encodings import *
from src.neural_network.utils.util import update_json
import glob

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

# Convert to numpy bool arrays with shape (T, H, W)
def to_bool_numpy(x):
    # x can be: torch.Tensor [T,H,W], list of tensors [H,W], or numpy
    if isinstance(x, (list, tuple)):
        x = [xi.detach().cpu().numpy() if hasattr(xi, "detach") else np.asarray(xi) for xi in x]
        x = np.stack(x, axis=0)
    else:
        x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return x.astype(bool)

def del_cache_of_folder(folder):
    for file in glob.glob(os.path.join(folder, "cached_*")):
        if os.path.isfile(file):
            os.remove(file)
    

class SiliconRetinaEventDatasetFromUpsampled(Dataset):

    def __init__(self, directory,
                       prediction_video_name,
                       preprocessing_type : EventRepresentation = EventRepresentation.EVENTS,
                       preprocessing_params = {},
                       height=180,
                       width=240,
                       config = SiliconRetinaConfig(),
                       force_recomputing=False,
                       input_framerate=200,
                       log_scale=False,
                       device="cuda",
                        limit=-1,
                        use_upsampled_from_OF=False,
                       grids_dtype=np.float16,
                       verbose : bool = False):

        self.verbose = verbose
        self.directory = directory
        self.preprocessing_type = preprocessing_type
        self.preprocessing_params = preprocessing_params
        self.use_upsampled_from_OF = use_upsampled_from_OF
        self.prediction_video_name = prediction_video_name

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
            self.simulator = SiliconRetinaFloatCUDAOnline(silicon_retina_config=self.config)

    def __len__(self):
        return len(self.data_files)

    def _mk_paths(self, folder):
        return _mk_cache_paths_SE(folder, self.preprocessing_params.get("num_bins", 5))

    @staticmethod
    def _load_video_gray(path, fit_to_size=None):
        """
        Load an MP4 (or image stack) as grayscale uint8.
        If fit_to_size=(W,H) is given, resize each frame to exactly (H, W).

        Returns:
            np.ndarray with shape:
                - (N, H, W) uint8
        """
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image

        vid = iio.imread(path)  # (N,H,W,C) or (N,H,W); dtype usually uint8

        # If RGB(A), convert to grayscale using luminance weights
        if vid.ndim == 4:
            # Keep only RGB if RGBA
            if vid.shape[-1] == 4:
                vid = vid[..., :3]
            # Convert to float for weighted sum, then back to uint8
            r = vid[..., 0].astype(np.float32)
            g = vid[..., 1].astype(np.float32)
            b = vid[..., 2].astype(np.float32)
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            vid = np.clip(gray, 0, 255).astype(np.uint8)
        elif vid.ndim == 3:
            # Already grayscale (N,H,W)
            if vid.dtype != np.uint8:
                # If it came in as float [0,1], scale up; otherwise just clip
                if np.issubdtype(vid.dtype, np.floating):
                    vid = np.clip(vid * 255.0, 0, 255).astype(np.uint8)
                else:
                    vid = np.clip(vid, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected video array shape {vid.shape}")

        # Resize if requested
        if fit_to_size is None:
            return vid

        W, H = fit_to_size
        if W <= 0 or H <= 0:
            raise ValueError(f"fit_to_size must be positive (got {fit_to_size})")

        N = vid.shape[0]
        out = np.empty((N, H, W), dtype=np.uint8)

        # Fast path if already the right size
        if vid.shape[1] == H and vid.shape[2] == W:
            return vid

        for i in range(N):
            # PIL expects (W,H)
            im = Image.fromarray(vid[i], mode="L")
            im = im.resize((W, H), resample=Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.uint8)

        return out


    def process_bitmaps(self, positive, negative, 
                        timestamps_ns_t, properties, 
                        raw_frames, H, W):

        # get properties and bounds 
        raw_frames_fps = properties["fps"]
        raw_frames_dt = 1 / raw_frames_fps
        t_bounds = []
        
        for i in range(len(raw_frames) - 1):
            t_bounds.append([i * raw_frames_dt * 1e9, (i+1) * raw_frames_dt * 1e9])
        
        t_bounds = torch.Tensor(t_bounds)        
        B = self.preprocessing_params.get("num_bins", 5)

        
        # process the bitmaps to events                
        if self.preprocessing_type == EventRepresentation.EVENTS or self.preprocessing_type == EventRepresentation.SPIKE_TRACE_BANKS5:
            x,y,t, p = bitmaps_to_events(pos=positive.cpu(), neg=negative.cpu(), ts_ns=timestamps_ns_t)

        elif self.preprocessing_type == EventRepresentation.EVENTS_TIME_PLUS_POS:
            x, y, t, p = bitmaps_to_events(pos=positive.cpu(), neg=negative.cpu(), ts_ns=timestamps_ns_t,
                                            c_ns=self.preprocessing_params.get("c_ns", None),
                                            jitter_ns=self.preprocessing_params.get("jitter_ns", None),
                                            hashed_jitter=self.preprocessing_params.get("hashed_jitter", False))
                                
    
        event_grids = None
        if self.preprocessing_type == EventRepresentation.SPIKE_TRACE_BANKS5:
            if self.verbose:
                print("Using SPIKE_TRACE_BANK5 event representation")
            event_grids = spike_trace_bank5_from_rasters(pos=positive.to(self.device), neg=negative.to(self.device),
                                                            ts_ns=timestamps_ns_t.to(self.device), t_bounds=t_bounds.to(self.device),
                                                            tau_fast_ms=self.preprocessing_params.get("tau_fast_ms", 20),
                                                            tau_slow_ms=self.preprocessing_params.get("tau_slow_ms", 80),
                                                            tau_ts_ms=self.preprocessing_params.get("tau_ts_ms", 50),
                                                            asinh_k=self.preprocessing_params.get("asinh_k", 0.1))
        
        # gehrig / scheerlinck approach of event encoding
        else:        
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

                # (optional) hard safety: clamp x/y once if your ESIM can emit border values
                # H, W = frames.shape[-2], frames.shape[-1]
                # xk = xk.clamp_(0, W-1); yk = yk.clamp_(0, H-1)
                g = events_to_voxels(xk, yk, tk, pk, t0, t1, device=x.device, H=H, W=W, B=B)
                grids.append(g)

            event_grids = torch.stack(grids, dim=0)
        
        return event_grids, x, y, t, p
    
    
    def _read_events_or_create(self, folder, properties):

        paths = self._mk_paths(folder)
        properties_dirty = False

        have_cache = (os.path.exists(paths['grids'])
                      and os.path.exists(paths['outputs'])
                      and os.path.exists(paths['x'])
                      and os.path.exists(paths['y'])
                      and os.path.exists(paths['t'])
                      and os.path.exists(paths['p']))
        
        # check for valid cache : i.e. the silicon retina config matches
        if have_cache and "silicon_retina_config" in properties:
            cfg_dict = properties["silicon_retina_config"]
            cfg_loaded = SiliconRetinaConfig.from_dict(cfg_dict)
            if cfg_loaded != self.config:
                if self.verbose:
                    print(f"Config mismatch, forcing recompute for {folder}")
                have_cache = False
                # update the config in the params.json
                properties["silicon_retina_config"] = self.config.to_dict()  
                properties_dirty = True   
        else:
            have_cache = False

        # check if the same video is used (upsampled using optical flow (OF) vs fixed frame rate (FF))
        if have_cache and "used_OF" in properties:
            if properties["used_OF"] != self.use_upsampled_from_OF:
                have_cache = False
        else:
            have_cache = False

        # another check : the preprocessing and params for preprocessing has to match 
        if have_cache and "preprocessing_type" in properties:
            if properties["preprocessing_type"] != self.preprocessing_type.name:
                have_cache = False
                properties["preprocessing_type"] = self.preprocessing_type.name
                properties_dirty = True
        else:
            have_cache = False       

        if have_cache and "preprocessing_params" in properties:
            if properties["preprocessing_params"] != self.preprocessing_params:
                have_cache = False
                properties["preprocessing_params"] = self.preprocessing_params
                properties_dirty = True
        else:
            have_cache = False       


        if have_cache and not self.force_recomputing:
            # fast path: mmapped reads

            # (T, B, H, W), dtype=np.float16
            grids_np = _memmap_load(paths['grids'])

            # (T+1, H, W) uint8
            frames_np = _memmap_load(paths['frames'])

            # read outputs 
            outputs_np = _memmap_load(paths['outputs'])

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
                "frames": np.array(frames_np['raw_images']).astype(np.float32) / 255.,  # keep as numpy (cheap, contiguous)
                "outputs" : np.array(outputs_np),
                "events": events,
                "grids": event_grids,  # half precision on load
            }

            # optional: optical flow (unchanged from your logic)
            optical_flow_file = os.path.join(folder, "optical_flow.npz")
            if os.path.exists(optical_flow_file):
                optical_flow = np.load(optical_flow_file)["optical_flow"]
                
                if optical_flow.shape[0] != 0:
                    result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

            if self.verbose:
                print("READ CACHE")
            return result, properties, properties_dirty

        # ensure simulator
        self._ensure_simulator()
        del_cache_of_folder(folder)

        # slow path ( first time read and data cacheing)
        if self.use_upsampled_from_OF:
            frames = self._load_video_gray(os.path.join(folder, "video_upsampled_using_optical_flow.mp4"))
            timestamps = np.asarray(properties["timestamps_OF"], dtype=np.float64)
            images = properties["new_images_per_pair_OF"]  # list of counts per segment
            if self.verbose:
                print("read upsampled from OF")
        else:
            frames = self._load_video_gray(os.path.join(folder, f"video_upsampled_fps_eq_{self.input_framerate}.mp4"))  # (N, H, W), uint8
            timestamps = np.asarray(properties["timestamps_FF"], dtype=np.float64)
            images = properties["new_images_per_pair_FF"]  # list of counts per segment

        timestamps_ns = (timestamps * 1e9).astype("int64")
        
        # read the output frames and resize to the frame size
        if self.prediction_video_name:
            # load the .mp4 / .avi
            root, ext = os.path.splitext(self.prediction_video_name) 
            pred_video_path = os.path.join(folder, self.prediction_video_name)
        
            if ext.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                if os.path.exists(pred_video_path):
                    pred_frames = self._load_video_gray(pred_video_path,
                                                    fit_to_size=(frames.shape[2], frames.shape[1]))  # (N,H,W) uint8
                    outputs = pred_frames
            
            elif ext.lower() in [".npy", ".npz"]:
                if os.path.exists(pred_video_path):
                    loaded = np.load(pred_video_path)
                    if ext.lower() == ".npy":
                        outputs = loaded
                    else:
                        # .npz
                        if "outputs" in loaded:
                            outputs = loaded["outputs"]
                        else:
                            # take the first array
                            outputs = loaded[list(loaded.keys())[0]]                   
                

        else:
            outputs = frames.copy()
    
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

        # run the silicon retina simulator
        positive, negative = [], []
        T, H, W = frames.shape
        
        with torch.no_grad():
            self.simulator.reset()
            for t in range(1, T+1, 1):

                input = frames[t-1][None]
                p, n = self.simulator(input, t)
                positive.append((p.clone()[0]))
                negative.append((n.clone()[0]))

        positive = torch.stack(positive, dim=0)
        negative = torch.stack(negative, dim=0)
        
        # create the grids and events from the local configuration
        event_grids, x, y, t, p = self.process_bitmaps(positive, negative, timestamps_ns_t, 
                                                       properties, raw_frames, H, W)
        

        outputs = outputs.astype(np.float32) / 255.0
        
        # write caches
        # cast & move to CPU arrays only once, right before saving
        _save_cached_npy(paths["frames"], (255.*raw_frames).astype(np.uint8))
        _save_cached_npy(paths["grids"], event_grids.detach().to("cpu").to(torch.float16).numpy())
        _save_cached_npy(paths["outputs"], outputs)

        # Compact events to smaller dtypes on disk
        _save_cached_npy(paths["x"], x.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["y"], y.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
        _save_cached_npy(paths["t"], t.detach().to("cpu").to(torch.int64).numpy())
        _save_cached_npy(paths["p"], p.detach().to("cpu").to(torch.int8).numpy())

        # Return in the same shapes/types you expect
        result = {
            "frames": raw_frames,                                 # np.uint8
            "outputs" : outputs,
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
            
            if optical_flow.shape[0] != 0:
                result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

        # update the properties
        properties["silicon_retina_config"] = self.config.to_dict()  
        properties["preprocessing_type"] = self.preprocessing_type.name
        properties["preprocessing_params"] = self.preprocessing_params
        properties["used_OF"] = self.use_upsampled_from_OF
        properties_dirty = True

        if self.verbose:
            print("RECREATED CACHE")
        return result, properties, properties_dirty


    def __getitem__(self, idx):
        folder = self.data_files[idx]
        props = self.properties[idx]

        result, properties, properties_dirty = self._read_events_or_create(folder, props)

        if properties_dirty:
            self.properties[idx] = properties
            properties['silicon_retina_config']["HPF1_mode"] = int(properties['silicon_retina_config'].get("HPF1_mode", 0))
            properties['silicon_retina_config']["HPF2_mode"] = int(properties['silicon_retina_config'].get("HPF2_mode", 0))

            properties['silicon_retina_config']["blur1_mode"] = int(properties['silicon_retina_config'].get("blur1_mode", 0))
            properties['silicon_retina_config']["blur2_mode"] = int(properties['silicon_retina_config'].get("blur2_mode", 0))


            with open(os.path.join(self.data_files[idx], "params.json"), "w") as f:
                json.dump(properties, f, indent=4)
        
        
        
        return result



if __name__ == "__main__":
    import time
    import tqdm

    cfg = SiliconRetinaConfig()    
    cfg.H = 180 
    cfg.W = 320

    HW = cfg.H * cfg.W

    ds = SiliconRetinaEventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_upsampled"),
                                                prediction_video_name="outputs.npz",
                                                preprocessing_type=EventRepresentation.EVENTS_TIME_PLUS_POS,
                                                preprocessing_params={"c_ns": 0.25 * 0.005 * 1e9 / HW, "hashed_jitter":True,
                                                                        "num_bins" : 5},
                                                input_framerate=200,
                                                config=cfg,
                                                force_recomputing=False)

    
    out_dir = os.path.join("visualizations", "retina_dhf1k")
    os.makedirs(out_dir, exist_ok=True)
    
    visualizations = 20

    for i in tqdm.tqdm(range(visualizations)):
        
        datapoint = ds[i]
        imgs, events, grids, outputs = datapoint["frames"], datapoint["events"], datapoint["grids"], datapoint['outputs']

        x,y, t, p = events['x'], events['y'], events['t'], events['p']

        optical_flow_on = False
        if "optical_flows" in datapoint:
            optical_flow = datapoint["optical_flows"]
            optical_flow_on = True

        # visualize a datapoint    
        render_events_video(datapoint, output_path=os.path.join(out_dir, f"retina_vis_{i:03d}.mp4"),
                            flows_on=optical_flow_on,
                            invert_rgb=True,
                            show_outputs=True,
                            overlay_outputs = True)

