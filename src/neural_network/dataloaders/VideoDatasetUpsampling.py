import os, sys, site, glob

# Build LD_LIBRARY_PATH to point at pip CUDA libs inside this venv
libroots = []
for pkg, sub in [
    ("nvidia/cuda_runtime", "lib"),
    ("nvidia/cublas",       "lib"),
    ("nvidia/cudnn",        "lib"),
    ("nvidia/cufft",        "lib"),
]:
    for sp in site.getsitepackages():
        path = os.path.join(sp, pkg, sub)
        if os.path.isdir(path):
            libroots.append(path)

# Prepend (don’t overwrite) so TF finds the right .so’s
os.environ["LD_LIBRARY_PATH"] = ":".join(libroots + [os.environ.get("LD_LIBRARY_PATH","")])

# Usual quality-of-life flags
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # <-- import FIRST
print("TF build:", tf.__version__, tf.sysconfig.get_build_info())
print("TF GPUs:", tf.config.list_physical_devices('GPU'))

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .VideoDataset import VideoDatasetFromPictures
from .interpolator import Interpolator
import tqdm
import imageio
import imageio.v3 as iio                      # ≥ 2.30, pulls imageio-ffmpeg

def write_sequence_to_video(images, filename):
    with imageio.get_writer(filename, fps=30) as writer:
        for i in range(len(images)):
            writer.append_data(np.clip(255. * images[i], 0, 255.).astype(np.uint8)  )


def write_sequence_and_original_to_video(upsampled, timestamps, original, original_fps, filename):
    i = 0
    j = 0
    dt = 1 / original_fps
    with imageio.get_writer(filename, fps = original_fps) as writer:
        for i, t in enumerate(timestamps):
            if t > (j+1) * dt:
                j += 1

            upsampled_new = np.clip(255 *  upsampled[i], 0, 255).astype(np.uint8)
            new_data = np.concatenate([upsampled_new, np.repeat(original[j][..., None], 3, axis=2)  ], axis=1)
            # is of shape
            writer.append_data(new_data)

def to_nhwc3_batched(img):
    """
        img : np.ndarray, shape (H, W) or (H, W, 1) or (H, W, 3) or already batched (1, H, W, C)
        returns : float32 in [0, 1] shape (1, H, W, 3)
    """
    
    x = img 
    
    # remove batch if already present with size 1 ( safe-guard )
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
        
    # ensure last 3 channels
    
    # (H, W)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
        
    elif x.ndim == 3 and x.shape[-1] == 1: # (H, W, 1)
        x = np.repeat(x, 3, axis=-1)
        
    elif x.ndim == 3 and x.shape[-1] == 3: # (H, W, 3)
        pass

    else:
        raise ValueError(f"Unexpected frame shape {x.shape}; expected (H,W),(H,W,1),(H,W,3).")
    
    x = x.astype(np.float32) 
    if x.max() > 1.0:
        x = x / 255.0
        
    x = x[None, ...]
    return x
    
class VideoDatasetUpsampling(Dataset):

    def __init__(self,
                 videoDataset,
                 model_path : str,
                 upsample_to_events : bool = False,
                 new_framerate : Optional[int] = None,
                 debug : bool = False,
                 align: Optional[int] = None) -> None:

        super(VideoDatasetUpsampling, self).__init__()
        self.videoDataset = videoDataset
        self.model_path = model_path
        self.upsample_to_events = upsample_to_events
        self.new_framerate = new_framerate
        self.debug = debug

        if not self.upsample_to_events and new_framerate is None:
            raise Exception("Upsampling to events was specified to False and new framerate was not provided !")


        self.interpolation = Interpolator(model_path=model_path,
                                          align=align)


    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolation.interpolate(I0, I1, dt)
        was_first = num_bisections < 0

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))

            if num_bisections == 0:
                return [image[0]], F_0_1, [(t0 + t1) / 2]

        left_images, left_of, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_of, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)

        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        of_return = []
        if was_first:
            of_return = F_0_1

        return images, of_return, timestamps

    def _upsample_optical_flow(self, datapoint):

        images = datapoint['images']
        fps = datapoint['fps']

        all_images = []
        all_timestamps = []
        optical_flows = []

        if self.debug:
            pbar = tqdm.tqdm(range(len(images)-1))
        else:
            pbar = range(len(images)-1)

        dt = 1. / fps
        for i in pbar:
            current_image, next_image = images[i], images[i + 1]
            t0, t1  = i * dt, (i + 1) * dt

            current_image = to_nhwc3_batched(images[i])            # (1,H,W,3)
            next_image    = to_nhwc3_batched(images[i + 1])        # (1,H,W,3)

            total_frames, optical_flow, total_timestamps = self._upsample_adaptive(current_image,
                                                                                 next_image,
                                                                                 t0,
                                                                                 t1)
            total_frames = [current_image[0]] + total_frames
            timestamps = [t0] + total_timestamps

            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]

            if self.debug:
                print(f"Generated : {len(total_frames)}")
            all_images.append(total_frames)
            all_timestamps.append(timestamps)
            optical_flows.append(optical_flow)

        images_per_patch = [len(a) for a in all_images]
        return np.array(images), np.concatenate(all_images), np.array(images_per_patch), np.concatenate(all_timestamps), np.concatenate(optical_flows)

    def _upsample_to_fixed_framerate(self, datapoint, new_framerate):
        images = datapoint['images']
        fps = datapoint['fps']

        all_images = []
        all_timestamps = []
        optical_flows = []

        new_dt = 1.0 / new_framerate
        img_dt = 1.0 / fps

        T = len(images)
        video_time = img_dt * T

        if self.debug:
            pbar = tqdm.tqdm(np.arange(0, video_time, new_dt))
        else:
            pbar = np.arange(0, video_time, new_dt)

        i = 0
        image_deltas = []
        prev = 0

        for time in pbar:

            if time >= (i + 1) * img_dt:
                i += 1
                image_deltas.append(prev)
                prev = 0

            original_frame_time = i * img_dt
            next_original_frame_time = (i + 1) * img_dt

            dt_val = (time - original_frame_time) / (next_original_frame_time - original_frame_time)
            dt_rel = np.array([dt_val], dtype=np.float32)

            prev += 1

            # preprocess the images to fit the shape
            I0 = to_nhwc3_batched(images[i])                  # (1,H,W,3) float32
            I1 = to_nhwc3_batched(images[i + 1] if i + 1 < T else images[i])

            if time == original_frame_time:
                all_images.append(I0)
                all_timestamps.append(time)
                continue

            interpolated_image, F_0_1, F_1_0 = self.interpolation.interpolate(I0, I1, dt_rel)
            all_images.append(interpolated_image)
            all_timestamps.append(time)

        image_deltas.append(prev)

        return np.array(images), np.concatenate(all_images, axis=0), np.array(image_deltas), np.array(all_timestamps), optical_flows
        
    def __len__(self): 
        return len(self.videoDataset)

    def __getitem__(self, idx):
        datapoint = self.videoDataset[idx]

        if self.upsample_to_events:
            return self._upsample_optical_flow(datapoint)
        else:
            return self._upsample_to_fixed_framerate(datapoint, self.new_framerate)

class VideoDatasetUpsamplingCacheToFolder(Dataset):
    def __init__(self, upsamplingDataset : VideoDatasetUpsampling,
                 output_path, save_mp4=True, bit_depth=8):
        self.upsamplingDataset = upsamplingDataset
        self.output_path = output_path
        self.save_mp4 = save_mp4
        self.bit_depth = bit_depth
        self.data = []
        self._cache()

    def _write_mp4(self, frames_f32, out_path, fps):
        """Save float32 [0,1] RGB video to MP4, 8- or 10-bit."""
        h, w = frames_f32.shape[1:3]

        if self.bit_depth == 8:  # ---- H.264 8-bit ----
            frames_u8 = (frames_f32 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

            writer = imageio.get_writer(
                out_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',  # 8-bit 4:2:0
                macro_block_size=None,  # disable automatic resizing
                # any ffmpeg option can go here, e.g.   ffmpeg_params=['-crf','18']
            )
            for f in frames_u8:
                writer.append_data(f)
            writer.close()

        elif self.bit_depth == 10:  # ---- HEVC Main10 ----
            frames_u16 = (frames_f32 * 1023.0 + 0.5).clip(0, 1023).astype(np.uint16)

            writer = imageio.get_writer(
                out_path,
                fps=fps,
                codec='libx265',
                pixelformat='yuv420p10le',  # 10-bit 4:2:0 (Main10)
                macro_block_size=None,
                input_params=['-pix_fmt', 'rgb30le']  # tell ffmpeg what we feed
            )
            for f in frames_u16:
                writer.append_data(f)
            writer.close()
        else:
            raise ValueError("bit_depth must be 8 or 10")

    def _cache(self):

        upsample_to_events = self.upsamplingDataset.upsample_to_events
        new_framerate = self.upsamplingDataset.new_framerate

        run_computation = True
        metadata = {"number_of_videos" : len(self.upsamplingDataset),
                    "bit_depth" : self.bit_depth,
                    "model_path" : self.upsamplingDataset.model_path,
                    "optical_flow_saved" : True,
                    "optical_flow_name" : "optical_flow.npz"
                    }

        if upsample_to_events:
            metadata["upsampling_OF_saved"] = True
        else:
            metadata["upsampling_to_fixed_fps"] = True
            metadata["new_framerate"] = new_framerate

        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        if os.path.exists(os.path.join(self.output_path, "metadata.json")):
            read_metadata = {}
            print("Found already existing metadata file")
            with open(os.path.join(self.output_path, "metadata.json"), "r") as f:
                read_metadata = json.load(f)

            # if the metadata matches the current metadata, skip computation ( if current metadata is a subset
            # of the read metadata )
            is_subset = True
            for k, v in metadata.items():
                if k in read_metadata:
                    if metadata[k] != read_metadata[k]:
                        is_subset = False
                        break
                else:
                    is_subset = False
                    break
            run_computation = not is_subset

            for k, v in read_metadata.items():
                if k not in metadata:
                    metadata[k] = v


        if run_computation:
            pbar = tqdm.tqdm(range(len(self.upsamplingDataset)))


            with open(os.path.join(self.output_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

            for i in pbar:
                raw_images, upsampled_images, images_per_patches, timestamps, optical_flows = self.upsamplingDataset[i]

                # shape (512, 180, 240, 3), dtype=float32

                params = { "raw_images" : "raw_images.npz",
                          "fps" : self.upsamplingDataset.videoDataset.fps,
                          "optical_flow_path" : "optical_flow.npz"}

                if upsample_to_events:
                    params['new_images_per_pair_OF'] =  images_per_patches.tolist()
                    params["timestamps_OF"] = timestamps.tolist()

                else:
                    params['new_images_per_pair_FF'] =  images_per_patches.tolist()
                    params["timestamps_FF"] = timestamps.tolist()

                root_folder = os.path.join(self.output_path, f"video_{i:05d}")
                os.makedirs(root_folder, exist_ok=True)

                if os.path.exists(os.path.join(root_folder, "params.json")):
                    r = json.load(open(os.path.join(root_folder, "params.json")))

                    for k, v in r.items():
                        if k not in params:
                            params[k] = v

                with open(os.path.join(root_folder, "params.json"), "w") as f:
                    json.dump(params, f, indent=4)

                if upsample_to_events:
                    mp4_path = os.path.join(root_folder, f"video_upsampled_using_optical_flow.mp4")
                else:
                    mp4_path = os.path.join(root_folder, f"video_upsampled_fps_eq_{new_framerate}.mp4")

                self._write_mp4(upsampled_images, mp4_path, fps=params["fps"])

                np.savez_compressed(os.path.join(root_folder, params["raw_images"]),
                                    raw_images = raw_images)

                if not os.path.exists(os.path.join(root_folder, params["optical_flow_path"])):
                    np.savez_compressed(os.path.join(root_folder, params["optical_flow_path"]),
                                    optical_flow=optical_flows)




if __name__ == "__main__":

    idx = 0
    debug = False
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = debug)

    original_images = video_ds[idx]
    x = upsampling_ds[idx]
    images, images_per_patches, timestamps, optical_flows = x
    np.savez_compressed(os.path.join("src", "prototyping", "opticalflow.npz"), optical_flows=optical_flows)

    raw_images = []

    prev = 0
    for delta in images_per_patches:
        raw_images.append(images[prev])
        prev = prev + delta

    raw_images.append(images[-1])

    print("shape of original_images ",len(original_images['images']))
    print(f"shape of images : {images.shape}")
    print(f"shape of timestamps : {timestamps.shape}")
    print(f"images_per_patches : {images_per_patches}")
    print(f"shape of images_per_patches : {images_per_patches.shape}")
    print("sum of images_per_patches : ",np.sum(images_per_patches))
    print("Shape of Optical Flow : ", optical_flows.shape)

    write_sequence_to_video(images, os.path.join("src", "prototyping","videoDatasetUpsamplingHighFPSOnly.mp4"))
    write_sequence_to_video(raw_images, os.path.join("src", "prototyping","videoDatasetUpsamplingOriginal.mp4"))

    write_sequence_and_original_to_video(upsampled=images, timestamps=timestamps,
                                         original=original_images["images"], original_fps=video_ds.fps,
                                         filename="videoDatasetUpsamplingHighFPSStiched.mp4")

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_train"))

    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="validation",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = debug)

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_validation"))
