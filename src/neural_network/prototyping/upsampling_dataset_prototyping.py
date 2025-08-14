import os, tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

import os 
from src.neural_network.dataloaders.VideoDatasetUpsampling import VideoDatasetUpsampling
from src.neural_network.dataloaders.VideoDataset import VideoDatasetFromPictures
import imageio
import numpy as np 

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



if __name__ == "__main__":
    idx = 0
    debug = True
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    original_images = video_ds[idx]

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=False,
                                           new_framerate=200,
                                           debug = debug)

    original_images = video_ds[idx]
    x = upsampling_ds[idx]
    images, deltas, timestamps = x[0], x[1], x[2]


    write_sequence_to_video(images, os.path.join("src", "prototyping","upsampling_fixed_framerate.mp4"))
    write_sequence_and_original_to_video(upsampled=images, timestamps=timestamps,
                                         original=original_images["images"], original_fps=video_ds.fps,
                                         filename=os.path.join("src", "prototyping", "upsampling_fixed_framerate_comparison.mp4"))
