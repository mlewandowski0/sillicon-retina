import os
from src.neural_network.dataloaders.VideoDatasetUpsampling import VideoDatasetUpsampling, \
    VideoDatasetUpsamplingCacheToFolder
from src.neural_network.dataloaders.VideoDataset import VideoDatasetFromPictures
from src.neural_network.dataloaders.VideoDatasetFromPairs import VideoDatasetFromVideoPairGlobal
import imageio
import numpy as np 

def write_sequence_to_video(images, filename, fps=30):
    with imageio.get_writer(filename, fps=fps) as writer:
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
            # is of shapes
            writer.append_data(new_data)


def upsample():
    video_val = VideoDatasetFromVideoPairGlobal(dirpath=os.path.join("src", "neural_network", "data", "saliency_dataset_processed"),
                                                   split=None,
                                                   global_fps=30,
                                                   input_video="input.AVI",
                                                   output_video="maps.mp4",
                                                   suffix=None)

    upsampling_ds_val = VideoDatasetUpsampling(video_val,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=False,
                                           new_framerate=200,
                                           debug = False)


    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds_val,
                                        output_path=os.path.join("src", "neural_network", "data", "saliency_dataset_upsampled"))



if __name__ == "__main__":
    upsample()