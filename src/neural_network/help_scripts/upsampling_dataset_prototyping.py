import os
from src.neural_network.dataloaders.VideoDatasetUpsampling import VideoDatasetUpsampling, \
    VideoDatasetUpsamplingCacheToFolder
from src.neural_network.dataloaders.VideoDataset import VideoDatasetFromPictures
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


def  make_fixed_framerate_validation():
    video_ds_val = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="validation",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds_val = VideoDatasetUpsampling(video_ds_val,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=False,
                                           new_framerate=200,
                                           debug = False)


    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds_val,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_validation"))


def make_fixed_framerate_train():
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=False,
                                           new_framerate=200,
                                           debug = False)

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_train"))


def make_OF_val():
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="validation",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = False)

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_validation"))


def make_OF_train():
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = False)

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_train"))


def cache_upsampling_new_framerate():
    idx = 0
    debug = True

    make_fixed_framerate_validation()
    make_fixed_framerate_train()


def cache_upsampling_optical_flow():
    make_OF_val()
    make_OF_train()



def quciker_check():
    make_OF_val()
    make_fixed_framerate_validation()

    make_OF_train()
    make_fixed_framerate_train()


if __name__ == "__main__":

    # cache_upsampling_optical_flow()
    # cache_upsampling_new_framerate()
    quciker_check()