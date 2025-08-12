from src.neural_network.dataloaders.EventDatasetFromUpsampled import EventDatasetFromUpsampled
from src.neural_network.utils.event_utils import render_events_video
from tqdm import tqdm
import os
import shutil


if __name__ == "__main__":

    limit = 50
    visualization_folder = "visualizations"

    ds = EventDatasetFromUpsampled(directory=os.path.join("src", "neural_network", "data", "upsampled_train"),
                                   mode="event",
                                   B=5,
                                   force_recomputing=True)

    pbar = tqdm(range(min(len(ds) ,limit)))
    for i in pbar:
        datapoint = ds[i]
        folder = ds.data_files[i]


    ds.force_recomputing = False

    os.makedirs(visualization_folder, exist_ok=True)

    for i in pbar:
        datapoint = ds[i]
        folder = ds.data_files[i]
        render_events_video(datapoint, os.path.join(folder, f"visualization.mp4"),fps=10)

        shutil.copy(os.path.join(folder, f"visualization.mp4"), os.path.join(visualization_folder, f"{i}.mp4"))


    shutil.make_archive(visualization_folder, format="zip", root_dir=visualization_folder)