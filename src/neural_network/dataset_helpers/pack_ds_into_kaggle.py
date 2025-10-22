import os
import shutil
from tqdm import tqdm

upsampled_train_path = os.path.join(".", "src", "neural_network", "data", "upsampled_train")
upsampled_val_path = os.path.join(".", "src", "neural_network", "data", "upsampled_validation")

dest_upsampled_train = os.path.join("dataset_packed", "upsampled_train")
dest_upsampled_val   = os.path.join("dataset_packed", "upsampled_validation")

os.makedirs(dest_upsampled_val, exist_ok=True)

def move(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    job= []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if not file.startswith("cache"):
                full_path = os.path.join(root, file)

                destination = full_path[len(source_folder)+1:]
                destination = os.path.join(dest_upsampled_train, destination)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                job.append((full_path, destination))

    pbar = tqdm(sorted(job))
    total = 0
    for i, (src, dst) in enumerate(pbar):
        shutil.copy(src, dst)
        total += os.path.getsize(dst)

        if i % 10 == 0:
            pbar.set_postfix({'GB moved ': f"{(total/(1000**3)):.4f} GB"})

move(upsampled_train_path, dest_upsampled_train)
move(upsampled_val_path, dest_upsampled_val)
