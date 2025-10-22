import os 
import shutil
from tqdm import tqdm
import json

def modify_metadata(filepath, new_number):
    with open(filepath, 'r') as f:
        data = json.load(f)
    data['number_of_videos'] = new_number
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

no_files_train = 30
no_files_valid = 10

ds_path = os.path.join("data","saliency_dataset_upsampled")
output_path_train = os.path.join("data","saliency_small_subset_train")
os.makedirs(output_path_train, exist_ok=True)

output_path_valid = os.path.join("data","saliency_small_subset_valid")
os.makedirs(output_path_valid, exist_ok=True)


folders = sorted([os.path.join(ds_path, path) for path in os.listdir(ds_path)])
datapoints = [el for el in folders if os.path.basename(el).startswith("video_")]

shutil.copy(os.path.join(ds_path, "metadata.json"), os.path.join(output_path_train, "metadata.json"))
modify_metadata(os.path.join(output_path_train, "metadata.json"), no_files_train)
for i in tqdm(range(no_files_train)):    
    shutil.copytree(datapoints[i],os.path.join(output_path_train, os.path.basename(datapoints[i])))
    
shutil.copy(os.path.join(ds_path, "metadata.json"), os.path.join(output_path_valid, "metadata.json"))
modify_metadata(os.path.join(output_path_valid, "metadata.json"), no_files_valid)
for i in tqdm(range(no_files_valid)):    
    shutil.copytree(datapoints[no_files_train + i],os.path.join(output_path_valid, f"video_{str(i).zfill(5)}"))