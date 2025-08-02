from torch.utils.data import Dataset
import os 
import torch 
import pandas as pd 
import numpy as np
import imageio
import tqdm
from .ECOCO_dataset import ECOCO_dataset
import json
from utils.measure import profile_dataset

class ECOCO_dataset_file_cached(Dataset):
    def __init__(self, root, split, cache_folder, limit=None):
        super().__init__()
        self.base = ECOCO_dataset(root, split, limit)
        self._cache_folder = cache_folder
        self.datapoints = []
        self.metadata = {"root": root, "split": split, "limit": limit}
        self._cache_data()

    
    def _cache_data(self):
        run_computation = True
        metadata_file = os.path.join(self._cache_folder, "metadata.json")
        
        
        if not os.path.exists(self._cache_folder):
            os.makedirs(self._cache_folder)
        
        else:
            if os.path.exists(metadata_file):
                read_metadata = None
                with open(metadata_file, "r", encoding="utf-8") as f:
                    read_metadata = json.load(f)
                
                if read_metadata == self.metadata:
                    run_computation = False
                    self.datapoints = sorted([os.path.join(self._cache_folder, f) for f in os.listdir(self._cache_folder) if f.endswith(".npz")])
                    memory_spend_for_cache = sum(os.path.getsize(path) for path in self.datapoints)
                    print(f"Cache already exists with {len(self.datapoints)} items, Memory used: {memory_spend_for_cache / (1024 * 1024):.2f} MB")                    
                else:
                    print("Metadata has changed, recomputing cache...")
                    run_computation = True
        
        if not run_computation:
            return 
        print("Caching ECOCO dataset metadata...")        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print("Saved  to ", metadata_file)
        print(self.metadata)
        print("Caching ECOCO dataset files...")       
        pb = tqdm.tqdm(total=len(self.base), desc="Caching ECOCO dataset files")
        memory_spend_for_cache = 0
        for i in range(len(self.base)):
            item = self.base[i]
            cache_path = os.path.join(self._cache_folder, f"item_{i:04d}.npz")
            self.datapoints.append(cache_path)
            np.savez_compressed(cache_path, **item)
            memory_spend_for_cache += os.path.getsize(cache_path)
            pb.update(1)
            pb.set_description("Caching ECOCO dataset files, Memory used: {:.2f} MB".format(memory_spend_for_cache / (1024 * 1024)))
        pb.close()

    def __getitem__(self, index):
        return np.load(self.datapoints[index])
        
    def __len__(self):
        return len(self.datapoints)
    

if __name__ == "__main__":
    # Example usage
    dataset = ECOCO_dataset_file_cached(root="data/ecoco_depthmaps_test", split="train", cache_folder="data/ecoco_cache", limit=100)
    print(f"Dataset length: {len(dataset)}")
    item = dataset[0]
    print(item.keys())
    print("frames shape: ", item["frames"].shape)
    print("sample frames : ", item["frames"])
    print("grids shape: ", item["grids"].shape)
    print("sample grids : ", item["grids"])
    print("frame timesteps : ", item["frame_timesteps"])
    print("grid timesteps : ", item["grid_timesteps"])
    print("grid boundary timesteps : ", item["grid_boundary_timesteps"])
    

    
    profile_dataset(dataset)