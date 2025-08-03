from torch.utils.data import Dataset
import os 
import torch 
import pandas as pd 
import numpy as np
import imageio
import time 
import cProfile, pstats
import tqdm
from src.neural_network.utils.measure import profile_dataset

class ECOCO_dataset(Dataset):
    def __init__(self, root, split, limit = None):
        super().__init__()
        self.root = root
        self.split = split
        self.path = os.path.join(self.root, self.split)
        self.paths = [os.path.join(self.path, x) for x in os.listdir(self.path)]
        self.limit = limit

        if self.limit is not None:
            self.paths = self.paths[:self.limit]

    def __len__(self):
        return len(self.paths)

    def __load__folder__(self, folder):
        items = []
        paths = sorted(os.listdir(folder))

        for f in paths:
            path = os.path.join(folder, f)

            if f.endswith(".png"):
                items.append(imageio.imread(path))

            elif f.endswith(".npy"):
                items.append(np.load(path))

        return np.array(items)

    def __load_txt__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        return [[float(v) for v in line.strip().split()] for line in lines]
        

    def __getitem__(self, index, with_path=False):
        frames = self.__load__folder__(os.path.join(self.paths[index], "frames"))
        frame_timesteps = os.path.join(self.paths[index], "frames", "timestamps.txt")
        frame_timesteps = self.__load_txt__(frame_timesteps)
        
        grids = self.__load__folder__(os.path.join(self.paths[index], "VoxelGrid-betweenframes-5"))
        
        grid_timesteps = os.path.join(self.paths[index], "VoxelGrid-betweenframes-5", "timestamps.txt")
        grid_timesteps = self.__load_txt__(grid_timesteps)
        
        grid_boundary_timesteps = os.path.join(self.paths[index], "VoxelGrid-betweenframes-5", "boundary_timestamps.txt")
        grid_boundary_timesteps = self.__load_txt__(grid_boundary_timesteps)
        
        ret = {"frames" : frames, 
                "grids" : grids,
                "frame_timesteps" : np.array(frame_timesteps),
                "grid_timesteps" : np.array(grid_timesteps),
                "grid_boundary_timesteps" : np.array(grid_boundary_timesteps)}

        if with_path:
            ret["path"] = self.path[index]
        
        return ret
                    

if __name__ == "__main__":
    import tqdm 
    dataset = ECOCO_dataset(root="data/ecoco_depthmaps_test", split="train")
    
    item = dataset[0]
    print(item.keys())
    print(item["frames"].shape)
    print(item["grids"].shape)
    print("frame timesteps : ", item["frame_timesteps"])
    print("grid timesteps : ", item["grid_timesteps"])
    print("grid boundary timesteps : ", item["grid_boundary_timesteps"])
    
    for i in range(5):
        _ = dataset[i]
        
    profile_dataset(dataset=dataset, data_points=20)