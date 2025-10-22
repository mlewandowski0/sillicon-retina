import os 
import shutil
from tqdm import tqdm

inp_root = os.path.join("data", "saliency_dataset_processed")
out_root = os.path.join("data","saliency_dataset_upsampled")

inps = os.listdir(inp_root)
outs = os.listdir(out_root)

for folder in tqdm(inps):
    inp_folder = os.path.join(inp_root, folder)
    
    out_val = str(int(folder) - 1).zfill(5)    
    out_folder = os.path.join(out_root, f"video_{out_val}")

    print(f"Copying from {inp_folder} to {out_folder}")
    shutil.copy(os.path.join(inp_folder, "maps.mp4"), os.path.join(out_folder, "maps.mp4"))
    