import os 
import shutil
import numpy as np 
import argparse 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, help="Source directory containing saliency upsampled outputs")
    ap.add_argument("--dest_dir", required=True, help="Destination directory to copy outputs to")
    args = ap.parse_args()
    
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    
    dest_folders = [os.path.join(dest_dir, x) for x in os.listdir(dest_dir)]
    
    
    for folder in dest_folders:
        print(folder)
     