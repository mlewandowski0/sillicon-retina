import os 
import argparse
import imageio as iio
import shutil
from tqdm import tqdm
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="Path to the folder containing the saliency map dataset to be reordered")
parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where the reordered dataset will be saved")

args = parser.parse_args()

max_val = 701
videos_path, annotations_path = os.path.join(args.input_path, "video-002", "video"), os.path.join(args.input_path, "annotation-003", "annotation")

video_list = sorted([os.path.join(videos_path, x) for x in os.listdir(videos_path)])
annotations_list = sorted([os.path.join(annotations_path, x) for x in os.listdir(annotations_path)])

os.makedirs(args.output_path, exist_ok=True)

for i in tqdm(range(1, max_val)):
    video_name = str(i).zfill(3)
    folder_name = str(i).zfill(4)
    
    out_folder = os.path.join(args.output_path, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    
    video_path = os.path.join(videos_path, f"{video_name}.AVI")
    #if not os.path.exists(video_path):
    #        print(f"Video {video_name} not found, skipping...")
    #        continue
        
    annotation_path = os.path.join(annotations_path, f"{folder_name}")
    #if not os.path.exists(annotation_path):
    #    print(f"Annotation {annotation_path} not found, skipping...")
    #    continue

    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    

    
    fixation_path = os.path.join(annotation_path, "fixation")
    maps_path = os.path.join(annotation_path, "maps")

    shutil.copy(video_path, os.path.join(out_folder, f"input.AVI"))
    
    fixation_frames = 0     
    with iio.get_writer(os.path.join(out_folder, "fixation.mp4"), fps=30,macro_block_size=1) as writer:
        fixation_list = sorted([os.path.join(fixation_path, x) for x in os.listdir(fixation_path) if x.endswith(".png") or x.endswith(".jpg")])
        for fixation in fixation_list:
            img = iio.imread(fixation)
            writer.append_data(img) 
            fixation_frames += 1  

    maps_frames = 0
    with iio.get_writer(os.path.join(out_folder, "maps.mp4"), fps=30, macro_block_size=1) as writer:
        fixation_list = sorted([os.path.join(maps_path, x) for x in os.listdir(maps_path) if x.endswith(".png") or x.endswith(".jpg")])
        for fixation in fixation_list:
            img = iio.imread(fixation)
            writer.append_data(img)
            maps_frames += 1   

    
    params = {"fps" : fps, "input_frame_count": frame_count, "fixation_frame_count": fixation_frames, "maps_frame_count": maps_frames}
    with open(os.path.join(out_folder, "params.json"), "w") as f:
        json.dump(params, f)
        
    if frame_count != fixation_frames or frame_count != maps_frames:
        print(f"Warning: Frame count mismatch in video {video_name}: video frames = {frame_count}, fixation frames = {fixation_frames}, maps frames = {maps_frames}")