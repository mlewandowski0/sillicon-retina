import numpy as np 
import os 
import cv2 
import imageio
from collections import defaultdict
from datetime import datetime
import glob 
import os

# text properties 
org = (10, 30)                              # bottom-left corner of the text string in the image
font = cv2.FONT_HERSHEY_SIMPLEX             # built‑in font
font_scale = 0.7                            # font scale factor
color = (255, 0, 0)                           # text color in BGR (here: black)
thickness = 2                               # thickness of the text strokes
line_type = cv2.LINE_AA                     # anti‑aliased line
per_row = 5
now = datetime.now()
formatted = now.strftime("%B %d – %H:%M")

d = defaultdict(list)

# remove the previous output file 
for filepath in glob.glob("output*"):
    if os.path.isfile(filepath):
        os.remove(filepath)


for root, folders, files in os.walk("out"):
    for file in files:
        if file.endswith(".png") or file.endswith(".npy"):
            path = os.path.join(root, file)
            d[root].append(path)
            
for k,v in d.items():
    a = [ ((el.split("/")[-1])[:-4], el) for el in v]
    els = sorted(a, key=lambda x: int(x[0]))
    els = [el[1] for el in els]
    d[k] = els


number_of_frames = len(d["out/fired"])    
order = ["imgs", "LPF1", "HPF1", "LPF2", "BLUR2", "I_OPL", "VBIP", "VBIP_LPF", "HPF2", "IGang", "VLIF","fired"]
is_scaled = [False, False, True, True, True, True, True, True, True, True, True, True]

print("ASD")
with imageio.get_writer(f"output_{formatted}.mp4", fps=30, codec="libx264") as writer:
    for i in range(number_of_frames):
        rows = []
        cnt = 1
        current_row = []
        
        for scaled_flag, k in zip(is_scaled, order): 
            keyword = "out/" + k
            if keyword not in d:
                continue
            
            path = d[keyword][i]
            
            if path.endswith(".npy"):
                f = np.load(path)
                if scaled_flag:
                    f = (127.5 + 127.5*np.clip(f, -1., 1.)).astype(np.uint8)
                else:
                    f = (255 * np.clip(f, 0., 1.)).astype(np.uint8)
            else:
                f = cv2.imread(path)
            
            if f.ndim == 2:  # if grayscale, convert to BGR
                f = np.repeat(f[:, :, np.newaxis], 3, axis=2)
            cv2.putText(f, k, org, font, font_scale, color, thickness, line_type)
            current_row.append(f)
        
            if cnt % per_row == 0:
                rows.append(np.concatenate(current_row, axis=1))
                current_row = []
            cnt += 1
        
        missing = per_row - len(current_row)
        rows.append(np.concatenate(current_row + [np.zeros_like(current_row[0])]*missing, axis=1))
        
        writer.append_data(np.concatenate(rows, axis=0))    
        
    