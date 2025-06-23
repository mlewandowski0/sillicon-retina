import cv2
import glob
import os
import re

# --- Parameters ---
image_folder = 'E:\\scamp\\scamp5d_host\\bin\\record_20250623_1635_57'  # Change this!
output_video = 'output.mp4'
fps = 30  # You can change FPS as you like

# --- Helper: extract number from filename for sorting ---
def extract_number(filename):
    match = re.search(r'display_10_0(\d+)\.BMP', filename)
    return int(match.group(1)) if match else -1

# --- Get image list ---
image_files = glob.glob(os.path.join(image_folder, 'display_10_0*.BMP'))
image_files.sort(key=extract_number)

# --- Read first image to get size ---
first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape
frame_size = (width, height)

# --- Define video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'avc1' for MP4
video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# --- Write frames ---
for idx, filename in enumerate(image_files):
    frame = cv2.imread(filename)
    if frame is None:
        print(f"Warning: couldn't read {filename}")
        continue
    video_writer.write(frame)
    if idx % 1000 == 0:
        print(f"Processed {idx}/{len(image_files)} frames")

video_writer.release()
print("Done! Video saved as", output_video)
