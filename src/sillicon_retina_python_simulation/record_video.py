import cv2
import os
import time

# Parameters
NUM_FRAMES = 200
FPS        = 25
W, H       = 256, 256
OUT_DIR    = "frames"

# Make sure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Open the default camera (use a filename instead of 0 to read a video file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open video source")

# Optional: set camera resolution (may not be supported on all devices)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS,          FPS)

# Time between frames
dt = 1.0 / FPS

print(f"Recording {NUM_FRAMES} frames at {FPS} FPS → saving to ./{OUT_DIR}/")

for i in range(NUM_FRAMES):
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i} capture failed, stopping.")
        break

    # Resize to exactly W×H
    frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

    # Save
    filename = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
    cv2.imwrite(filename, frame_resized)

    # Wait to match target FPS
    t_elapsed = time.time() - t_start
    t_sleep   = dt - t_elapsed
    if t_sleep > 0:
        time.sleep(t_sleep)

cap.release()
print("Done.")
