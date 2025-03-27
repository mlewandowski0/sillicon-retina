import cv2 
import numpy as np 
import time 
import os 

# define the desired frames per second and caluclate the time between frames
fps = 30
interval = 1.0 / fps
frames = 300
path = "dataset"

# open a default camera 
cap = cv2.VideoCapture(0)

# Attempty to set the capture FPS to 30 (may depend on your camera)
cap.set(cv2.CAP_PROP_FPS, fps)

frame_counter = 0

os.makedirs(path, exist_ok=True)

while frame_counter < frames: 
    start_time = time.time()    
    
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break 
    

    # Convert the freame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # resize the frame to 256x256 pixels
    resized_frame = cv2.resize(gray_frame, (256, 256))
    
    # Save the frame as a BMP image 
    filename = os.path.join(path, f"{frame_counter:04d}.bmp")
    cv2.imwrite(filename, resized_frame)
    frame_counter += 1
    
    # Display the processed frame (optional)
    cv2.imshow("Video Capture", resized_frame)
        
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # Ensure the loop runs at approximately 30 FPS
    elapsed = time.time() - start_time
    if elapsed < interval:
        time.sleep(interval - elapsed)

cap.release()
cv2.destroyAllWindows()