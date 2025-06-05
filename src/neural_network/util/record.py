import cv2
import time

# === Parameters ===
output_filename = 'output.mp4'
frame_width = 640
frame_height = 480
fps = 25
recording_time = 5

# === Open the default camera ===
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)  # Not all cameras respect this

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'MJPG', 'XVID', or 'MP4V'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

print("Recording started. Press 'q' to stop.")

start = time.time()
while time.time() - start < recording_time:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Recording...', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Release resources ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording saved to", output_filename)
