import numpy as np
import cv2


def scamp_colorscheme(img):
    return (np.clip(img, -128, 127) + 128).astype(np.uint8)


# Initialize variables
alpha = 0.1  # Smoothing factor for low-pass filter
filtered_frame = None

alpha2 = 0.9
filtered_frame2 = None

alpha3 = None
alpha4 = 0.1
VBip = None
Q_lpf = None

V_mem = None
alpha5 = 0.1

# Stream from camera (simulate using a video file for example)
G0A = 0.1
lambda_A = 0.1

threshold = 10

# Callback function for trackbar
def on_trackbar(val):
    global alpha
    alpha = val / 100.0  # Convert integer trackbar value to float (0.0 - 1.0)

# Callback function for trackbar
def on_trackbar2(val):
    global alpha2
    alpha2 = val / 100.0  # Convert integer trackbar value to float (0.0 - 1.0)

def on_trackbar_threshold(val):
    global threshold
    threshold = val

def on_trackbar4(val):
    global alpha4
    alpha4 = val / 100.0  # Convert integer trackbar value to float (0.0 - 1.0)

def on_trackbar5(val):
    global alpha5
    alpha5 = val / 100.0  # Convert integer trackbar value to float (0.0 - 1.0)

def mouse_click(ev):
    pass

cap = cv2.VideoCapture(0)  # Replace with video path or camera index

# Create a named window
cv2.namedWindow("pipeline")

while cap.isOpened():
    ret, current_frame = cap.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_frame = cv2.resize(current_frame, (256, 256))


    cv2.imshow("img", current_frame)
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()