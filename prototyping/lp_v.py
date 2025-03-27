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


def on_trackbar6(val):
    global G0A
    G0A = val / 100.0


def on_trackbar7(val):
    global G0A
    G0A = val / 100.0


def on_trackbar8(val):
    global lambda_A
    lambda_A = val / 100.0


def mouse_click(ev):
    pass


cap = cv2.VideoCapture(0)  # Replace with video path or camera index

# Create a named window
cv2.namedWindow("event")

# Add a trackbar for alpha (0 to 100)
cv2.createTrackbar("Alpha", "event", int(alpha * 100), 100, on_trackbar)
cv2.createTrackbar("Alpha2", "event", int(alpha2 * 100), 100, on_trackbar2)
cv2.createTrackbar("Alpha4", "event", int(alpha4 * 100), 100, on_trackbar4)
cv2.createTrackbar("Alpha5", "event", int(alpha5 * 100), 100, on_trackbar5)
cv2.createTrackbar("threshold", "event", threshold, 255, on_trackbar_threshold)
cv2.createTrackbar("enable blur", "event", 0, 1, mouse_click)
cv2.createTrackbar("G0A", "event", int(G0A * 100), 100, on_trackbar6)
cv2.createTrackbar("lambda_A", "event", int(lambda_A * 100), 100, on_trackbar7)

while cap.isOpened():
    ret, current_frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for simplicity (optional, can work with color too)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # current_frame = cv2.resize(current_frame, (256, 256))

    # apply the first blur
    blurred = cv2.GaussianBlur(current_frame, (3, 3), 0)

    # Initialize the low-pass filtered frame with the first frame
    if filtered_frame is None:
        filtered_frame = blurred.astype(np.float32)

    # Apply the low-pass filter1
    filtered_frame = alpha * blurred.astype(np.float32) + (1 - alpha) * filtered_frame

    # Compute the high-pass frame
    high_pass_frame = current_frame.astype(np.float32) - filtered_frame

    HPF_blurred = cv2.GaussianBlur(high_pass_frame, (5, 5), 0)
    if filtered_frame2 is None:
        filtered_frame2 = HPF_blurred.astype(np.float32)

    filtered_frame2 = alpha2 * filtered_frame2 + (1 - alpha2) * HPF_blurred.astype(np.float32)

    # Normalize the high-pass frame for better visualization (optional)
    IML = filtered_frame2 - high_pass_frame

    # Compute VBip
    if VBip is None:
        VBip = IML.astype(np.float32)

    if alpha3 is None:
        alpha3 = np.ones_like(IML)

    VBip = alpha3 * VBip + (1 - alpha3) * IML.astype(np.float32)

    VBip_norm = VBip / np.max(VBip)
    Q = G0A + lambda_A * VBip_norm * VBip_norm
    if np.isnan(Q).any():
        print("Q")

    if Q_lpf is None:
        Q_lpf = Q.copy()

    Q_lpf = alpha4 * Q_lpf + (1 - alpha4) * Q

    if np.isnan(Q_lpf).any():
        print("Q_lpf")

    alpha3 = np.clip(cv2.GaussianBlur(Q_lpf, (3, 3), 0), 0, 1)
    if np.isnan(Q_lpf).any():
        print("alpha3")

    # High Pass Filter
    HPF = VBip - IML

    I_Gang = np.maximum(0, HPF)

    if V_mem is None:
        V_mem = I_Gang

    V_mem = alpha5 * I_Gang + (1 - alpha5) * V_mem

    im1 = [scamp_colorscheme(current_frame),
           scamp_colorscheme(blurred),
           scamp_colorscheme(filtered_frame),
           scamp_colorscheme(high_pass_frame),
           scamp_colorscheme(filtered_frame2),
           scamp_colorscheme(IML)]
    im2 = [VBip,
           alpha3,
           Q_lpf,
           HPF,
           I_Gang,
           V_mem]

    im1 = np.concatenate(im1, axis=1)
    im2 = np.concatenate(im2, axis=1)

    im1_im2 = np.concatenate([im1, im2], axis=0)

    cv2.imshow("pipeline1", im1)
    cv2.imshow("pipeline2", im2)
    cv2.imshow("event", (255 * (255 * c > threshold)).astype(np.uint8))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()