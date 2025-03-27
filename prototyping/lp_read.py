import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

def scamp_colorscheme(img):
    map = (np.clip(img, -128, 127) + 128).astype(np.uint8)

    return np.stack([map, map, map], axis=-1)


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
G0A = 192
lambda_A = 10

threshold = 128
V_rest = 128

dt = 0.5
tau = 0.25
Ibias = 128

max_val = 16
max_val_alpha = 32

fps = 10
delay = int(1000 / fps)

first_blur = 0

fire_delta = 10

def decode_val(val):
    if val == max_val:
        return 1.
    decoded = 0
    i = 1
    thresh = max_val // 2
    while val:
        if val >= thresh:
            val -= thresh
            decoded += 1 / 2 ** i

        thresh //= 2
        i += 1
    return decoded


def decode_map(val, start=0, prec=4):
    decoded = np.zeros_like(val).astype(np.float32)
    i = start

    threshold = 1 << ((prec+1-start) - 1)
    while threshold:
        decoded[val >= threshold] += 1 / 2 ** i
        val[val >= threshold] -= threshold

        threshold //= 2
        i += 1

    return decoded


# Callback function for trackbar
def trackbar_alpha(val):
    global alpha
    alpha = decode_val(val)  # Convert integer trackbar value to float (0.0 - 1.0)


# Callback function for trackbar
def trackbar_alpha2(val):
    global alpha2
    alpha2 = decode_val(val)  # Convert integer trackbar value to float (0.0 - 1.0)


def trackbar_threshold(val):
    global threshold
    threshold = val - 128


def trackbar_alpha4(val):
    global alpha4
    alpha4 = decode_val(val)  # Convert integer trackbar value to float (0.0 - 1.0)


def trackbar_alpha5(val):
    global alpha5
    alpha5 = decode_val(val)  # Convert integer trackbar value to float (0.0 - 1.0)


def trackbar_G0A(val):
    global G0A
    G0A = val - 128


def trackbar_lambda_A(val):
    global lambda_A
    lambda_A = decode_val(val)


def trackbar_V_rest(val):
    global V_rest
    V_rest = val - 128


def trackbar_dt(val):
    global dt
    dt = decode_val(val)


def trackbar_tau(val):
    global tau
    tau = decode_val(val)


def trackbar_Ibias(val):
    global Ibias
    Ibias = val - 128


def trackbar_fire_delta(val):
    global fire_delta
    fire_delta = val

def enable_blur(ev):
    global first_blur
    first_blur = ev > 0


def SCAMP_clip(val):
    return np.clip(val, -128, 127).astype(np.int32)


cap = cv2.VideoCapture(0)  # Replace with video path or camera index

# Create a named window
cv2.namedWindow("pipeline")

# Add a trackbar for alpha (0 to 100)
cv2.createTrackbar("Alpha", "pipeline", 1, max_val - 1, trackbar_alpha)
cv2.createTrackbar("Alpha2", "pipeline", 1, max_val - 1, trackbar_alpha2)
cv2.createTrackbar("Alpha4", "pipeline", 1, max_val - 1, trackbar_alpha4)
cv2.createTrackbar("Alpha5", "pipeline", 1, max_val - 1, trackbar_alpha5)
cv2.createTrackbar("threshold", "pipeline", threshold, 255, trackbar_threshold)
cv2.createTrackbar("enable blur", "pipeline", 0, 1, enable_blur)
cv2.createTrackbar("G0A", "pipeline", G0A, 255, trackbar_G0A)
cv2.createTrackbar("lambda_A", "pipeline", lambda_A, max_val_alpha, trackbar_lambda_A)
cv2.createTrackbar("I_bias", "pipeline", Ibias, 255, trackbar_Ibias)
cv2.createTrackbar("V_rest", "pipeline", V_rest, 255, trackbar_V_rest)
cv2.createTrackbar("dt", "pipeline", max_val//2, max_val, trackbar_dt)
cv2.createTrackbar("tau", "pipeline", 1, max_val, trackbar_tau)
cv2.createTrackbar("fire_delta", "pipeline", fire_delta, 128, trackbar_fire_delta)

files = sorted(glob.glob("dataset/*bmp"))
os.makedirs("output", exist_ok=True)

for i,file in enumerate(files):
    # Convert to grayscale for simplicity (optional, can work with color too)
    current_frame = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # apply the first blur
    if enable_blur:
        blurred = cv2.GaussianBlur(current_frame, (3, 3), 0)
    else:
        blurred = current_frame.copy()

    blurred_scamp = blurred.astype(np.int32) - 128

    # Initialize the low-pass filtered frame with the first frame
    if filtered_frame is None:
        filtered_frame = blurred_scamp.copy()

    # Apply the low-pass filter1
    filtered_frame = SCAMP_clip((alpha * blurred_scamp) + (1 - alpha) * filtered_frame)

    # Compute the high-pass frame
    high_pass_frame = SCAMP_clip(blurred_scamp - filtered_frame)
    HPF_blurred = SCAMP_clip(cv2.GaussianBlur(high_pass_frame.astype(np.float32), (5, 5), 0))

    if filtered_frame2 is None:
        filtered_frame2 = HPF_blurred.copy()

    filtered_frame2 = SCAMP_clip(alpha2 * filtered_frame2 + (1 - alpha2) * HPF_blurred)

    # Normalize the high-pass frame for better visualization (optional)
    IML = SCAMP_clip(filtered_frame2 - high_pass_frame)

    # Compute VBip
    if VBip is None:
        VBip = IML.copy()

    if alpha3 is None:
        alpha3 = np.ones_like(IML).astype(np.uint8)

    VBip = SCAMP_clip(alpha3 * VBip + (1 - alpha3) * IML)

    Q = SCAMP_clip(G0A + lambda_A * np.abs(VBip))

    if np.isnan(Q).any():
        print("Q")

    if Q_lpf is None:
        Q_lpf = Q.copy()

    Q_lpf = SCAMP_clip(alpha4 * Q_lpf + (1 - alpha4) * Q)

    if np.isnan(Q_lpf).any():
        print("Q_lpf")

    alpha3 = SCAMP_clip(np.abs(cv2.GaussianBlur(Q_lpf.astype(np.float32), (5, 5), 0)))
    alpha3 = alpha3 // 8
    alpha3 = decode_map(alpha3, start=0, prec=4)

    # High Pass Filter
    HPF = SCAMP_clip(VBip - IML)

    I_Gang = np.maximum(0, HPF) + Ibias

    if V_mem is None:
        V_mem = np.zeros_like(I_Gang)

    V_mem = SCAMP_clip(V_mem + dt * SCAMP_clip(I_Gang - SCAMP_clip(tau * np.abs(V_mem - V_rest))))
    events = V_mem > threshold
    V_mem[events] -= fire_delta

    a3 = (255 * alpha3).astype(np.uint8)
    ev = (255 * events).astype(np.uint8)

    im1 = [np.stack([current_frame,current_frame,current_frame], axis=-1),
           np.stack([blurred, blurred, blurred], axis=-1),
           scamp_colorscheme(filtered_frame),
           scamp_colorscheme(high_pass_frame),
           scamp_colorscheme(filtered_frame2),
           scamp_colorscheme(IML),
           np.zeros_like(scamp_colorscheme(IML)).astype(np.uint8)]

    im2 = [scamp_colorscheme(VBip),
           np.stack([a3, a3, a3], axis=-1),
           scamp_colorscheme(Q_lpf),
           scamp_colorscheme(HPF),
           scamp_colorscheme(I_Gang),
           scamp_colorscheme(V_mem),
           np.stack([ev, ev, ev], axis=-1)]

    im1 = np.concatenate(im1, axis=1)
    im2 = np.concatenate(im2, axis=1)

    im1_im2 = np.concatenate([im1, im2], axis=0)
    cv2.putText(im1_im2, "Frame", (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "Blur", (256, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "LPF1", (512, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "HPF1", (768, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "LPF2", (1024, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "IML", (1280, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(im1_im2, "VBip", (0, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "alpha3", (256, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "Q_lpf", (512, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "HPF", (768, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "I_gang", (1024, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "V_mem", (1280, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im1_im2, "events", (1536, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("pipeline", im1_im2)

    plt.figure(figsize=(im1_im2.shape[0] / 10, im1_im2.shape[1] / 10))
    plt.imshow(im1_im2)
    plt.axis("off")
    plt.savefig(os.path.join("output", f"{i:04d}.png"), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Exit on 'q' key
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()