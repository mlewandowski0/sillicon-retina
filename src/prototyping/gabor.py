import cv2
import numpy as np

# Callback for trackbars (does nothing but needed for createTrackbar)
def nothing(x):
    pass

# Create window and trackbars
cv2.namedWindow('Gabor Filter Viewer')
cv2.createTrackbar('Orientation (deg)', 'Gabor Filter Viewer', 0, 180, nothing)
cv2.createTrackbar('Frequency (1/λ)', 'Gabor Filter Viewer', 1, 20, nothing)

# Gabor kernel constants
ksize = 31
sigma = 4.0
gamma = 0.5
psi = 0

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Read trackbar positions
    theta_deg = cv2.getTrackbarPos('Orientation (deg)', 'Gabor Filter Viewer')
    freq_val = cv2.getTrackbarPos('Frequency (1/λ)', 'Gabor Filter Viewer')

    # Avoid division by zero
    if freq_val == 0:
        freq_val = 1

    # Compute parameters
    theta = np.deg2rad(theta_deg)
    lambd = 1.0 / freq_val * 20  # scale to make visually useful

    # Generate kernel and filter image
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)

    # Display both original and filtered image
    combined = np.hstack((gray, filtered))
    cv2.imshow('Gabor Filter Viewer', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
