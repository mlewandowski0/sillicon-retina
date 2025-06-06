import cv2
import numpy as np



BIT_WIDTH = 16
ALPHA = 0.5
SOURCE = 0

def on_trackbar(pos):
    global ALPHA
    ALPHA = pos / BIT_WIDTH

cap = cv2.VideoCapture(SOURCE)
prev_x      = None # x[n-1]
state_y     = None # y[n-1]

cv2.namedWindow('band-pass-temporal')
cv2.createTrackbar('Alpha', 'band-pass-temporal', int(ALPHA*BIT_WIDTH), BIT_WIDTH, on_trackbar)


while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
        break

    # pre-process : gray & resize to keep maths integer-friendly
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (512,512)).astype(np.int16)


    # high pass
    if prev_x is None:
        prev_x = x.copy()   # e[n] = x[n] - x[n-1]

    e = x - prev_x
    prev_x = x

    # single-pole low-pass
    if state_y is None:
        state_y = np.zeros_like(e)

    state_y += ((e - state_y) * ALPHA).astype(np.int16)
    disp = cv2.convertScaleAbs(state_y, alpha=4)  # exaggerate a little
    cv2.imshow("band-pass-temporal", disp)

    if cv2.waitKey(1) == 27:
        break  # ESC to quit

cap.release()
cv2.destroyAllWindows()
