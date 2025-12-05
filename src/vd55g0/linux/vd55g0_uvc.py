# https://github.com/letmaik/pyvirtualcam
# Dependency : Need to install OBS, https://obsproject.com/
import pyvirtualcam

from vd55g0_constants import *
from vd55g0_frame_decoding import decode_frame
from image_sensor_python_sdk import *
import cv2

bits_per_pixel = BITS_8

# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

x_start = 2
x_end = 641
y_start = 2
y_end = 601


sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, x_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, x_end)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, y_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, y_end)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)


width = x_end - x_start + 1
# Start the stream
sensor_dll.start_stream()

with pyvirtualcam.Camera(width=(x_end-x_start)+1, height=(y_end-y_start)+1, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        metadata, image = sensor_dll.get_image(width)
        frame = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        cam.send(frame)
        cam.sleep_until_next_frame()