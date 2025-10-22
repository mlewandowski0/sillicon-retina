import os
from vd55g0_constants import *
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
import time

from vd55g0_frame_decoding import decode_frame

bits_per_pixel = BITS_10

# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

x_start = 2
x_end = 641
y_start = 2
y_end = 601


# Configure the sensor
bits_per_pixel = BITS_10
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, 641)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, 601)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, 0x1)
sensor_dll.Write8(VD55G0_REG_STATICS_ORIENTATION, 0x1)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, 700)

# Start the stream
sensor_dll.start_stream()
width = sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)

img_x = sensor_dll.Read16(VD55G0_REG_STATUS_X_SIZE)
img_y = sensor_dll.Read16(VD55G0_REG_STATUS_Y_SIZE)

y_end = sensor_dll.Read16(VD55G0_REG_STATUS_Y_END)
y_start = sensor_dll.Read16(VD55G0_REG_STATUS_Y_START)

print(f"Image size : {img_x}x{img_y}")
print(f"y_start = {y_start}, y_end={y_end}")

# Start the stream
sensor_dll.start_stream()

raw_images = []

frame_length = sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_LENGTH)
line_length = sensor_dll.Read16(VD55G0_REG_STATUS_LINE_LENGTH)

print(f"framerate = {sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_RATE) // 16}")
print(f"Frame length : {frame_length}")
print(f"line lenth : {line_length}")


# Grab N first frames

frames = 20 

start = time.time()
for i in range(frames):
    sensor_dll.get_raw_image()

end = time.time()
print("asd")
print(f"Measured framerate = {frames/(end-start)}")


sensor_dll.stop_stream()
sensor_dll.deinit_board()
