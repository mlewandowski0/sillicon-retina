import os
from vd55g0_constants import *
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
import time
from vd55g0_frame_decoding import decode_frame



VD55G0_REG_DPHYTX_CTRL = 0x093A
VD55G0_REG_MAX_EXPOSURE_LINES = 0x096c
VD55G0_REG_LOW_POWER_MODE = 0x0957
bits_per_pixel = BITS_10


# Init the board
sensor_dll = SensorDll(LIB_NAME, None)
sensor_dll.init_board()

Frame_length = 250
x_start = 2
x_end = 641
y_start = 2
y_end = 601

NO_BIN_SUB = 0
BIN_2x2 = 1
BIN_4x4 = 2
SUB_2x2 = 3
SUB_4x4 = 4
EXP_AUTOMATIC_MODE = 0
EXP_AE_FREEZE = 1
EXP_MANUAL_MODE = 2


sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, x_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, x_end)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, y_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, y_end)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, SUB_4x4)
sensor_dll.Write16(VD55G0_REG_STATICS_ORIENTATION, 0x1)
sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_EXT_CLOCK, 12000000)
sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_CLK_PLL_MIPI, 1200000000)


frame_length = sensor_dll.Read16(VD55G0_REG_CONTEXTS_FRAME_LENGTH)

print(f"Frame length: {frame_length} lines")


sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, Frame_length)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_EXP_MODE, EXP_AUTOMATIC_MODE)


print("wait delay : ",sensor_dll.Read16(VD55G0_REG_CONTEXTS_WAIT_DELAY))
print(f"DPHYTX_CTRL : {sensor_dll.Read8(VD55G0_REG_DPHYTX_CTRL)}")
print(f"MAX_EXPOSURE_LINES : {sensor_dll.Read8(VD55G0_REG_MAX_EXPOSURE_LINES)}")
print(f"LOW_POWER_MODE : {sensor_dll.Read8(VD55G0_REG_LOW_POWER_MODE)}")


# Start the stream

sensor_dll.start_stream()
print(f"FRAMERATE : {sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_RATE)//16}")
start_frames = sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_COUNTER)

raw_images = []

# Grab N first frames

# Split images
t = time.time()

for i in range(10):
    sensor_dll.get_raw_image()


end_frames = sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_COUNTER)
finish = time.time()
no_frames = end_frames - start_frames
print(f"in {finish-t} we got {end_frames - start_frames} frames")
print(f"{no_frames/(finish-t)} FPS")

print("Processing time for 10 frames: ", finish - t)
print(f"FPS : {10/(finish - t)}")

sensor_dll.stop_stream()
sensor_dll.deinit_board() 