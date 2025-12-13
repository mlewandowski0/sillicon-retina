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


sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, x_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, x_end)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, y_start)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, y_end)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)


# Start the stream
sensor_dll.start_stream()

raw_images = []

# Grab N first frames
# Split images
t = time.time()
for i in range(10):
    raw_images.append(sensor_dll.get_raw_image())

finish = time.time()
print("Processing time for 10 frames: ", finish - t)
print(f"FPS : {10/(finish - t)}")


for frame in raw_images:


    frame_width_in_bytes = int((x_end-x_start+1) * bits_per_pixel / 8)
    x_size = np.uint16(frame[frame_width_in_bytes + 2 * (VD55G0_REG_STATUS_OUT_ROI_X_SIZE - 0x7d) + 6])+ 256 * np.uint16(frame[frame_width_in_bytes + 2 * (VD55G0_REG_STATUS_OUT_ROI_X_SIZE+1 - 0x7d) + 6])
    y_size = np.uint16(frame[frame_width_in_bytes + 2 * (VD55G0_REG_STATUS_OUT_ROI_Y_SIZE - 0x7d) + 6])+ 256 * np.uint16(frame[frame_width_in_bytes + 2 * (VD55G0_REG_STATUS_OUT_ROI_Y_SIZE+1 - 0x7d) + 6])
    x_size_in_bytes = int(5*x_size/4)
    frame_counter= np.uint16(frame[2 * VD55G0_REG_STATUS_FRAME_COUNTER + 6]) +256*np.uint16(frame[2 *( VD55G0_REG_STATUS_FRAME_COUNTER+1) + 6])
    exposure = np.uint16(frame[2 * VD55G0_REG_STATUS_APPLIED_COARSE_EXPOSURE + 6]) + 256 * np.uint16(
        frame[2 * (VD55G0_REG_STATUS_APPLIED_COARSE_EXPOSURE + 1) + 6])
    raw10_data = frame[x_size_in_bytes*2:].reshape((y_size, -1))
    decoded_image = sensor_dll.decode_raw_10(raw10_data)

    print(frame_counter, exposure, np.average(decoded_image))



sensor_dll.stop_stream()
sensor_dll.deinit_board()
