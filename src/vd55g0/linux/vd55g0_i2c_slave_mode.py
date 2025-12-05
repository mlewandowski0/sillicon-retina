import os

from vd55g0_constants import *
from vd55g0_frame_decoding import decode_frame
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
import cv2
import threading

bits_per_pixel = BITS_8

# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
# Slave mode
slave_mode = 2

# Set the sensor configuration (VD55G1 specific)

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
height = y_end - y_start + 1

sensor_dll.Write8(VD55G0_REG_STATUS_VT_CTRL, slave_mode)

# Start the stream
sensor_dll.start_stream()

# Init window for stream with a blank image
smart_imshow = SmartImshow(os.path.basename(__file__))
metadata = None
image = np.zeros((1360, 1120), np.uint8)
smart_imshow.set_image(image)


# Capture thread in parallel with I2C command
def capture_thread_function(p_image):
    _metadata, _image = sensor_dll.get_image(width)
    p_image[0] = _image


# Loop on frames
while 1:
    k = cv2.waitKeyEx(1)
    # Press ENTER to have a new frame
    if k == 13:
        # Get frame request in a thread then run I2C command request
        image_list = [0]
        x = threading.Thread(target=capture_thread_function, args=(image_list, ))
        x.start()
        sensor_dll.Write8(VD55G0_REG_CMD_STREAMING, 2)
        x.join()
        image = image_list[0]

        # Show captured frames
        smart_imshow.set_image(image, bits_per_pixel)
    # Press ESC to Exit
    if k == 27:
        break


sensor_dll.stop_stream()
sensor_dll.deinit_board()
