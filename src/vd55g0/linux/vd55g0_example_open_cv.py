import os
from vd55g0_constants import *
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
from vd55g0_frame_decoding import *


# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

# Configure the sensor
bits_per_pixel = BITS_10
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, 641)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, 601)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, 0)

# Start the stream
sensor_dll.start_stream()
width = sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)

# Init window for stream
smart_imshow = SmartImshow(os.path.basename(__file__))

# Loop on frames
while 1:
    # Get frames
    metadata, image = sensor_dll.get_image(width)
    # Show frames
    k = smart_imshow.set_image(image, bits_per_pixel)
    # Press ESC to Exit
    if k == 27:
        break

# Stop the sensor
sensor_dll.stop_stream()
sensor_dll.deinit_board()
