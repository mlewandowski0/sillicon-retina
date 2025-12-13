import os
from vd55g0_constants import *
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
from vd55g0_frame_decoding import *

BIN_2x2 = 1
BIN_4x4 = 2
SUB_2x2 = 3
SUB_4x4 = 4

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
sensor_dll.Write16(VD55G0_REG_STATICS_ORIENTATION, 0x1)

sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_EXT_CLOCK, 12000000)
sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_CLK_PLL_MIPI, 1200000000)



print("VD55G0_REG_STATUS_READOUT_CTRL : ", sensor_dll.Read8(VD55G0_REG_CONTEXTS_READOUT_CTRL))

# Start the stream
sensor_dll.start_stream()
width = sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)

# Init window for stream
smart_imshow = SmartImshow(os.path.basename(__file__))

# Loop on frames
try:
    while 1:
        # Get frames
        metadata, image = sensor_dll.get_image(width)
        # Show frames
        print(image.shape)
        k = smart_imshow.set_image(image, bits_per_pixel)

        # Press ESC to Exit
except KeyboardInterrupt:
    pass
finally:
    print("DONE")
    # Stop the sensor
    sensor_dll.stop_stream()
    sensor_dll.deinit_board()
