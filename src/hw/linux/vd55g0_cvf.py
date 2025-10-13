from vd55g0_constants import *
from vd55g0_frame_decoding import *
from image_sensor_python_sdk import *
import Cvf_calculator


bits_per_pixel = BITS_8
# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

min_expo = 21
expo_margin = 0x5E3

# Set the sensor configuration (VD55G0 specific)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_EXP_MODE, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_MANUAL_COARSE_EXPOSURE, min_expo)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_MANUAL_ANALOG_GAIN, 0)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_GPIO_2_CTRL, 2)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)

# # Bypass digital
sensor_dll.Write8(VD55G0_REG_STATICS_DARKCAL_CTRL, 0x2)
sensor_dll.Write16(VD55G0_REG_DYNAMICS_DARKCAL_PEDESTAL, 0x0000)
sensor_dll.Write8(VD55G0_REG_STATICS_DUSTER_CTRL, 0x0)

def VD55G0_exposure_function(exposure, frame_length):
    if frame_length:
        # Set frame length, needs to be higher than Y_HEIGHT + FRAME_LENGTH_OFFSET
        sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, max(frame_length, 604 + 76))
    else:
        # Set frame length at the minimum value that allows the exposure time
        sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, max(exposure + expo_margin, 604 + 76))

    # Set the exposure
    sensor_dll.Write16(VD55G0_REG_CONTEXTS_MANUAL_COARSE_EXPOSURE, exposure)


# Start streaming
sensor_dll.start_stream()

# Generate CVF
width = sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)
Cvf_calculator.CVFCalculator(sensor_dll, VD55G0_exposure_function, min_expo, expo_margin, bits_per_pixel, width=width)

# Stop streaming
sensor_dll.stop_stream()
sensor_dll.deinit_board()

