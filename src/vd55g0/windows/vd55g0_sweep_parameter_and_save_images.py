from PIL import Image

from vd55g0_constants import *
from vd55g0_frame_decoding import decode_frame
from image_sensor_python_sdk import *

bits_per_pixel = BITS_8


# Init the board
sensor_dll = SensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)

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
# Subtraction
sensor_dll.Write8(VD55G0_REG_CONTEXTS_EXP_MODE, 0x02)# mode manual

# Start the stream
sensor_dll.start_stream()

# Loop on frames
list_images = []
for integration in range(50, 1000, 50):

    # Set integration time
    sensor_dll.Write16(VD55G0_REG_CONTEXTS_MANUAL_COARSE_EXPOSURE, integration)

    # Get frames
    metadata, image = sensor_dll.get_image(width)
    print(integration, np.mean(image))
    list_images.append((integration, image))

# Saving image has to be done out of the grabbing loop.
# This reduces the computation time inside the loop and reduces the chance to skip a frame at high fps
for integration, image in list_images:
    png_image = Image.fromarray(image)
    png_image.save("VD55G0_int_%d.png" % integration)


sensor_dll.stop_stream()
sensor_dll.deinit_board()
