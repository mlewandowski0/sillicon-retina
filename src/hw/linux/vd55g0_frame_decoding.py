from vd55g0_constants import *
from image_sensor_python_sdk import SensorDll, BITS_10, BITS_8
from math import gcd


def extract_first_status_line_value8(status_register_address, raw_frame):
    assert status_register_address < 0x7d
    register_offset = 2 * status_register_address + 6
    return raw_frame[register_offset]


def extract_second_status_line_value8(status_register_address, raw_frame, width, bits_per_pixel):
    assert status_register_address >= 0x7d
    frame_width_in_bytes = int(width * bits_per_pixel / 8)
    register_offset = frame_width_in_bytes + 2 * (status_register_address - 0x7d) + 6
    return raw_frame[register_offset]


def extract_status_line_value8(status_register_address, raw_frame, width=None, bits_per_pixel=None):
    if status_register_address < 0x7d:
        return extract_first_status_line_value8(status_register_address, raw_frame)
    else:
        return extract_second_status_line_value8(status_register_address, raw_frame, width, bits_per_pixel)


def extract_status_line_value16(status_register_address, raw_frame, width=None, bits_per_pixel=None):
    val1 = extract_status_line_value8(status_register_address, raw_frame, width, bits_per_pixel)
    val2 = extract_status_line_value8(status_register_address + 1, raw_frame, width, bits_per_pixel)
    return int(val1) + 256 * int(val2)


def extract_status_line_value32(status_register_address, raw_frame, width=None, bits_per_pixel=None):
    val1 = extract_status_line_value16(status_register_address, raw_frame, width, bits_per_pixel)
    val2 = extract_status_line_value16(status_register_address + 2, raw_frame, width, bits_per_pixel)
    return int(val1) + 65536 * int(val2)


def decode_frame(raw_frame, width):

    # Extract bits per pixel from the status line
    bits_per_pixel = int(extract_status_line_value8(VD55G0_REG_STATUS_FORMAT_CTRL, raw_frame))
    assert bits_per_pixel in [BITS_8, BITS_10], f"Unsupported bits per pixel: {bits_per_pixel}"

    # Check that the frame width can be sent by 4-bytes packet
    cx3_limits = gcd(bits_per_pixel, 32)
    x_size = width
    assert (x_size * bits_per_pixel) % 32 == 0, (f"frame width must be a multiple of {32 // cx3_limits} "
                                                 f"when working with {bits_per_pixel} bits due to cx3 limitations")

    # Extract Y size, frame counter, and current context
    y_size = extract_status_line_value16(VD55G0_REG_STATUS_OUT_ROI_Y_SIZE, raw_frame, width, bits_per_pixel)
    frame_counter = extract_status_line_value16(VD55G0_REG_STATUS_FRAME_COUNTER, raw_frame)
    current_context = extract_status_line_value8(VD55G0_REG_STATUS_CURRENT_CONTEXT, raw_frame)

    # Calculate the image width in bytes
    x_size_in_bytes = int(bits_per_pixel * x_size / 8)
    status_line_number = 2

    # Extract the image buffer from the raw data
    image_buffer = raw_frame[status_line_number * x_size_in_bytes:].reshape((y_size, -1))

    # Decode the raw image based on bits per pixel
    if bits_per_pixel == BITS_10:
        decoded_image = SensorDll.decode_raw_10(image_buffer)
    else:
        decoded_image = image_buffer

    # Generate metadata
    metadata = {
        'width': x_size,
        'height': y_size,
        'bits_per_pixels': bits_per_pixel,
        'frame_counter': frame_counter,
        'current_context': current_context,
    }

    return metadata, decoded_image
