from vd55g0_constants import *
from image_sensor_python_sdk import SensorDll, BITS_10, BITS_8
from math import gcd

def active_line_bytes(width, bpp):
    return int(width * bpp / 8)

def status_line_bytes(width, bpp):
    # VD55G0: embedded status line width is active line width, with a minimum of 256 bytes
    return max(256, active_line_bytes(width, bpp))

def extract_status8(addr, raw_frame, width, bpp):
    slb = status_line_bytes(width, bpp)
    if addr < 0x7d:
        off = 2 * addr + 6
    else:
        off = slb + 2 * (addr - 0x7d) + 6
    return raw_frame[off]

def extract_status16(addr, raw_frame, width, bpp):
    lo = extract_status8(addr, raw_frame, width, bpp)
    hi = extract_status8(addr + 1, raw_frame, width, bpp)
    return int(lo) | (int(hi) << 8)

def decode_frame(raw_frame, width):
    # FORMAT_CTRL is in first status line (<0x7d), so width/bpp not needed yet
    bpp = int(raw_frame[2 * VD55G0_REG_STATUS_FORMAT_CTRL + 6])
    assert bpp in (BITS_8, BITS_10), f"Unsupported bits per pixel: {bpp}"

    cx3_limits = gcd(bpp, 32)
    assert (width * bpp) % 32 == 0, (
        f"frame width must be a multiple of {32 // cx3_limits} when working with {bpp} bits"
    )

    y_size = extract_status16(VD55G0_REG_STATUS_OUT_ROI_Y_SIZE, raw_frame, width, bpp)
    frame_counter = extract_status16(VD55G0_REG_STATUS_FRAME_COUNTER, raw_frame, width, bpp)
    current_context = extract_status8(VD55G0_REG_STATUS_CURRENT_CONTEXT, raw_frame, width, bpp)

    alb = active_line_bytes(width, bpp)
    slb = status_line_bytes(width, bpp)

    payload_off = 2 * slb
    need = payload_off + y_size * alb
    if len(raw_frame) < need:
        raise ValueError(
            f"Frame too small for decode: len={len(raw_frame)}, "
            f"width={width}, bpp={bpp}, y_size={y_size}, "
            f"status_line_bytes={slb}, active_line_bytes={alb}, need>={need}. "
            f"(If ISL is disabled, enable it while using this decoder.)"
        )

    image_bytes = raw_frame[payload_off: payload_off + y_size * alb]
    image_buffer = image_bytes.reshape((y_size, alb))

    if bpp == BITS_10:
        decoded = SensorDll.decode_raw_10(image_buffer)
    else:
        decoded = image_buffer

    meta = dict(
        width=width,
        height=y_size,
        bits_per_pixel=bpp,
        frame_counter=frame_counter,
        current_context=current_context,
    )
    return meta, decoded
