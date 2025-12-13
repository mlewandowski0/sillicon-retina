import time
import ctypes
import numpy as np
from vd55g0_constants import *
from image_sensor_python_sdk import *
from vd55g0_frame_decoding import decode_frame

# --- 1. Define the Optimized Class (Copy-Paste this) ---
class FastSensorDll(SensorDll):
    def __init__(self, lib_name, frame_decode_function=None):
        super().__init__(lib_name, frame_decode_function)
        self.ring_buffer = []
        self.ring_buffer_index = 0
        self.total_buffers = 0

    def prepare_ring_buffer(self, num_frames):
        # Pre-allocate memory so we don't ask the OS for RAM during capture
        print(f"Allocating RAM for {num_frames} frames...")
        self.ring_buffer = []
        self.total_buffers = num_frames
        self.ring_buffer_index = 0
        size_needed = self._buffer_size

        for _ in range(num_frames):
            buf = ctypes.create_string_buffer(size_needed)
            p_size = ctypes.pointer(ctypes.c_uint32(0))
            p_offset = ctypes.pointer(ctypes.c_int32(0))
            self.ring_buffer.append((buf, p_size, p_offset))
        print("Memory allocation complete.")

    def get_raw_image_fast(self):
        # Zero allocation capture
        current_set = self.ring_buffer[self.ring_buffer_index]
        capture_buffer, payload_size_ptr, payload_offset_ptr = current_set

        self._sensor_dll.get_raw_frame(
            capture_buffer,
            ctypes.c_uint32(self._buffer_size),
            payload_offset_ptr,
            payload_size_ptr
        )

        self.ring_buffer_index = (self.ring_buffer_index + 1) % self.total_buffers
        return capture_buffer, payload_offset_ptr.contents.value, payload_size_ptr.contents.value


# --- 2. Main Script ---
NO_BIN_SUB = 0
BIN_2x2 = 1
BIN_4x4 = 2
SUB_2x2 = 3
SUB_4x4 = 4
EXP_AUTOMATIC_MODE = 0
EXP_AE_FREEZE = 1
EXP_MANUAL_MODE = 2

LINE = 800
bits_per_pixel = BITS_10
sensor_dll = FastSensorDll(LIB_NAME, decode_frame)
sensor_dll.init_board()

# ROI Settings
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, 641)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, 2)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, 601)
sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, NO_BIN_SUB)  # SUB_2x2
sensor_dll.Write16(VD55G0_REG_STATICS_ORIENTATION, 0x1)

# --- CRITICAL: MANUAL EXPOSURE SETTINGS ---
# We force the sensor to be fast. Auto-exposure is banned.
EXP_MANUAL_MODE = 2
sensor_dll.Write8(VD55G0_REG_CONTEXTS_EXP_MODE, EXP_MANUAL_MODE)
# Exposure time = 100 lines (approx 0.5ms). Must be < Frame Length.
sensor_dll.Write16(VD55G0_REG_CONTEXTS_MANUAL_COARSE_EXPOSURE, 50)
# Frame Length = 400 lines (approx 3ms)
sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, LINE)

# Pre-allocate buffer for 300 frames (approx 1 second of video)
sensor_dll.prepare_ring_buffer(300)

sensor_dll.start_stream()
start_frames_internal = sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_COUNTER)

print("Starting High-Speed Burst...")
t_start = time.time()
captured_meta = []

# --- THE FAST LOOP ---
for i in range(300):
    # This call must happen in < 3ms
    buf, start, length = sensor_dll.get_raw_image_fast()
    # We only store the pointer reference, very fast
    captured_meta.append((buf, start, length))

t_end = time.time()
end_frames_internal = sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_COUNTER)
sensor_dll.stop_stream()

# --- Analysis ---
duration = t_end - t_start
frames_captured_python = len(captured_meta)
frames_generated_sensor = end_frames_internal - start_frames_internal

print("-" * 30)
print(f"Time Elapsed: {duration:.4f}s")
print(f"Sensor Generated: {frames_generated_sensor} frames (Internal Counter)")
print(f"Python Captured:  {frames_captured_python} frames")
print("-" * 30)
print(f"SENSOR FPS: {frames_generated_sensor / duration:.2f}")
print(f"CAPTURE FPS: {frames_captured_python / duration:.2f}")
print("-" * 30)

if frames_generated_sensor > frames_captured_python + 5:
    print("WARNING: Still dropping frames. The C++ driver itself might be blocking.")
else:
    print("SUCCESS: 1-to-1 capture achieved!")

sensor_dll.deinit_board()