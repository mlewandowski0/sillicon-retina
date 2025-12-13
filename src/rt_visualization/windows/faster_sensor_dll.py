import collections
from image_sensor_python_sdk import *
import ctypes

# Inherit from your existing class so we don't break anything
class FastSensorDll(SensorDll):
    def __init__(self, lib_name, frame_decode_function=None):
        super().__init__(lib_name, frame_decode_function)
        self.ring_buffer = []
        self.ring_buffer_index = 0
        self.total_buffers = 0

    def prepare_ring_buffer(self, num_frames):
        """
        Pre-allocates 'num_frames' buffers in RAM before we start streaming.
        This prevents the OS from slowing us down during capture.
        """
        print(f"Allocating memory for {num_frames} frames...")
        self.ring_buffer = []
        self.total_buffers = num_frames
        self.ring_buffer_index = 0

        # Determine size needed
        size_needed = self._buffer_size

        for _ in range(num_frames):
            # 1. Create the buffer once
            buf = ctypes.create_string_buffer(size_needed)
            # 2. Create pointers for size/offset once
            p_size = ctypes.pointer(ctypes.c_uint32(0))
            p_offset = ctypes.pointer(ctypes.c_int32(0))

            # Store tuple: (buffer, pointer_to_size, pointer_to_offset)
            self.ring_buffer.append((buf, p_size, p_offset))

        print("Memory allocation complete.")

    def get_raw_image_fast(self):
        """
        Captures a frame into the next pre-allocated buffer.
        Zero memory allocation overhead.
        """
        # 1. Get the next available buffer from our pool
        current_set = self.ring_buffer[self.ring_buffer_index]
        capture_buffer, payload_size_ptr, payload_offset_ptr = current_set

        # 2. Call the C++ DLL (This is the only blocking part now)
        # Note: We pass ctypes.c_uint32(self._buffer_size) directly to save creation time
        self._sensor_dll.get_raw_frame(
            capture_buffer,
            ctypes.c_uint32(self._buffer_size),
            payload_offset_ptr,
            payload_size_ptr
        )

        # 3. Advance index for next time (Circular/Ring buffer)
        self.ring_buffer_index = (self.ring_buffer_index + 1) % self.total_buffers

        # 4. Return the buffer directly (we will process it later!)
        # We return the specific slice of data that contains the image
        valid_data_start = payload_offset_ptr.contents.value
        valid_data_len = payload_size_ptr.contents.value

        return capture_buffer, valid_data_start, valid_data_len