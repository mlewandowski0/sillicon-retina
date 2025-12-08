import ctypes
import numpy as np
import platform

def get_value_based_on_system_os(windows_value=None, linux_value=None, mac_value=None):
    system = platform.system()
    if system == "Linux":
        assert linux_value is not None, "linux_value should be set"
        return linux_value
    elif system == "Windows":
        assert windows_value is not None, "windows_value should be set"
        return windows_value
    else:
        raise SystemError(f"This is an unknown platform: {system}")

def get_lib_name(sensor_name):
    return get_value_based_on_system_os(
        windows_value=f"{sensor_name}.dll",
        linux_value=f"lib{sensor_name}.so"
    )

# Defines
BITS_8 = 8
BITS_10 = 10
BITS_16 = 16
BITS_32 = 32

GPIO_0_ID = 0


class SensorDll(object):
    def __init__(self, lib_name, frame_decode_function=None):
        print("Loading " + lib_name)
        lib_path = get_value_based_on_system_os(windows_value=".\\", linux_value="./") + lib_name
        self._sensor_dll = ctypes.cdll.LoadLibrary(lib_path)
        self._frame_decode_function = frame_decode_function
        self._max_frame_size = 0
        self._buffer_size = 0

    @property
    def sensor_dll(self):
        return  self._sensor_dll

    def init_board(self, i2c_address=0x20, board_id=-1):
        # Define the argument and return types for the init_board function
        self._sensor_dll.init_board(ctypes.c_uint32(i2c_address), ctypes.c_int8(board_id))
        # set buffer size bigger than max frame size to have capture overhead margin
        self._max_frame_size = self.get_32bits_param(self._sensor_dll.get_max_frame_size)
        self._buffer_size = 1024 + self._max_frame_size*2

    def search_i2c_address(self):
        self._sensor_dll.search_i2c_address()

    def set_gpio(self, gpio_id, gpio_state):
        self._sensor_dll.set_gpio(ctypes.c_uint32(gpio_id), ctypes.c_uint32(gpio_state))

    def reset_device(self):
        self._sensor_dll.reset_device()

    def reconfigure_csi_receiver(self, lane_number, data_rate_mbps, width, height, bit_per_pixel, pixel_clock_hz):
        self._sensor_dll.configureCsiReceiver(ctypes.c_uint8(lane_number), ctypes.c_uint32(data_rate_mbps), ctypes.c_uint32(width), ctypes.c_uint32(height), ctypes.c_uint32(bit_per_pixel), ctypes.c_uint32(int(pixel_clock_hz)))

    def start_stream(self):
        self._sensor_dll.start_stream()

    def stop_stream(self):
        self._sensor_dll.stop_stream()

    def deinit_board(self):
        self._sensor_dll.deinit_board()

    def set_capture_buffer_size(self, capture_buffer_size):
        self._buffer_size = capture_buffer_size
        print(f"Capture buffer size set to {self._buffer_size}, ({100.*capture_buffer_size/self._max_frame_size:.2f}% of max frame size)")

    def get_raw_image(self):
        assert self._buffer_size > 0, "Board is not initialized"
        capture_buffer_size = ctypes.c_uint32(self._buffer_size)
        capture_buffer = ctypes.create_string_buffer(capture_buffer_size.value)
        payload_buffer_size = ctypes.pointer(ctypes.c_uint32(0))
        payload_buffer_offset = ctypes.pointer(ctypes.c_int32(0))

        # Get the frame
        self._sensor_dll.get_raw_frame(capture_buffer, capture_buffer_size, payload_buffer_offset, payload_buffer_size)

        # Return the frame
        return np.frombuffer(capture_buffer[payload_buffer_offset.contents.value:], dtype=np.uint8, count=payload_buffer_size.contents.value)

    def get_image(self, width=None):
        """width parameter is only needed for VD55G5 and VDx6Gx (and derivatives) sensors"""
        assert self._frame_decode_function is not None, "Frame decode function is not set"
        return self._frame_decode_function(self.get_raw_image(), width)

    def read_register(self, register_address, register_length_in_bits):
        print("method read_register will be deprecated, please use Read8, Read16 or Read32")
        return self._read_register(register_address, register_length_in_bits)

    def _read_register(self, register_address, register_length_in_bits):
        if register_length_in_bits == BITS_8:
            p_uint8 = ctypes.pointer(ctypes.c_uint8(0))
            self._sensor_dll.Read8(ctypes.c_uint64(register_address), p_uint8)
            return p_uint8.contents.value
        elif register_length_in_bits == BITS_16:
            p_uint16 = ctypes.pointer(ctypes.c_uint16(0))
            self._sensor_dll.Read16(ctypes.c_uint64(register_address), p_uint16)
            return p_uint16.contents.value
        elif register_length_in_bits == BITS_32:
            p_uint32 = ctypes.pointer(ctypes.c_uint32(0))
            self._sensor_dll.Read32(ctypes.c_uint64(register_address), p_uint32)
            return p_uint32.contents.value
        else:
            raise ValueError("register_length_in_bits should be 8 or 16 or 32")

    def Read8(self, register_address):
        return self._read_register(register_address, 8)

    def Read16(self, register_address):
        return self._read_register(register_address, 16)

    def Read32(self, register_address):
        return self._read_register(register_address, 32)

    def write_register(self, register_address, register_value, register_length_in_bits):
        print("method write_register will be deprecated, please use Write8, Write16 or Write32")
        return self._write_register(register_address, register_value, register_length_in_bits)

    def _write_register(self, register_address, register_value, register_length_in_bits):
        if register_length_in_bits == BITS_8:
            self._sensor_dll.Write8(ctypes.c_uint64(register_address), ctypes.c_uint8(register_value))
        elif register_length_in_bits == BITS_16:
            self._sensor_dll.Write16(ctypes.c_uint64(register_address), ctypes.c_uint16(register_value))
        elif register_length_in_bits == BITS_32:
            self._sensor_dll.Write32(ctypes.c_uint64(register_address), ctypes.c_uint32(register_value))
        else:
            raise ValueError("register_length_in_bits should be 8 or 16 or 32")

    def Write8(self, register_address, register_value):
        self._write_register(register_address, register_value, 8)

    def Write16(self, register_address, register_value):
        self._write_register(register_address, register_value, 16)

    def Write32(self, register_address, register_value):
        self._write_register(register_address, register_value, 32)

    def Wait8(self, register_address, expected_value, timeout_in_ms, pulling_in_ms):
        self._sensor_dll.Wait8(register_address, expected_value, timeout_in_ms, pulling_in_ms)

    def Wait16(self, register_address, expected_value, timeout_in_ms, pulling_in_ms):
        self._sensor_dll.Wait16(register_address, expected_value, timeout_in_ms, pulling_in_ms)

    def Wait32(self, register_address, expected_value, timeout_in_ms, pulling_in_ms):
        self._sensor_dll.Wait32(register_address, expected_value, timeout_in_ms, pulling_in_ms)

    @staticmethod
    def get_16bits_param(dll_function):
        p_uint16 = ctypes.pointer(ctypes.c_uint16(0))
        dll_function(p_uint16)
        return p_uint16.contents.value

    @staticmethod
    def get_32bits_param(dll_function):
        p_uint32 = ctypes.pointer(ctypes.c_uint32(0))
        dll_function(p_uint32)
        return p_uint32.contents.value

    def load_script(self, script_path):
        string_buffer = ctypes.create_string_buffer(len(script_path)+1)
        string_buffer.value = bytes(script_path, "utf8")
        self._sensor_dll.ExecuteScript(string_buffer)

    @staticmethod
    def decode_raw_10(raw10_data):

        raw10_data_flat = raw10_data.flatten()

        # Extract the 10-bit values
        byte_0 = raw10_data_flat[0::5].astype(np.uint16)
        byte_1 = raw10_data_flat[1::5].astype(np.uint16)
        byte_2 = raw10_data_flat[2::5].astype(np.uint16)
        byte_3 = raw10_data_flat[3::5].astype(np.uint16)
        byte_4 = raw10_data_flat[4::5].astype(np.uint16)

        # Combine the bytes to form the 10-bit pixel values
        pixel_0 = (byte_0 << 2) | ((byte_4 >> 0) & 0x03)
        pixel_1 = (byte_1 << 2) | ((byte_4 >> 2) & 0x03)
        pixel_2 = (byte_2 << 2) | ((byte_4 >> 4) & 0x03)
        pixel_3 = (byte_3 << 2) | ((byte_4 >> 6) & 0x03)

        # Stack the pixels into a single array
        decoded_data = np.stack((pixel_0, pixel_1, pixel_2, pixel_3), axis=-1).flatten().reshape((raw10_data.shape[0], -1))

        return decoded_data

    def load_binary(self, binary_file_path, start_register, burst_size):
        string_buffer = ctypes.create_string_buffer(len(binary_file_path) + 1)
        string_buffer.value = bytes(binary_file_path, "utf8")
        self._sensor_dll.load_binary(string_buffer, start_register, burst_size)
