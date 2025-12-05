#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the board. The sensor is already boot and ready to stream.
 */
void init_board(uint16_t p_i2c_address, int8_t p_board_id);

/**
 * @brief Set capture board GPIO.
 */
void set_gpio(uint32_t p_gpio_id, uint32_t p_gpio_state);

/**
 * @brief Resets the device to its initial state.
 */
void reset_device();

/**
 * @brief Reboots the device.
 */
void reboot_device();

/**
 * @brief Reconfigure the CSI receiver.
 * 
 * @param[out] p_lane_number Lanes used by the device.
 * @param[out] p_data_rate_mbps Data rate used by the device (Mbps).
 * @param[out] p_width Image width.
 * @param[out] p_height Image height (2 lines will be added for status lines).
 * @param[out] p_bit_per_pixel Image pixel bit width.
 * @param[out] p_pixel_clock_hz Device pixel frquency (Hz).
 */
extern "C" void configureCsiReceiver(uint8_t p_lane_number, uint32_t p_data_rate_mbps, uint32_t p_width, uint32_t p_height, uint32_t p_bit_per_pixel, uint32_t p_pixel_clock_hz);

/**
 * @brief Starts the streaming of data from the device.
 */
void start_stream();

/**
 * @brief Retrieves the maximum frame size supported by the device.
 * 
 * @param[out] p_max_frame_size Pointer to store the maximum frame size.
 */
void get_max_frame_size(uint32_t* p_max_frame_size);

/**
 * @brief Retrieves a raw frame from the device.
 * 
 * @param[out] p_buffer_pointer Pointer to the buffer where the raw frame will be stored.
 * @param[out] p_frame_size Pointer to store the size of the raw frame.
 */
void get_raw_frame(uint8_t* p_capture_buffer, uint32_t p_capture_buffer_size, int32_t* p_payload_buffer_offset, uint32_t* p_payload_buffer_size);

/**
 * @brief Stops the streaming of data from the device.
 */
void stop_stream();

/**
 * @brief Deinitializes the board, releasing any resources.
 */
void deinit_board();

/**
 * @brief Writes an 8-bit value to a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[in] Val The 8-bit value to write.
 */
void Write8(uint64_t RegIndex, uint8_t Val);

/**
 * @brief Writes a 16-bit value to a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[in] Val The 16-bit value to write.
 */
void Write16(uint64_t RegIndex, uint16_t Val);

/**
 * @brief Writes a 32-bit value to a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[in] Val The 32-bit value to write.
 */
void Write32(uint64_t RegIndex, uint32_t Val);

/**
 * @brief Reads an 8-bit value from a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[out] pVal Pointer to store the 8-bit value read.
 */
void Read8(uint64_t RegIndex, uint8_t* pVal);

/**
 * @brief Reads a 16-bit value from a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[out] pVal Pointer to store the 16-bit value read.
 */
void Read16(uint64_t RegIndex, uint16_t* pVal);

/**
 * @brief Reads a 32-bit value from a specified register.
 * 
 * @param[in] RegIndex The index of the register.
 * @param[out] pVal Pointer to store the 32-bit value read.
 */
void Read32(uint64_t RegIndex, uint32_t* pVal);

/** @brief
*	Wait until a register reaches a specific 8 bits value.
*	This is done thru polling.
*	@param[in] RegIndex Index of the register.
*	@param[in] ValueExpected Expected value.
*	@param[in] TimeOutInMs Timeout in millisecond.
*	@param[in] PullingInMs Waiting time between reads.
*	@return false on success else call GetError().
*/
void Wait8(uint64_t RegIndex, uint8_t ValueExpected, int64_t TimeOutInMs, int32_t PullingInMs);

/** @brief
*	Wait until a register reaches a specific 16 bits value.
*	This is done thru polling.
*	@param[in] RegIndex Index of the register.
*	@param[in] ValueExpected Expected value.
*	@param[in] TimeOutInMs Timeout in millisecond.
*	@param[in] PullingInMs Waiting time between reads.
*	@return false on success else call GetError().
*/
void Wait16(uint64_t RegIndex, uint16_t ValueExpected, int64_t  TimeOutInMs, int32_t PullingInMs);
	
/** @brief
*	Wait until a register reaches a specific 32 bits value.
*	This is done thru polling.
*	@param[in] RegIndex Index of the register.
*	@param[in] ValueExpected Expected value.
*	@param[in] TimeOutInMs Timeout in millisecond.
*	@param[in] PullingInMs Waiting time between reads.
*	@return false on success else call GetError().
*/
void Wait32(uint64_t RegIndex, uint32_t ValueExpected, int64_t TimeOutInMs, int32_t PullingInMs);

/**
 * @brief Executes a script from the specified absolute path.
 * 
 * @param[in] pAbsolutePathOfScriptFile The absolute path of the script file to execute.
 */
void ExecuteScript(const char* pAbsolutePathOfScriptFile);

/** @brief
*	Load a binary file thru the comms interface.
*	@param[in] pAbsolutePathOfBinaryFile Absolute path of the binary file.
*	@param[in] RegIndex  Starting base address to write the binary file.
*	@param[in] BurstSize The size of the burst in bytes.
*	@return false on success else call GetError().
*/
int load_binary(const char* pAbsolutePathOfBinaryFile, uint64_t RegIndex, uint32_t BurstSize);

#ifdef __cplusplus
}
#endif