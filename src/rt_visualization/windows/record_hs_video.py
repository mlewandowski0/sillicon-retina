import threading
import time 
import queue 
import cv2 
import numpy as np 
# Ensure these imports match your actual file structure
from vd55g0_constants import *
from image_sensor_python_sdk import *
from vd55g0_frame_decoding import *

RECORD_DURATION = 10       
PLAYBACK_FPS    = 30       
FILENAME        = "high_speed_output.mp4"

class CamWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.record_finished_evt = threading.Event()
        
        self.record_queue = queue.Queue()
        self.recording = False 
        
        # Statistics
        self.actual_fps = 0.0
        self.total_captured = 0
        
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Init Board
        BIN_2x2 = 1
        BIN_4x4 = 2
        SUB_2x2 = 3
        SUB_4x4 = 4
        
        self.sensor_dll = SensorDll(LIB_NAME, decode_frame)
        self.sensor_dll.init_board()
        
        bits_per_pixel = BITS_10
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, 2)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, 641)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, 2)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, 601)
        self.sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
        self.sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, 0)
        self.sensor_dll.Write16(VD55G0_REG_STATICS_ORIENTATION, 0x1)

        self.sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_EXT_CLOCK, 12000000)
        self.sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_CLK_PLL_MIPI, 1200000000)
        
        frame_length = self.sensor_dll.Read16(VD55G0_REG_CONTEXTS_FRAME_LENGTH)
        print(f"Frame length: {frame_length} lines")

        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_FRAME_LENGTH, 700)
        self.sensor_dll.Write8(VD55G0_REG_CONTEXTS_EXP_MODE, 0x0)

        

    def start_recording(self):
        print(f"--- STARTED RECORDING ({RECORD_DURATION}s) ---")
        self.total_captured = 0 # Reset counter
        self.recording = True
        
    def stop_recording(self):
        print("--- STOP COMMAND RECEIVED ---")
        self.recording = False
        self.record_queue.put(None)
        
    def _writer_loop(self):
        writer = None 
        frames_written = 0
        
        while True:
            frame = self.record_queue.get()
            
            if frame is None:
                if writer is not None:
                    writer.release()
                    print(f"\n--- DONE ---")
                    print(f"Frames Captured: {self.total_captured}")
                    print(f"Frames Written:  {frames_written}")
                    # Calculate expected video length
                    vid_len = frames_written / PLAYBACK_FPS
                    print(f"Output Video Length: {vid_len:.2f} seconds")
                self.record_finished_evt.set()
                break
            
            # Bit shift 10-bit -> 8-bit
            frame = (frame // 4).astype(np.uint8)
                        
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                writer = cv2.VideoWriter(FILENAME, fourcc, PLAYBACK_FPS, (w, h), False) 
                
            writer.write(frame)
            frames_written += 1
            
    def run(self):
        self.sensor_dll.start_stream()
        width = self.sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)

        framerate = self.sensor_dll.Read16(VD55G0_REG_STATUS_FRAME_RATE) // 16
        print(f"FRAMERATE : {framerate}")

        # FPS Calculation Variables
        frame_count = 0
        start_time = time.time()

        try:
            while not self.stop_evt.is_set():
                metadata, frame = self.sensor_dll.get_image(width)

                if self.recording:
                    self.record_queue.put(frame.copy())
                    self.total_captured += 1

                # --- FPS Calculation ---
                frame_count += 1
                if frame_count % 10 == 0: # Update stats every 10 frames
                    now = time.time()
                    dt = now - start_time
                    if dt > 0:
                        self.actual_fps = frame_count / dt
                        # Reset window every 1 second to keep average fresh
                        if dt > 1.0:
                            frame_count = 0
                            start_time = now
                # -----------------------

        except KeyboardInterrupt:
            pass
        finally:
            self.sensor_dll.stop_stream()
            self.sensor_dll.deinit_board()

if __name__ == "__main__":
    
    cam = CamWorker()
    cam.start()
    
    print("Initializing Sensor... waiting 2 seconds...")
    time.sleep(2)
    
    cam.start_recording()
    
    start_time = time.time()
    
    # MONITORING LOOP
    while (time.time() - start_time) < RECORD_DURATION:
        time.sleep(0.5) # Update console every 0.5s
        
        # Get stats
        fps = cam.actual_fps
        q_size = cam.record_queue.qsize()
        
        # Print status line (overwrites previous line for clean output)
        print(f"Status: Recording | Capture FPS: {fps:.1f} | Buffer Queue: {q_size} frames", end='\r')
            
    print("") # New line
    cam.stop_recording()
    
    print("Flushing buffer to disk... please wait.")
    cam.record_finished_evt.wait()

    cam.stop_evt.set()
    cam.join()
    print("Script Finished.")