import dearpygui.dearpygui as dpg
import random
import numpy as np
import cv2
import time
from esim_torch import SiliconRetinaFloatCUDAOnline , SiliconRetinaConfig
from esim_torch.silicon_retina_cuda import BLUR_MODE, HPF_MODE
import threading
import torch
import time
import cv2

from vd55g0_constants import *
from image_sensor_python_sdk import *
from smart_imshow import SmartImshow
from vd55g0_frame_decoding import *


selected_testpoint1 = "events"
width, height = 300, 320
cfg = SiliconRetinaConfig(H=height, W=width)
simulator = SiliconRetinaFloatCUDAOnline(cfg)

img_upscale = 2
APP_WIDTH = 1600
APP_HEIGHT = 1024

input_image_tag = "input_image_tag"
display1_tag = "display1_tag"

CAP_INDEX = 0
REQ_W, REQ_H = width, height
REQ_FPS = 60
current_t = 1
image_idx = 0

def normalize01(a, eps=1e-8):
    m, M = torch.min(a), torch.max(a)
    if M - m < eps:
        return torch.zeros_like(a)
    return (a - m) / (M - m)

def colorize_two_maps(pos, neg):
    """
    Make a simple dual-polarity heatmap:
      pos -> red channel (normalized)
      neg -> blue channel (normalized)
    """
    pr = normalize01(pos)
    nr = normalize01(neg)
    rgb = torch.zeros((pos.shape[0], pos.shape[1], 3)).float()
    rgb[...,0] = pr      # R
    rgb[...,2] = nr      # B
    return torch.clip(rgb, 0.0, 1.0)


class CamWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.fps = 0 
        self.idx = 0

        BIN_2x2 = 1
        BIN_4x4 = 2
        SUB_2x2 = 3
        SUB_4x4 = 4

        # Init the board
        self.sensor_dll = SensorDll(LIB_NAME, decode_frame)
        self.sensor_dll.init_board()

        # Configure the sensor
        bits_per_pixel = BITS_10
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_START, 2)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_X_END, 641)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_START, 2)
        self.sensor_dll.Write16(VD55G0_REG_CONTEXTS_OUT_ROI_Y_END, 601)
        self.sensor_dll.Write16(VD55G0_REG_STATICS_FORMAT_CTRL, bits_per_pixel)
        self.sensor_dll.Write8(VD55G0_REG_CONTEXTS_READOUT_CTRL, BIN_2x2)
        self.sensor_dll.Write16(VD55G0_REG_STATICS_ORIENTATION, 0x1)

        self.sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_EXT_CLOCK, 12000000)
        self.sensor_dll.Write32(VD55G0_REG_SENSOR_SETTINGS_CLK_PLL_MIPI, 1200000000)

    def run(self):
                
        t0 = time.time()
        cnt = 0

        self.sensor_dll.start_stream()
        width = self.sensor_dll.Read16(VD55G0_REG_STATUS_OUT_ROI_X_SIZE)

        try:
            while not self.stop_evt.is_set():
                metadata, frame = self.sensor_dll.get_image(width)


                # ensure size
                if frame.shape[1] != REQ_W or frame.shape[0] != REQ_H:
                    frame = cv2.resize(frame, (REQ_W, REQ_H), interpolation=cv2.INTER_NEAREST)

                with self.lock:
                    self.frame = frame
                    self.idx += 1

                cnt += 1
                # FPS meter (1-sec window)
                now = time.time()
                if now - t0 > 1.0:
                    self.fps = cnt / (now - t0)
                    t0 = now
                    cnt = 0
        except KeyboardInterrupt:
            pass
        finally:
            print("DONE")
            # Stop the sensor
            self.sensor_dll.stop_stream()
            self.sensor_dll.deinit_board()
        
    def get_latest(self):
        with self.lock:
            return (None,None,None) if self.frame is None else self.frame.copy(), self.idx, self.fps
        

def change_alpha1(sender, app_data, user_data):
    global cfg
    cfg.alpha1 = app_data

def change_alpha2(sender, app_data, user_data):
    global cfg
    cfg.alpha2 = app_data

def change_VBIP_LPF_alpha(sender, app_data, user_data):
    global cfg
    cfg.VBIP_LPF_alpha = app_data

def change_lambda_OPL(sender, app_data, user_data):
    global cfg
    cfg.lambda_OPL = app_data
    
def change_G_A_zero(sender, app_data, user_data):
    global cfg
    cfg.G_A_zero = app_data
    
def change_lambda_A(sender, app_data, user_data):
    global cfg
    cfg.lambda_A = app_data
    
def change_p1(sender, app_data, user_data):
    global cfg
    cfg.p1 = app_data
    
def change_I_bias_on(sender, app_data, user_data):
    global cfg
    cfg.I_bias_on = app_data

def change_I_bias_off(sender, app_data, user_data):
    global cfg
    cfg.I_bias_off = app_data
    
def change_p2(sender, app_data, user_data):
    global cfg
    cfg.p2 = app_data

def change_p3(sender, app_data, user_data):
    global cfg
    cfg.p3 = app_data
    
def change_U_rest_on(sender, app_data, user_data):
    global cfg
    cfg.U_rest_on = app_data

def change_U_rest_off(sender, app_data, user_data):
    global cfg
    cfg.U_rest_off = app_data
    
def change_U_th_on(sender, app_data, user_data):
    global cfg
    cfg.U_th_on = app_data
    
def change_U_th_off(sender, app_data, user_data):
    global cfg
    cfg.U_th_off = app_data
    
def change_U_reset_on(sender, app_data, user_data):
    global cfg
    cfg.U_reset_on = app_data
    
def change_U_reset_off(sender, app_data, user_data):
    global cfg
    cfg.U_reset_off = app_data
    

def decode_HPF_type(str_type):
    if str_type == "SUB":
        return HPF_MODE.SUB
    elif str_type == "MODE1":
        return HPF_MODE.MODE1
    elif str_type == "MODE2":
        return HPF_MODE.MODE2
    else:
        raise NotImplementedError(f"The selected HPF type of {str_type} is not recognized")    

def change_HPF1_type(sender, app_data, user_data):
    global cfg
    cfg.HPF1_type = decode_HPF_type(app_data)

def change_HPF2_type(sender, app_data, user_data):
    global cfg
    cfg.HPF2_type = decode_HPF_type(app_data)
    
def decode_blur_type(str_type):
    if str_type == "BLUR_AVG":
        return BLUR_MODE.BLUR_AVG
    elif str_type == "BLUR_MAX":
        return BLUR_MODE.BLUR_MAX
    elif str_type == "BLUR_NONE":
        return BLUR_MODE.BLUR_NONE
    else:
        raise NotImplementedError(f"The selected blur type of {str_type} is not recognized")
    
def change_blur1_type(sender, app_data, user_data):
    global cfg
    cfg.blur1_mode = decode_blur_type(app_data)
    
def change_blur2_type(sender, app_data, user_data):
    global cfg
    cfg.blur2_mode = decode_blur_type(app_data)

def toggle_apply_log_to_input(sender, app_data, user_data):
    global cfg
    cfg.apply_log_to_input = app_data
    
def toggle_turn_off_feedback(sender, app_data, user_data):
    global cfg
    cfg.turn_off_feedback = app_data
    
def reset_callback():
    global simulator, cfg, current_t
    simulator.reset()
    print("Simulator reset")
    current_t = 1
    
def float_to_SCAMP_img(inp):
    return np.clip( 255. * (1.0 + inp), 0, 255).astype(np.uint8)

def map_01_to_texture_img(inp): 
    return np.clip(inp.cpu().numpy(), 0.0, 1.0)

def slider_callback(sender, app_data, user_data):
    print(f"Slider changed value to: {app_data}")

def set_testpoint1(sender, appdata):
    global selected_testpoint1
    selected_testpoint1 = appdata
    print(selected_testpoint1)

def update_texture():
    global prev_time, worker, image_idx
    
    frame, curr_idx, fps = worker.get_latest()
    
    if curr_idx == image_idx:
        return
    image_idx = curr_idx

    norm_frame = frame / 1023.0

    input_image_data = np.ones((height, width, 4))
    input_image_data[:, :, 0] = norm_frame[:, :]
    input_image_data[:, :, 1] = norm_frame[:, :]
    input_image_data[:, :, 2] = norm_frame[:, :]
    dpg.set_value(input_image_tag, input_image_data.flatten().tolist())
    
    
    current = time.time()
    dt = current - prev_time
    prev_time = current

    # compute the sillion retina pipeline
    pos, neg = compute_sillicon_retina_pipeline(norm_frame)

    # display the testpoints
    display_from_sillicon_retina_pipeline(selected_testpoint1, display1_tag, pos, neg)

def compute_sillicon_retina_pipeline(input_img):
    global simulator, cfg, current_t
    current_idx = current_t % 2 
    
    input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).to("cuda")
    simulator.cfg = cfg
    pos, neg = simulator.forward(input_tensor, current_t)
    current_t += 1
    return pos[0], neg[0]


def display_from_sillicon_retina_pipeline(testpoint, tag, pos, neg):
    global simulator, width, height, current_t
    new_texture_data = np.ones((height, width, 4))
    
    if testpoint == "events":
        img = colorize_two_maps(pos, neg).cpu().numpy()
        new_texture_data[:, :, 0] = img[:, :, 0]
        new_texture_data[:, :, 1] = img[:, :, 1]
        new_texture_data[:, :, 2] = img[:, :, 2]
    else:
        buffer = simulator.__getattr__(testpoint)[current_t % 2]
        new_texture_data[:, :, 0] = map_01_to_texture_img(buffer)
        new_texture_data[:, :, 1] = map_01_to_texture_img(buffer)
        new_texture_data[:, :, 2] = map_01_to_texture_img(buffer)

    dpg.set_value(tag, new_texture_data.flatten().tolist())

if __name__ == "__main__":
    global worker
    worker = CamWorker()
    worker.start()
    time.sleep(1.0)
    dpg.create_context()

    el = np.ones((width, height, 4))

    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width=width, height=height,
                                default_value=el.flatten().tolist(), tag=input_image_tag)
        dpg.add_dynamic_texture(width=width, height=height,
                                default_value=el.flatten().tolist(), tag=display1_tag)

    with dpg.window(label="Sillion retina", height=img_upscale * height + 35, width=2 * img_upscale * width + 10):
        with dpg.group(horizontal=True):
            dpg.add_image(input_image_tag, width=img_upscale * width, height=img_upscale * height)
            dpg.add_image(display1_tag, width=img_upscale * width, height=img_upscale * height)

    with dpg.window(label="Sillion retina Options", pos=(2 * img_upscale * width + 20, 0), width=300):
        
        named_buffers = [name for name, _ in simulator.named_buffers(recurse=False)]
        options = named_buffers + ["events"]
        
        dpg.add_combo(options, label="Inspect", default_value="events", callback=set_testpoint1)
        
        dpg.add_slider_float(label="alpha 1", min_value= 0.0, max_value = 1.0, default_value=0.5, callback=change_alpha1)
        dpg.add_slider_float(label="alpha 2", min_value= 0.0, max_value = 1.0, default_value=0.5, callback=change_alpha2)
        dpg.add_slider_float(label="VBIP_LPF_alpha", min_value= 0.0, max_value = 1.0, default_value=0.5, callback=change_VBIP_LPF_alpha)
        dpg.add_slider_float(label="lambda_OPL", min_value= 0.0, max_value = 10.0, default_value=3.0, callback=change_lambda_OPL)
        dpg.add_slider_float(label="G_A_zero", min_value= 0.0, max_value = 1.0, default_value=0.5, callback=change_G_A_zero)
        dpg.add_slider_float(label="lambda_A", min_value= 0.0, max_value = 1.0, default_value=0.5, callback=change_lambda_A)
        dpg.add_slider_float(label="p1", min_value= 0.0, max_value = 1.0, default_value=0.25, callback=change_p1)
        dpg.add_slider_float(label="I_bias_on", min_value = -1.0, max_value = 1.0, default_value=0.01, callback=change_I_bias_on)
        dpg.add_slider_float(label="I_bias_off", min_value = -1.0, max_value = 1.0, default_value=0.01, callback=change_I_bias_off)
        dpg.add_slider_float(label="p2", min_value= 0.0, max_value = 4.0, default_value=1.0, callback=change_p2)
        dpg.add_slider_float(label="p3", min_value= 0.0, max_value = 4.0, default_value=0.2, callback=change_p3)

        dpg.add_slider_float(label="U_rest_on", min_value = -1.0, max_value = 1.0, default_value=-0.02, callback=change_U_rest_on)
        dpg.add_slider_float(label="U_rest_off", min_value = -1.0, max_value = 1.0, default_value=0.02, callback=change_U_rest_off)
        
        dpg.add_slider_float(label="U_th_on", min_value = -1.0, max_value = 1.0, default_value=0.05, callback=change_U_th_on)
        dpg.add_slider_float(label="U_th_off", min_value = -1.0, max_value = 1.0, default_value=0.05, callback=change_U_th_off)

        dpg.add_slider_float(label="U_reset_on", min_value = -1.0, max_value = 1.0, default_value=0.00, callback=change_U_reset_on)
        dpg.add_slider_float(label="U_reset_off", min_value = -1.0, max_value = 1.0, default_value=0.00, callback=change_U_reset_off)
        
        dpg.add_combo(["SUB", "MODE1", "MODE2"], label="HPF1 type", default_value="SUB", callback=change_HPF1_type)        
        dpg.add_slider_float(label="HPF1_param", min_value = -1.0, max_value = 1.0, default_value=0.00, callback=change_U_reset_on)

        dpg.add_combo(["SUB", "MODE1", "MODE2"], label="HPF2 type", default_value="SUB", callback=change_HPF2_type)        
        dpg.add_slider_float(label="HPF2_param", min_value = -1.0, max_value = 1.0, default_value=0.00, callback=change_U_reset_off)

        dpg.add_combo(["BLUR_AVG", "BLUR_MAX", "BLUR_NONE"], label="Blur 1 type", default_value="BLUR_NONE", callback=change_blur1_type)
        dpg.add_combo(["BLUR_AVG", "BLUR_MAX", "BLUR_NONE"], label="Blur 2 type", default_value="BLUR_NONE", callback=change_blur2_type)
        
        dpg.add_checkbox(label="apply log to input", default_value=False, callback=toggle_apply_log_to_input)
        dpg.add_checkbox(label="turn off feedback", default_value=False, callback=toggle_turn_off_feedback)

        dpg.add_button(label="Reset", callback=reset_callback)


    dpg.create_viewport(title='Sillion retina', width=APP_WIDTH, height=APP_HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    prev_time = time.time()
    #dpg.show_metrics()

    while dpg.is_dearpygui_running():
        # insert here any code you would like to run in the render loop
        # you can manually stop by using stop_dearpygui()
        update_texture()
        dpg.render_dearpygui_frame()

    dpg.start_dearpygui()
    dpg.destroy_context()   