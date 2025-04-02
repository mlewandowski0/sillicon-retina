import dearpygui.dearpygui as dpg
import random
import numpy as np
import cv2
import time
from retina_func import *

width, height = 256, 256
img_upscale = 2

APP_WIDTH = 1600
APP_HEIGHT = 1024
input_image_tag = "input_image_tag"
display1_tag = "display1_tag"




cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)


dpg.create_context()

el = np.ones((width, height, 4))

with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=width, height=height,
                            default_value=el.flatten().tolist(), tag=input_image_tag)
    dpg.add_dynamic_texture(width=width, height=height,
                            default_value=el.flatten().tolist(), tag=display1_tag)

def update_texture():
    global prev_time
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256)) / 256.

    new_texture_data = np.ones((width, height, 4))
    new_texture_data[:, :, 0] = frame[:, :]
    new_texture_data[:, :, 1] = frame[:, :]
    new_texture_data[:, :, 2] = frame[:, :]

    dpg.set_value(input_image_tag, new_texture_data.flatten().tolist())
    current = time.time()
    dt = current - prev_time
    prev_time = current


def inspect_selected(sender, app_data):
    print("Selected item : ", app_data)

with dpg.window(label="Sillion retina", height=2 * img_upscale * height + 20, width=img_upscale * width):
    dpg.add_image(input_image_tag, width=img_upscale*width, height=img_upscale*height)
    dpg.add_image(display1_tag, width=img_upscale*width, height=img_upscale*height)


with dpg.window(label="Sillion retina Options", pos=(img_upscale * width + 50,0), width=300):
    dpg.add_combo(["", "Option 2", "Option 3"],
                  label="Inspect",
                  default_value="Option 1",
                  callback=inspect_selected)

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