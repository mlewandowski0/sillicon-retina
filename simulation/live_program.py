import dearpygui.dearpygui as dpg
import random
import numpy as np
import cv2
import time
from sillicon_retina_scamp_approx import *

G_C, E_tauC, T_wr, G_s, E_tauS, Q = None, None, None, None, None, None
map2, V_bip, out, g_A, pad, map_contrast, Vm = None, None, None, None, None, None, None
map5, I_opl, events, Igang, hpf_out = None, None, None, None, None

apply_1_formulation_of_LPF = True
total_bits = 5
frac_bits = 4
lsb = 1 / 2**frac_bits
g_0 = 0
lambda_A = 0.5

a1 = lsb
a2 = lsb
a3 = lsb
a4 = lsb
a5 = lsb
alpha = 0.5

I_bias = 0
p1 = 0.25
p2 = 1.0
V_rest = -50
threshold = 20
sub = 50

selected_testpoint1 = "Vm"

K1 = gaussian_kernel(3)
K2 = gaussian_kernel(3)
K3 = gaussian_kernel(3)

width, height = 256, 256
img_upscale = 2
enable_blur_1 = True


APP_WIDTH = 1600
APP_HEIGHT = 1024
input_image_tag = "input_image_tag"
display1_tag = "display1_tag"

def toggle_blur_1(sender, app_data, user_data):
    global enable_blur_1
    enable_blur_1 = app_data

def change_alpha1(sender, app_data, user_data):
    global a1
    a1 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha2(sender, app_data, user_data):
    global a2
    a2 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha3(sender, app_data, user_data):
    global a3
    a3 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha4(sender, app_data, user_data):
    global a4
    a4 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha5(sender, app_data, user_data):
    global a5
    a5 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha(sender, app_data, user_data):
    global alpha
    alpha = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_g_0(sender, app_data, user_data):
    global g_0
    g_0 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_lambda_A(sender, app_data, user_data):
    global lambda_A
    lambda_A = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_I_bias(sender, app_data, user_data):
    global I_bias
    I_bias = app_data

def change_p1(sender, app_data, user_data):
    global p1
    p1 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_p2(sender, app_data, user_data):
    global p2
    p2 = integer_to_fixed(app_data, total_bits=total_bits, frac_bits=frac_bits)

def change_V_rest(sender, app_data, user_data):
    global V_rest
    V_rest = app_data

def change_threshold(sender, app_data, user_data):
    global threshold
    threshold = app_data

def change_sub(sender, app_data, user_data):
    global sub
    sub = app_data

def reset_memory_of_pipeline():
    global G_c, E_tauC, T_wr, G_s, E_tauS, Q
    global map2, V_bip, out, g_A_blurred, pad, map_contrast, Vm
    global map5, I_opl, events, Igang, hpf_out

    G_C, E_tauC, T_wr, G_s, E_tauS, Q = None, None, None, None, None, None
    map2, V_bip, out, g_A, pad, map_contrast, Vm = None, None, None, None, None, None, None
    map5, I_opl, events, Igang, hpf_out = None, None, None, None, None


def SCAMP_img_to_0to1_range(img):
    return np.clip(img + 128, 0, 255) / 255.

def slider_callback(sender, app_data, user_data):
    print(f"Slider changed value to: {app_data}")

def set_testpoint1(sender, appdata):
    global selected_testpoint1
    selected_testpoint1 = appdata
    print(selected_testpoint1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)


def compute_sillicon_retina_pipeline(input_img):
    global I_opl, E_tauC, map2, E_tauS, G_C, T_wr, G_s
    global a1, a2, a3, K1, K2, apply_1_formulation_of_LPF, threshold, sub
    global V_bip, K3, alpha, a4, g_0, lambda_A, map_contrast, I_bias, Vm, p1, p2, a5, map5, V_rest
    global Vm, events, Igang, map5, hpf_out, g_A

    I_opl, E_tauC, map2, E_tauS, G_C, T_wr, G_s = compute_I_opl(input_img,
                                                                a1=a1,
                                                                a2=a2,
                                                                a3=a3,
                                                                K1=K1,
                                                                K2=K2,
                                                                map2=map2,
                                                                apply_lpf1=apply_1_formulation_of_LPF,
                                                                E_tauC=E_tauC,
                                                                E_tauS=E_tauS,
                                                                enable_blur_1=enable_blur_1)

    if V_bip is None:
        V_bip = np.zeros_like(input_img)

    V_bip, map_contrast, Q, g_A = contrast_gain_control(V_bip, I_opl, K3, alpha=alpha,
                                                        a=a4,
                                                        g_0=g_0,
                                                        lambda_A=lambda_A,
                                                        map=map_contrast,
                                                        apply_lpf1=apply_1_formulation_of_LPF,
                                                        quant_bits=4)

    if Vm is None:
        Vm = np.zeros_like(input_img)

    Vm, events, Igang, map5, hpf_out = excitatory_current_upon_ganglion_cell(Vbip=V_bip,
                                                                             I_bias=I_bias,
                                                                             Vm=Vm,
                                                                             p1=p1,
                                                                             p2=p2,
                                                                             a5=a5,
                                                                             map=map5,
                                                                             Vrest=V_rest,
                                                                             threshold=threshold,
                                                                             sub=sub,
                                                                             apply_lpf1=apply_1_formulation_of_LPF)

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
    frame = cv2.resize(frame, (256, 256))

    scamp_frame = (frame - 128.)
    norm_frame = frame / 255.0

    new_texture_data = np.ones((width, height, 4))
    new_texture_data[:, :, 0] = norm_frame[:, :]
    new_texture_data[:, :, 1] = norm_frame[:, :]
    new_texture_data[:, :, 2] = norm_frame[:, :]
    dpg.set_value(input_image_tag, new_texture_data.flatten().tolist())
    current = time.time()
    dt = current - prev_time
    prev_time = current

    # compute the sillion retina pipeline
    compute_sillicon_retina_pipeline(scamp_frame)

    # display the testpoints
    display_from_sillicon_retina_pipeline(selected_testpoint1, display1_tag)


def display_from_sillicon_retina_pipeline(testpoint, tag):
    global G_C, E_tauC, T_wr, G_s, E_tauS, I_opl, Q, map2, V_bip, g_A, Vm, Igang, map5, hpf_out, events

    img = None

    if testpoint == "G_C":
        img = np.clip(G_C + 128, 0, 255) / 255.
    elif testpoint == "E_tauC":
        img = np.clip(E_tauC + 128, 0, 255) / 255.
    elif testpoint == "T_wr":
        img = np.clip(T_wr + 128, 0, 255) / 255.
    elif testpoint == "G_s":
        img = np.clip(G_s + 128, 0, 255) / 255.
    elif testpoint == "E_tauS":
        img = np.clip(E_tauS + 128, 0, 255) / 255.
    elif testpoint == "I_opl":
        img = np.clip(I_opl + 128, 0, 255) / 255.
    elif testpoint == "Q":
        img = np.clip(Q + 128, 0, 255) / 255.
    elif testpoint == "map2":
        img = np.clip(map2 + 128, 0, 255) / 255.
    elif testpoint == "V_bip":
        img = np.clip(V_bip + 128, 0, 255) / 255.
    elif testpoint == "g_A":
        img = g_A
    elif testpoint == "Vm":
        img = np.clip(Vm + 128, 0, 255) / 255.
    elif testpoint == "Igang":
        img = np.clip(Igang + 128, 0, 255) / 255.
    elif testpoint == "events":
        img = events

    else:
        raise NotImplemented(f"the selected value of {testpoint} was not recognized")


    new_texture_data = np.ones((width, height, 4))
    new_texture_data[:, :, 0] = img[:, :]
    new_texture_data[:, :, 1] = img[:, :]
    new_texture_data[:, :, 2] = img[:, :]
    dpg.set_value(tag, new_texture_data.flatten().tolist())



with dpg.window(label="Sillion retina", height=img_upscale * height + 50, width=2 * img_upscale * width + 10):
    with dpg.group(horizontal=True):
        dpg.add_image(input_image_tag, width=img_upscale * width, height=img_upscale * height)
        dpg.add_image(display1_tag, width=img_upscale * width, height=img_upscale * height)

with dpg.window(label="Sillion retina Options", pos=(2 * img_upscale * width + 70, 0), width=300):
    dpg.add_combo(["G_C", "E_tauC", "T_wr", "G_s", "E_tauS", "I_opl", "Q", "map2", "V_bip",
                   "g_A", "Vm", "Igang", "events"],
                  label="Inspect",
                  default_value="Vm",
                  callback=set_testpoint1)
    dpg.add_slider_int(label="alpha 1", min_value=1, max_value=31, default_value=1, callback=change_alpha1)
    dpg.add_slider_int(label="alpha 2", min_value=1, max_value=31, default_value=1, callback=change_alpha2)
    dpg.add_slider_int(label="alpha 3", min_value=1, max_value=31, default_value=1, callback=change_alpha3)
    dpg.add_slider_int(label="alpha 4", min_value=1, max_value=31, default_value=1, callback=change_alpha4)
    dpg.add_slider_int(label="alpha 5", min_value=1, max_value=31, default_value=1, callback=change_alpha5)
    dpg.add_slider_int(label="alpha", min_value=1, max_value=31, default_value=8, callback=change_alpha)
    dpg.add_slider_int(label="g_0", min_value=0, max_value=31, default_value=0, callback=change_g_0)
    dpg.add_slider_int(label="lambda_A", min_value=1, max_value=31, default_value=8, callback=change_lambda_A)
    dpg.add_slider_int(label="I_bias", min_value=-128, max_value=127, default_value=0, callback=change_I_bias)
    dpg.add_slider_int(label="p1", min_value=1, max_value=31, default_value=1, callback=change_p1)
    dpg.add_slider_int(label="p2", min_value=1, max_value=31, default_value=16, callback=change_p2)
    dpg.add_slider_int(label="V_rest", min_value=-128, max_value=127, default_value=-50, callback=change_V_rest)
    dpg.add_slider_int(label="threshold", min_value=-128, max_value=127, default_value=20, callback=change_threshold)
    dpg.add_slider_int(label="sub", min_value=-128, max_value=127, default_value=20, callback=change_sub)
    dpg.add_checkbox(label="Enable 1st Blur", default_value=True, callback=toggle_blur_1)




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