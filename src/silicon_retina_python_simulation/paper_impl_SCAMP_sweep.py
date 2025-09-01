import os

import cv2
from scipy.signal import convolve2d
import argparse
import numpy as np
from retina_func import integer_to_fixed
from tqdm import tqdm
from sillicon_retina_scamp_approx import *

apply_1_formulation_of_LPF = True
G_C = None
E_tauC = None
T_wr = None
G_s = None
E_tauS = None
Q = None
map2 = None
V_bip = None
out = None
g_A_blurred = None
pad = None
fps = None

g_0 = 0
lambda_A = 0.5
map_contrast = None
Vm = None
map5 = None


total_bits = 5
frac_bits = 4
lsb = 1 / 2**frac_bits

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

no_of_values = (1 << total_bits) - 1


def annotate_image(image, text, position=(5, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    annotated = image.copy()
    if len(text) > 0:
        cv2.putText(annotated, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return annotated


def concatenate_images_with_labels(images, labels, orientation='horizontal'):
    annotated_images = []
    for img, label in zip(images, labels):
        annotated = annotate_image(SCAMP5_img_to_regular(img), label)
        annotated_images.append(annotated)

    if orientation == 'horizontal':
        concatenated = cv2.hconcat(annotated_images)
    else:
        concatenated = cv2.vconcat(annotated_images)
    return concatenated


def uint8_to_pm1(img):
    return np.clip(img / 128 - 1., -1, 1)

def pm1_to_uint8(img):
    return np.clip(128*img + 128, 0., 255).astype(np.uint8)

def gaussian_kernel(kernel_size, sigma=1.0):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian kernel
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))

    # Normalize the kernel so that the sum is 1
    kernel = kernel / np.sum(kernel)
    return kernel

K1 = gaussian_kernel(3)
K2 = gaussian_kernel(3)
K3 = gaussian_kernel(3)


def SCAMP5_img_to_regular(img):
    return np.clip(img + 128, 0, 255).astype(np.uint8)

def change_alpha1(val):
    global a1
    a1 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha2(val):
    global a2
    a2 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha3(val):
    global a3
    a3 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha4(val):
    global a4
    a4 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha5(val):
    global a5
    a5 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_alpha(val):
    global alpha
    alpha = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_g_0(val):
    global g_0
    g_0 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_lambda_A(val):
    global lambda_A
    lambda_A = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_I_bias(val):
    global I_bias
    I_bias = val - 128

def change_p1(val):
    global p1
    p1 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_p2(val):
    global p2
    p2 = integer_to_fixed(val, total_bits=total_bits, frac_bits=frac_bits)

def change_V_rest(val):
    global V_rest
    V_rest = val - 128

def change_threshold(val):
    global threshold
    threshold = val - 128

def change_sub(val):
    global sub
    sub = val - 128


def main():
    global map2, V_bip, map_contrast, out, fps, pad, I_bias, Vm, p1, p2, V_rest, threshold, sub, map5, E_tauS, E_tauC
    global g_0, lambda_A, total_bits, frac_bits, alpha, a1, a2, a3, a4, I_bias, p1, p2, V_rest, threshold, sub
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str, default="", help="")
    parser.add_argument("--save_path", required=True, type=str, default="", help="")
    parser.add_argument("--g0", required=True, type=int, default = 0, help="g_{0}")
    parser.add_argument("--lambda_A", required=True, type=int, default =1, help="lambda_{A}")
    parser.add_argument("--total_bits", required=True, type=int, default = 5, help="width of the multiplicative terms")
    parser.add_argument("--frac_bits", required=True, type=int, default=4, help="fractional part of the total bits")
    parser.add_argument("--alpha1", required=True, type=int, default = 0.1, help="alpha 1")
    parser.add_argument("--alpha2", required=True, type=int, default = 0.1, help="alpha 2")
    parser.add_argument("--alpha3", required=True, type=int, default = 0.1, help="alpha 3")
    parser.add_argument("--alpha4", required=True, type=int, default = 0.1, help="alpha 4")
    parser.add_argument("--alpha5", required=True, type=int, default = 0.1, help="alpha 5")
    parser.add_argument("--alpha", required=True, type=int, default=0.5, help="alpha value")
    parser.add_argument("--I_bias", required=True, type=int, default=0, help="I_{bias}")
    parser.add_argument("--p1", required=True, type=int, default=0.25, help="p1")
    parser.add_argument("--p2", required=True, type=int, default = 1.0, help="p2")
    parser.add_argument("--V_rest", required=True, type=int, default =-50, help="v_{rest}")
    parser.add_argument("--threshold", required=True, type=int, default=20, help="threshold theta")
    parser.add_argument("--sub", required=True, type=int, default=20, help="the subtraction term, when the LIF fired")
    parser.add_argument("--fps",  type=int, default=15, help="frames per second for video")
    parser.add_argument("--target_size", type=int, default=128, help="width and height of the image")
    args = parser.parse_args()

    input_path = args.input_path
    save_path = args.save_path
    total_bits = args.total_bits
    frac_bits = args.frac_bits
    g_0 = args.g0
    lambda_A = integer_to_fixed(args.lambda_A, total_bits=total_bits, frac_bits=frac_bits)
    a1 = integer_to_fixed(args.alpha1, total_bits=total_bits, frac_bits=frac_bits)
    a2 = integer_to_fixed(args.alpha2, total_bits=total_bits, frac_bits=frac_bits)
    a3 = integer_to_fixed(args.alpha3, total_bits=total_bits, frac_bits=frac_bits)
    a4 = integer_to_fixed(args.alpha4, total_bits=total_bits, frac_bits=frac_bits)
    a5 = integer_to_fixed(args.alpha5, total_bits=total_bits, frac_bits=frac_bits)
    alpha = integer_to_fixed(args.alpha, total_bits=total_bits, frac_bits=frac_bits)
    I_bias = args.I_bias
    p1 = integer_to_fixed(args.p1, total_bits=total_bits, frac_bits=frac_bits)
    p2 = integer_to_fixed(args.p2, total_bits=total_bits, frac_bits=frac_bits)
    V_rest = args.V_rest
    threshold = args.threshold
    sub = args.sub
    fps = args.fps


    if not os.path.exists(input_path):
            path = os.path.abspath(input_path)
            print(f"path {input_path} does not exists !")
            exit(-1)

    if not os.path.isdir(input_path):
            path = os.path.abspath(input_path)
            print(f"path {path} must be a directory !")
            exit(-1)

    filepaths = sorted([os.path.join(input_path, _path) for _path in os.listdir(input_path)])

    param_name = f"g0={g_0}-lambda_A={lambda_A}-tb={total_bits}-fb={frac_bits}-a1={a1}-a2={a2}-a3={a3}-a4={a4}-a5={a5}-alpha={alpha}-I_bias={I_bias}-p1={p1}-p2={p2}-V_rest={V_rest}-thr={threshold}-sub={sub}.mp4"
    full_path = os.path.join(save_path, param_name)
    if os.path.exists(full_path):
        print(f"run {full_path} already exists !")
        exit(-2)

    for filepath in tqdm(filepaths):
            img = cv2.imread(filepath)
            img = cv2.resize(img, (args.target_size, args.target_size))
            img = img[:,:, 0]
            input_img = img - 128.

            if pad is None:
                pad = -np.ones_like(input_img)

            I_opl, E_tauC, map2, E_tauS, G_C, T_wr, G_s = compute_I_opl(input_img,
                                                                          a1=a1,
                                                                          a2=a2,
                                                                          a3=a3,
                                                                          K1=K1,
                                                                          K2=K2,
                                                                          map2=map2,
                                                                          apply_lpf1=apply_1_formulation_of_LPF,
                                                                          E_tauC=E_tauC,
                                                                          E_tauS=E_tauS)

            if V_bip is None:
                V_bip = np.zeros_like(input_img)

            V_bip, map_contrast, Q, g_A = contrast_gain_control(V_bip, I_opl, K3, alpha=alpha,
                                                                  a = a4,
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





            stage1 = concatenate_images_with_labels([input_img, G_C, E_tauC, T_wr, G_s, E_tauS, I_opl],
                                                   ["Input", "G_C", "E_tauC", "T_wr", "G_s", "E_tauS", "I_opl"])

            stage2 = concatenate_images_with_labels([Q, map_contrast, 127*g_A, V_bip, pad, pad, pad],
                                                    ["Q", "map_contrast", "g_A", "V_bip", "", "", ""])

            stage3 = concatenate_images_with_labels([map5, hpf_out, Igang, Vm, 127*events, pad, pad],
                                                    labels=["hpf_map", "hpf_out", "I_gang", "Vm", "events", "", ""])

            final = np.concatenate((stage1, stage2, stage3), axis=0)


            # Initialize the VideoWriter with the size of the final frame
            if out is None:
                height, width = final.shape[:2]
                print(full_path)
                out = cv2.VideoWriter(full_path,
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps,
                                      (width, height))

            out.write(final)


if __name__ == '__main__':
    main()
