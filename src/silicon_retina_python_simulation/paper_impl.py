import os

import cv2
from scipy.signal import convolve2d
import argparse
import numpy as np
import imageio
from datetime import datetime

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
        annotated = annotate_image(pm1_to_uint8(img), label)
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
fps = 15

a1 = 0.875
b1 = 1 - a1

a2 = 0.875
b2 = 1 - a2

a3 = 0.9
b3 = 1 - a3

a4 = 0.25
b4 = 1 - a4

a5 = 0.9
b5 = 1 - a5

alpha = 0.5
g_0 = 0.1
lambda_A = 0.5
map_contrast = None
Vm = None
map5 = None

I_bias = 0
p1 = 0.25
p2 = 1.0
V_rest = -0.5
threshold = 0.2
sub = 0.5

K1 = gaussian_kernel(3)
K2 = gaussian_kernel(3)
K3 = gaussian_kernel(3)

def change_alpha1(val):
    global a1, b1
    a1 = val / 100.
    b1 = 1. - a1

def change_alpha2(val):
    global a2, b2
    a2 = val / 100.
    b2 = 1. - a2

def change_alpha3(val):
    global a3, b3
    a3 = val / 100.
    b3 = 1. - a3

def change_alpha4(val):
    global a4, b4
    a4 = val / 100.
    b4 = 1. - a4

def change_alpha5(val):
    global a5, b5
    a5 = val / 100.
    b5 = 1. - a4

def change_alpha(val):
    global alpha
    alpha = val / 100

def change_g_0(val):
    global g_0
    g_0 = val / 100

def change_lambda_A(val):
    global lambda_A
    lambda_A = val / 100

def change_I_bias(val):
    global I_bias
    I_bias = val / 100 - 1.

def change_p1(val):
    global p1
    p1 = val / 100

def change_p2(val):
    global p2
    p2 = val / 100

def change_V_rest(val):
    global V_rest
    V_rest = val / 100 - 1.0

def change_threshold(val):
    global threshold
    threshold = val / 100 - 1.0

def change_sub(val):
    global sub
    sub = val / 100 - 1.0


def scamp5_quantize(alpha, bits):
    return alpha

def apply_lowpass(incoming, LPF_map, a, b):
    if LPF_map is None:
        LPF_map = np.zeros_like(incoming)

    LPF_map_next = a * LPF_map + b * incoming
    return LPF_map_next

def apply_highpass(incoming, map, a, b):
    if map is None:
        map = np.zeros_like(incoming)

    next_map = a * map + b * incoming
    return incoming - next_map, next_map

def apply_convolution(image, kernel):
    """Fast 2D convolution using SciPy (same as OpenCV filter2D)."""
    return convolve2d(image, kernel, mode='same', boundary='symm')

def compute_I_opl(X, a1, a2, a3, b1, b2, b3, K1, K2, map2):
    global G_C, E_tauC, T_wr, G_s, E_tauS
    print(a1)
    G_C = apply_convolution(X, K1)
    E_tauC = apply_lowpass(G_C, E_tauC, a1, b1)
    T_wr, map2 = apply_highpass(E_tauC, map2, a2, b2)
    G_s = apply_convolution(T_wr, K2)
    E_tauS = apply_lowpass(G_s, E_tauS, a3, b3)

    S = E_tauS
    C = T_wr
    I_opl = C- S
    return I_opl, E_tauC, map2, E_tauS

# compute the bipolar-level contrast gain control
"""
    C \frac{V_{bip}}{dt} = I_{opl} - g_{A} V_{bip} (7)
    g_{A} = G_{A} * E_{A} * Q(V_{bip})             (8)
    Q(V_{bip}) = g_{A}^{0} + \lambda_{A} V_{bip}^2 (9)
    
    this will be done using euler method
    
    V_{bip}(t+\delta t) = V_{bip} + \delta t \frac{V_{bip}}{dt}
"""
def contrast_gain_control(V_bip, I_opl, K3, alpha, a, b,
                          g_0, lambda_A, map, quant_bits):
    global Q, g_A, g_A_blurred

    # Compute the Q(V_{bip})
    # where alpha = \frac{\delta t}{C}
    Q = g_0 + lambda_A * V_bip * V_bip

    # Compute E_A
    map = apply_lowpass(Q, map, a, b)

    # compute the g_{A}
    g_A_blurred = apply_convolution(map, K3)

    g_A = np.clip(g_A_blurred, 0, 1)
    g_A = scamp5_quantize(g_A, bits=quant_bits)

    new_V_bip = V_bip + alpha * (I_opl - g_A * V_bip)

    return new_V_bip, map

"""
* 
*		The computation here looks like following 
*		the input is I_{gang}
* 
*		du/dt = 1/tau ( R * I_{gang}(t)  - |u(t) - u_{rest}| )
* 
*		and is being solved by euler method 
* 
*		u(t + \delta t) = u(t) + \delta t / tau (R * I_{gang}(t) - |u(t) - u_{rest}|)
* 
*		where the computation will be done as following 
* 
*		u(t) + p_{2} * (p_{1} * I_{gang}(t) - |u(t) - u_{rest}|)
*
*		where p_{2} and p_{1} will represent the 5 bit fixed point values in a for b_1.b_2 b_3 b_4 b_5
"""
def excitatory_current_upon_ganglion_cell(Vbip, I_bias,
                                          Vm, p1, p2,
                                          a5, b5, map,
                                          Vrest,
                                          threshold,
                                          sub):

    hpf_out, map = apply_highpass(Vbip, map, a5, b5)

    Igang = hpf_out.copy()
    Igang[Igang < 0] = 0
    Igang += I_bias

    Vm = Vm + p2 * (Igang - p1 * np.abs(Vm - Vrest))

    events = Vm > threshold
    Vm[events] -= sub

    return Vm, events, Igang, map, hpf_out

def main():
    global map2, V_bip, map_contrast, out, fps, pad, I_bias, Vm, p1, p2, V_rest, threshold, sub, map5
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action="store_true", help='live video feed from webcam')
    parser.add_argument("--path", type=str, default="", help="")
    args = parser.parse_args()

    now = datetime.now()
    formatted = now.strftime("%B %d – %H:%M")
    with imageio.get_writer(f"output_{formatted}.mp4", fps=30, codec="libx264") as writer:

        if args.live:
            cap = cv2.VideoCapture(0)  # Use default webcam
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                exit(1)

            while True:
                ret, frame = cap.read()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (256, 256))


                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            path = args.path
            if path == "":
                print("--live flag was not specified, however the dataset path was not set ! one of the two are required")
                return -1

            if not os.path.exists(path):
                path = os.path.abspath(path)
                print(f"path {path} does not exists !")
                return -1

            if not os.path.isdir(path):
                path = os.path.abspath(path)
                print(f"path {path} must be a directory !")
                return -1

            cv2.namedWindow('frames')
            cv2.createTrackbar('a1', 'frames', int(a1*100), 100, change_alpha1)
            cv2.createTrackbar('a2', 'frames', int(a2*100), 100, change_alpha2)
            cv2.createTrackbar('a3', 'frames', int(a3*100), 100, change_alpha3)
            cv2.createTrackbar('a4', 'frames', int(a4*100), 100, change_alpha4)
            cv2.createTrackbar('a5', 'frames', int(a5*100), 100, change_alpha5)
            cv2.createTrackbar("alpha", "frames", int(alpha * 100), 100, change_alpha)
            cv2.createTrackbar('g_0', 'frames', int(g_0*100), 100, change_g_0)
            cv2.createTrackbar('lambda_A', 'frames', int(lambda_A*100), 100, change_lambda_A)
            cv2.createTrackbar("I_bias", "frames", int((1+I_bias)*100), 200, change_I_bias)
            cv2.createTrackbar("p1", "frames", int(p1 * 100), 200, change_p1)
            cv2.createTrackbar("p2", "frames", int(p2 * 100), 200, change_p2)
            cv2.createTrackbar("V_rest", "frames", int((1 + V_rest) * 100), 200, change_V_rest)
            cv2.createTrackbar("threshold", "frames", int((1 + threshold) * 100), 200, change_threshold)
            cv2.createTrackbar("sub", "frames", int((1 + sub) * 100), 200, change_sub)

            filepaths = sorted([os.path.join(path, _path) for _path in os.listdir(path)])

            for filepath in filepaths:
                img = cv2.imread(filepath)
                img = img[:,:, 0]
                input_img = (img - 128.0) / 128.0

                if pad is None:
                    pad = -np.ones_like(input_img)

                I_opl, E_tauC, map2, E_tauS = compute_I_opl(input_img,
                                            a1=a1,
                                            a2=a2,
                                            a3=a3,
                                            b1=b1,
                                            b2=b2,
                                            b3=b3,
                                            K1=K1,
                                            K2=K2,
                                            map2=map2)

                if V_bip is None:
                    V_bip = np.zeros_like(input_img)

                V_bip, map_contrast = contrast_gain_control(V_bip, I_opl, K3, alpha=alpha,
                                    a = a4,
                                    b = b4,
                                    g_0=g_0,
                                    lambda_A=lambda_A,
                                    map=map_contrast,
                                    quant_bits=4)

                if Vm is None:
                    Vm = np.zeros_like(input_img)

                Vm, events, Igang, map5, hpf_out = excitatory_current_upon_ganglion_cell(Vbip=V_bip,
                                                                                        I_bias=I_bias,
                                                                                        Vm=Vm,
                                                                                        p1=p1,
                                                                                        p2=p2,
                                                                                        a5=a5, b5=b5,
                                                                                        map=map5,
                                                                                        Vrest=V_rest,
                                                                                        threshold=threshold,
                                                                                        sub=sub)





                stage1 = concatenate_images_with_labels([input_img, G_C, E_tauC, T_wr, G_s, E_tauS, I_opl],
                                                    ["Input", "G_C", "E_tauC", "T_wr", "G_s", "E_tauS", "I_opl"])

                stage2 = concatenate_images_with_labels([Q, map_contrast, g_A, V_bip, pad, pad, pad],
                                                        ["Q", "map_contrast", "g_A", "V_bip", "", "", ""])

                stage3 = concatenate_images_with_labels([map5, hpf_out, Igang, Vm, events, pad, pad],
                                                        labels=["hpf_map", "hpf_out", "I_gang", "Vm", "events", "", ""])

                final = np.concatenate((stage1, stage2, stage3), axis=0)


                writer.append_data(final)


                cv2.imshow('frames', final)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    main()
