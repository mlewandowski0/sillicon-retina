import os
import cv2
import argparse
import numpy as np
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
fps = 15

g_0 = 0
lambda_A = 0.5
map_contrast = None
Vm = None
map5 = None

event_prev = None
event_thresh = 20

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

K1 = gaussian_kernel(3)
K2 = gaussian_kernel(3)
K3 = gaussian_kernel(3)


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

def scamp5_quantize(alpha, bits):
    return alpha

def main():
    global map2, V_bip, map_contrast, out, fps, pad, I_bias, Vm, p1, p2, V_rest, threshold, sub
    global E_tauC, E_tauS, map5, event_prev
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action="store_true", help='live video feed from webcam')
    parser.add_argument("--path", type=str, default="", help="")
    args = parser.parse_args()



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
        cv2.createTrackbar('a1', 'frames', int(a1 / lsb), no_of_values, change_alpha1)
        cv2.createTrackbar('a2', 'frames', int(a2 / lsb), no_of_values, change_alpha2)
        cv2.createTrackbar('a3', 'frames', int(a3 / lsb), no_of_values, change_alpha3)
        cv2.createTrackbar('a4', 'frames', int(a4 / lsb), no_of_values, change_alpha4)
        cv2.createTrackbar('a5', 'frames', int(a5 / lsb), no_of_values, change_alpha5)

        cv2.createTrackbar("alpha", "frames", int(alpha / lsb), no_of_values, change_alpha)
        cv2.createTrackbar('g_0 + 128', 'frames', g_0 + 128, 255, change_g_0)
        cv2.createTrackbar('lambda_A', 'frames', int(lambda_A / lsb), no_of_values, change_lambda_A)
        cv2.createTrackbar("I_bias", "frames", I_bias + 128, 255, change_I_bias)
        cv2.createTrackbar("p1", "frames", int(p1 / lsb), no_of_values, change_p1)
        cv2.createTrackbar("p2", "frames", int(p2 / lsb), no_of_values, change_p2)
        cv2.createTrackbar("V_rest", "frames", V_rest + 128, 255, change_V_rest)
        cv2.createTrackbar("threshold", "frames", threshold + 128, 255, change_threshold)
        cv2.createTrackbar("sub", "frames", sub + 128, 255, change_sub)

        filepaths = sorted([os.path.join(path, _path) for _path in os.listdir(path)])

        for filepath in filepaths:
            img = cv2.imread(filepath)
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




            event_map = np.zeros(input_img.shape + (3, ))

            if event_prev is None:
                event_prev = input_img.copy()
            else:
                diff = input_img - event_prev

                pos_mask = diff > event_thresh
                event_map[pos_mask] = [255, 0, 0]  # red

                neg_mask = diff < -event_thresh
                event_map[neg_mask] = [0, 0, 255]  # blue

                change_mask = np.abs(diff) > event_thresh
                event_prev[change_mask] = input_img[change_mask]


            stage1 = concatenate_images_with_labels([input_img, G_C, E_tauC, T_wr, G_s, E_tauS, I_opl],
                                                   ["Input", "G_C", "E_tauC", "T_wr", "G_s", "E_tauS", "I_opl"])

            stage2 = concatenate_images_with_labels([Q, map_contrast, g_A, V_bip, pad, pad, pad],
                                                    ["Q", "map_contrast", "g_A", "V_bip", "", "", ""])

            stage3 = concatenate_images_with_labels([map5, hpf_out, Igang, Vm, 127*events, pad, event_map],
                                                    labels=["hpf_map", "hpf_out", "I_gang", "Vm", "events", "", "event_map"])

            final = np.concatenate((stage1, stage2, stage3), axis=0)


            # Initialize the VideoWriter with the size of the final frame
            if out is None:
                height, width = final.shape[:2]
                out = cv2.VideoWriter("output.mp4",
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps,
                                      (width, height))

            out.write(final)


            cv2.imshow('frames', final)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()
