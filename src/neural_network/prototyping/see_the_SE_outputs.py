import numpy as np

from esim_torch.silicon_retina_torch import SiliconRetinaVerboseFloatCUDA, SiliconRetinaConfig
import os
import torch
import cv2
import imageio
import math

quantization = 4
ffs = np.array([i / (2**quantization) for i in range(2**quantization)])


def find_float_closest_to_quantization(val):
    idx = np.argmin(np.abs(val - ffs))
    return ffs[idx]


labels_row1 = ["Input", "LPF1", "HPF1", "BLURRED2", "LPF2", "I_OPL"]
labels_row2 = ["V_BIP", "Q(V_BIP)", "V_BIP_LPF", "HPF2", "None", "None"]
labels_row3 = ["I_Gang_on", "I_Gang_off", "V_LIF_on", "V_LIF_off", "Fired ON", "Fired OFF"]

def _to_uint8_bgr(x):
    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = (255.0 *x).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

def _annotate(tile_bgr, label):
    h, w = tile_bgr.shape[:2]
    bar_h = max(18, h // 20)  # label bar height
    overlay = tile_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, tile_bgr, 0.45, 0, tile_bgr)
    cv2.putText(tile_bgr, label, (6, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile_bgr

def _prep(x, label):
    return _annotate(_to_uint8_bgr(x), label)

if __name__ == "__main__":
    cfg = SiliconRetinaConfig()

    # Load the model
    T = 300
    fps = 200

    imgs = []

    cur_dir = os.path.join(".")
    root_path = os.path.join(cur_dir, "frames")
    frames = [os.path.join(root_path, f) for f in sorted(os.listdir(root_path)) if f.endswith('.png')]

    for i in range(T):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32)
        imgs.append(img / 255.0)

    imgs = torch.stack(imgs, dim=0).to("cuda:0")

    T, H, W = imgs.shape

    dt = 1.0 / fps
    tau = 0.01
    tau_adaptation = 0.005
    IPL_tau = 0.02
    lambda_G = 400
    g_L = 50
    C = 0.01

    alpha1 = 1.0 - math.exp(-dt / tau)
    alpha1_q = find_float_closest_to_quantization(alpha1)

    alpha2 = 1.0 - math.exp(-dt / tau)
    alpha2_q = find_float_closest_to_quantization(alpha2)

    V_BIP_LPF = 1.0 - math.exp(-dt / tau_adaptation)
    V_BIP_LPF_q = find_float_closest_to_quantization(V_BIP_LPF)

    p1 = dt / C
    p1_q = find_float_closest_to_quantization(alpha1_q)

    p2 = dt * lambda_G
    p2_q = find_float_closest_to_quantization(alpha2_q)

    p3 = dt * g_L
    p3_q = find_float_closest_to_quantization(p3)

    I_bias = 37 / 255.
    I_bias = I_bias

    config = SiliconRetinaConfig(alpha1=alpha1,
                                 alpha2=alpha2,
                                 V_BIP_LPF_alpha=V_BIP_LPF,
                                 lambda_OPL=5.0,
                                 G_A_zero=0.01,
                                 lambda_A=0.75,
                                 p1=p1_q,
                                 I_bias_on=I_bias,
                                 I_bias_off=I_bias,
                                 p2= p2_q,
                                 p3 = p3_q,
                                 U_rest_on=-0.2,
                                 U_rest_off=-0.2,
                                 U_th_on=0.5,
                                 U_th_off=0.5,
                                 U_reset_on=0.3,
                                 U_reset_off=0.3)



    model = SiliconRetinaVerboseFloatCUDA(config, height=imgs.shape[1], width=imgs.shape[2], T=T)

    # Run the model
    fires_on, fires_off = model.forward(imgs)  # expect bool-like masks per frame

    # get all of the internal buffers
    imgs        = imgs.cpu().numpy()
    LPF1        = model.LPF1.cpu().numpy()
    HPF1        = model.HPF1.cpu().numpy()
    BLURRED2    = model.BLURRED2.cpu().numpy()
    LPF2        = model.LPF2.cpu().numpy()
    I_OPL       = model.I_OPL.cpu().numpy()

    V_BIP       = model.V_BIP.cpu().numpy()
    Q_V_BIP     = model.Q_V_BIP.cpu().numpy()
    V_BIP_LPF   = model.V_BIP_LPF.cpu().numpy()
    HPF2        = model.HPF2.cpu().numpy()

    I_Gang_on   = model.I_Gang_on.cpu().numpy()
    I_Gang_off  = model.I_Gang_off.cpu().numpy()

    V_LIF_on    = model.V_LIF_on.cpu().numpy()
    V_LIF_off   = model.V_LIF_off.cpu().numpy()

    fired_on    = model.fired_on.cpu().numpy()
    fired_off   = model.fired_off.cpu().numpy()

    T, H, W = LPF1.shape

    with imageio.get_writer(os.path.join(cur_dir, "silicon_retina_visualization.mp4"), fps=30) as writer:
        for i in range(T):
            zeros_tile = np.zeros_like(V_BIP[i])

            row1_tiles = [
                _prep(imgs[i], labels_row1[0]),
                _prep(LPF1[i], labels_row1[1]),
                _prep(HPF1[i], labels_row1[2]),
                _prep(BLURRED2[i], labels_row1[3]),
                _prep(LPF2[i], labels_row1[4]),
                _prep(I_OPL[i], labels_row1[5]),
            ]
            row2_tiles = [
                _prep(V_BIP[i], labels_row2[0]),
                _prep(Q_V_BIP[i], labels_row2[1]),
                _prep(V_BIP_LPF[i], labels_row2[2]),
                _prep(HPF2[i], labels_row2[3]),
                _prep(zeros_tile, labels_row2[4]),
                _prep(zeros_tile, labels_row2[5]),
            ]
            row3_tiles = [
                _prep(I_Gang_on[i], labels_row3[0]),
                _prep(I_Gang_off[i], labels_row3[1]),
                _prep(V_LIF_on[i], labels_row3[2]),
                _prep(V_LIF_off[i], labels_row3[3]),
                _prep(fired_on[i] / 1.0, labels_row3[4]),
                _prep(fired_off[i] / 1.0, labels_row3[5]),
            ]

            row1 = np.concatenate(row1_tiles, axis=1)
            row2 = np.concatenate(row2_tiles, axis=1)
            row3 = np.concatenate(row3_tiles, axis=1)
            frame_bgr = np.concatenate([row1, row2, row3], axis=0)

            # add timestamp/frame index
            ts = f"t = {i / fps:.3f}s (frame {i})"
            cv2.putText(frame_bgr, ts, (8, frame_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # imageio expects RGB
            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))