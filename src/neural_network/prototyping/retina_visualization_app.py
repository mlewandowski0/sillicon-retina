import os, math, cv2, numpy as np, torch, imageio
from esim_torch.silicon_retina_torch import SiliconRetinaVerboseFloatCUDA, SiliconRetinaConfig

# ---------- labels & helpers ----------
labels_row1 = ["Input", "LPF1", "HPF1", "BLURRED2", "LPF2", "I_OPL"]
labels_row2 = ["V_BIP", "Q(V_BIP)", "V_BIP_LPF", "HPF2", "None", "None"]
labels_row3 = ["I_Gang_on", "I_Gang_off", "V_LIF_on", "V_LIF_off", "Fired ON", "Fired OFF"]

def _to_uint8_bgr(x):
    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)   # display range
    x = (x + 1.0) / 2.0
    x = (255.0 * x).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

def _annotate(tile_bgr, label):
    h, w = tile_bgr.shape[:2]
    bar_h = max(18, h // 20)
    overlay = tile_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, tile_bgr, 0.45, 0, tile_bgr)
    cv2.putText(tile_bgr, label, (6, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile_bgr

def _prep(x, label):
    return _annotate(_to_uint8_bgr(x), label)

def quantize01(val, q_bits):
    if q_bits <= 0:
        return val
    steps = 2 ** q_bits
    grid = np.linspace(0.0, 1.0, steps)
    return float(grid[np.argmin(np.abs(grid - val))])

def make_config_from_sliders(fps, tau_ms, tauA_ms, lamG, gL, C_val,
                             lamOPL, GAzero, lamA, Ibias01,
                             Urest, Uth, Ureset, q_bits):
    dt = 1.0 / max(1, fps)
    tau = max(1e-6, tau_ms / 1000.0)
    tauA = max(1e-6, tauA_ms / 1000.0)

    alpha1 = 1.0 - math.exp(-dt / tau)
    alpha2 = 1.0 - math.exp(-dt / tau)
    alphaA = 1.0 - math.exp(-dt / tauA)

    # quantize the coefficients (to emulate limited precision if you want)
    alpha1_q = quantize01(alpha1, q_bits)
    alpha2_q = quantize01(alpha2, q_bits)
    alphaA_q = quantize01(alphaA, q_bits)

    # ODE integrator coefficients
    p1 = dt / max(1e-6, C_val)
    p1_q = quantize01(min(1.0, p1), q_bits)

    p2 = dt * lamG
    p3 = dt * gL
    # clamp for safety then (optionally) quantize to 0..1
    p2_q = quantize01(min(1.0, p2), q_bits)
    p3_q = quantize01(min(1.0, p3), q_bits)

    return SiliconRetinaConfig(
        alpha1=alpha1_q,
        alpha2=alpha2_q,
        V_BIP_LPF_alpha=alphaA_q,
        lambda_OPL=lamOPL,
        G_A_zero=GAzero,
        lambda_A=lamA,
        p1=p1_q,
        I_bias_on=Ibias01,
        I_bias_off=Ibias01,
        p2=p2_q,
        p3=p3_q,
        U_rest_on=Urest,
        U_rest_off=Urest,
        U_th_on=Uth,
        U_th_off=Uth,
        U_reset_on=Ureset,
        U_reset_off=Ureset
    )

def run_model_and_collect(imgs_cuda, config):
    T, H, W = imgs_cuda.shape
    model = SiliconRetinaVerboseFloatCUDA(config, height=H, width=W, T=T)
    fires_on, fires_off = model.forward(imgs_cuda)  # GPU

    # pull to CPU numpy
    out = {
        "imgs":        imgs_cuda.detach().cpu().numpy(),
        "LPF1":        model.LPF1.detach().cpu().numpy(),
        "HPF1":        model.HPF1.detach().cpu().numpy(),
        "BLURRED2":    model.BLURRED2.detach().cpu().numpy(),
        "LPF2":        model.LPF2.detach().cpu().numpy(),
        "I_OPL":       model.I_OPL.detach().cpu().numpy(),
        "V_BIP":       model.V_BIP.detach().cpu().numpy(),
        "Q_V_BIP":     model.Q_V_BIP.detach().cpu().numpy(),
        "V_BIP_LPF":   model.V_BIP_LPF.detach().cpu().numpy(),
        "HPF2":        model.HPF2.detach().cpu().numpy(),
        "I_Gang_on":   model.I_Gang_on.detach().cpu().numpy(),
        "I_Gang_off":  model.I_Gang_off.detach().cpu().numpy(),
        "V_LIF_on":    model.V_LIF_on.detach().cpu().numpy(),
        "V_LIF_off":   model.V_LIF_off.detach().cpu().numpy(),
        "fired_on":    model.fired_on.detach().cpu().numpy(),
        "fired_off":   model.fired_off.detach().cpu().numpy(),
    }
    return out

def build_mosaic_frames(out, fps):
    imgs      = out["imgs"]
    LPF1      = out["LPF1"];   HPF1    = out["HPF1"];   BLURRED2  = out["BLURRED2"]
    LPF2      = out["LPF2"];   I_OPL   = out["I_OPL"]
    V_BIP     = out["V_BIP"];  Q_V_BIP = out["Q_V_BIP"]; V_BIP_LPF = out["V_BIP_LPF"]; HPF2 = out["HPF2"]
    I_Gang_on = out["I_Gang_on"]; I_Gang_off = out["I_Gang_off"]
    V_LIF_on  = out["V_LIF_on"];  V_LIF_off  = out["V_LIF_off"]
    fired_on  = out["fired_on"];  fired_off  = out["fired_off"]

    T = LPF1.shape[0]
    frames = []
    for i in range(T):
        zeros_tile = np.zeros_like(V_BIP[i])

        row1_tiles = [
            _prep(imgs[i],   labels_row1[0]),
            _prep(LPF1[i],   labels_row1[1]),
            _prep(HPF1[i],   labels_row1[2]),
            _prep(BLURRED2[i], labels_row1[3]),
            _prep(LPF2[i],   labels_row1[4]),
            _prep(I_OPL[i],  labels_row1[5]),
        ]
        row2_tiles = [
            _prep(V_BIP[i],       labels_row2[0]),
            _prep(Q_V_BIP[i],     labels_row2[1]),
            _prep(V_BIP_LPF[i],   labels_row2[2]),
            _prep(HPF2[i],        labels_row2[3]),
            _prep(zeros_tile,     labels_row2[4]),
            _prep(zeros_tile,     labels_row2[5]),
        ]
        row3_tiles = [
            _prep(I_Gang_on[i],   labels_row3[0]),
            _prep(I_Gang_off[i],  labels_row3[1]),
            _prep(V_LIF_on[i],    labels_row3[2]),
            _prep(V_LIF_off[i],   labels_row3[3]),
            _prep(fired_on[i] / 1.0, labels_row3[4]),
            _prep(fired_off[i] / 1.0, labels_row3[5]),
        ]

        row1 = np.concatenate(row1_tiles, axis=1)
        row2 = np.concatenate(row2_tiles, axis=1)
        row3 = np.concatenate(row3_tiles, axis=1)
        frame_bgr = np.concatenate([row1, row2, row3], axis=0)

        ts = f"t = {i / fps:.3f}s (frame {i})"
        cv2.putText(frame_bgr, ts, (8, frame_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        frames.append(frame_bgr)
    return frames

def play_frames(window_name, frames, play_fps):
    delay = max(1, int(1000 / play_fps))
    for f in frames:
        cv2.imshow(window_name, f)
        if cv2.waitKey(delay) & 0xFF == ord('x'):
            break

def save_mp4(frames, path, fps=30):
    with imageio.get_writer(path, fps=fps) as w:
        for f in frames:
            w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

# ---------- main ----------
if __name__ == "__main__":
    # Load frames
    cur_dir = "."
    root_path = os.path.join(cur_dir, "frames")
    files = [os.path.join(root_path, f) for f in sorted(os.listdir(root_path)) if f.endswith(".png")]
    T = min(300, len(files))   # keep this modest for interactivity
    imgs_list = []
    for i in range(T):
        img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        imgs_list.append(torch.tensor(img, dtype=torch.float32) / 255.0)
    imgs_cuda = torch.stack(imgs_list, dim=0).to("cuda:0")  # [T,H,W]

    # UI window + sliders
    win = "Silicon Retina Interactive"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # Helper to add a slider with initial value
    def add_slider(name, maxv, init):
        cv2.createTrackbar(name, win, 0, maxv, lambda v: None)
        cv2.setTrackbarPos(name, win, init)

    # Sliders (integer ranges mapped to floats later)
    add_slider("fps",            600, 200)   # 1..600
    add_slider("tau_ms",         100, 10)    # 1..100
    add_slider("tauA_ms",        100, 5)     # 1..100
    add_slider("lambda_G",      1000, 400)   # 0..1000
    add_slider("g_L",            500, 50)    # 0..500
    add_slider("C_x1000",        100, 10)    # 1..100 -> 0.001..0.100
    add_slider("lambda_OPL_(-5..5)", 100, 25) # 0..100 -> -5..+5
    add_slider("G_A_zero",       200, 50)    # 0..200
    add_slider("lambda_A_x100",  200, 0)     # 0..2.00
    add_slider("I_bias_(0..255)",255, 37)    # 0..255 -> /255
    add_slider("U_rest_(−1..1)", 200, 80)    # 0..200 -> -1..1 (80 => -0.2)
    add_slider("U_th_(0..1)",    100, 50)    # 0..100 -> 0..1
    add_slider("U_reset_(0..1)", 100, 30)    # 0..100 -> 0..1
    add_slider("quant_bits",       8, 4)     # 0..8

    help_img = np.zeros((480, 960, 3), np.uint8)
    cv2.putText(help_img, "Controls: r=recompute  p=replay  s=save mp4  q=quit",
                (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(help_img, "Use sliders to change params before pressing 'r'.",
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

    last_frames = None
    last_fps = 30

    def read_params():
        fps = max(1, cv2.getTrackbarPos("fps", win))
        tau_ms = max(1, cv2.getTrackbarPos("tau_ms", win))
        tauA_ms = max(1, cv2.getTrackbarPos("tauA_ms", win))
        lamG = float(cv2.getTrackbarPos("lambda_G", win))
        gL = float(cv2.getTrackbarPos("g_L", win))
        C_val = max(0.001, cv2.getTrackbarPos("C_x1000", win) / 1000.0)  # 0.001..0.100
        lamOPL = -5.0 + 10.0 * (cv2.getTrackbarPos("lambda_OPL_(-5..5)", win) / 100.0)
        GAzero = float(cv2.getTrackbarPos("G_A_zero", win))
        lamA = cv2.getTrackbarPos("lambda_A_x100", win) / 100.0
        Ibias01 = cv2.getTrackbarPos("I_bias_(0..255)", win) / 255.0
        Urest = -1.0 + 2.0 * (cv2.getTrackbarPos("U_rest_(−1..1)", win) / 200.0)
        Uth = cv2.getTrackbarPos("U_th_(0..1)", win) / 100.0
        Ureset = cv2.getTrackbarPos("U_reset_(0..1)", win) / 100.0
        q_bits = cv2.getTrackbarPos("quant_bits", win)
        return fps, tau_ms, tauA_ms, lamG, gL, C_val, lamOPL, GAzero, lamA, Ibias01, Urest, Uth, Ureset, q_bits

    def recompute_and_play():
        nonlocal last_frames, last_fps
        fps, tau_ms, tauA_ms, lamG, gL, C_val, lamOPL, GAzero, lamA, Ibias01, Urest, Uth, Ureset, q_bits = read_params()
        cfg = make_config_from_sliders(fps, tau_ms, tauA_ms, lamG, gL, C_val,
                                       lamOPL, GAzero, lamA, Ibias01,
                                       Urest, Uth, Ureset, q_bits)
        out = run_model_and_collect(imgs_cuda, cfg)
        frames = build_mosaic_frames(out, fps=fps)
        last_frames = frames
        last_fps = fps
        play_frames(win, frames, min(30, fps))  # preview at up to 30 fps

    # first screen
    cv2.imshow(win, help_img)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            recompute_and_play()
        elif k == ord('p'):
            if last_frames is not None:
                play_frames(win, last_frames, min(30, last_fps))
        elif k == ord('s'):
            if last_frames is not None:
                out_path = os.path.join(".", "silicon_retina_visualization.mp4")
                save_mp4(last_frames, out_path, fps=30)
                print(f"Saved: {out_path}")
        else:
            # refresh help screen (so you see the window if it got covered)
            cv2.imshow(win, help_img)

    cv2.destroyAllWindows()
