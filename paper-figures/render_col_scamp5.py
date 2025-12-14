import os
import re
import cv2
import numpy as np

def make_single_mp4_from_two_displays_with_counter(
    frames_dir: str,
    out_mp4: str,
    out_folder = os.path.join("paper-figures", "scamp5_videos", "vid00"),
    fps: float = 30.0,
    output_mode : str = "both",
    layout: str = "hstack",       
    codec: str = "mp4v",   
    resize_mode: str = "match_00",
    counter_mode: str = "index", 
    draw_on: str = "combined",
):
    pat = re.compile(r"^display_(\d{2})_(0\d+)\.bmp$", re.IGNORECASE)

    frames = {"00": {}, "10": {}}
    for fn in os.listdir(frames_dir):
        m = pat.match(fn)
        if not m:
            continue
        disp, idx = m.group(1), int(m.group(2))
        if disp in frames:
            frames[disp][idx] = fn

    common_idxs = sorted(set(frames["00"].keys()) & set(frames["10"].keys()))
    if not common_idxs:
        raise FileNotFoundError(f"No matching frame indices found for both display_00 and display_01 in {frames_dir}")

    def read_frame(disp: str, idx: int):
        p = os.path.join(frames_dir, frames[disp][idx])
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read: {p}")
        return img

    i0 = common_idxs[0]
    img00 = read_frame("00", i0)
    img01 = read_frame("10", i0)

    h00, w00 = img00.shape[:2]
    h01, w01 = img01.shape[:2]

    def maybe_resize(a, target_hw):
        th, tw = target_hw
        if a.shape[0] == th and a.shape[1] == tw:
            return a
        return cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)

    if resize_mode == "match_00":
        target_hw = (h00, w00)
    elif resize_mode == "match_10":
        target_hw = (h01, w01)
    elif resize_mode == "none":
        if (h00, w00) != (h01, w01):
            raise ValueError(
                f"Frame sizes differ: display_00={w00}x{h00}, display_10={w01}x{h01}. "
                f"Use resize_mode='match_00' or 'match_10'."
            )
        target_hw = (h00, w00)
    else:
        raise ValueError("resize_mode must be one of: match_00, match_10, none")

    th, tw = target_hw

    if output_mode.lower() == "both":
        if layout == "hstack":
            out_size = (tw * 2, th)
            combine = lambda a, b: np.hstack([a, b])
            roi_00 = (0, 0, tw, th)
            roi_01 = (tw, 0, tw, th)
        elif layout == "vstack":
            out_size = (tw, th * 2)
            combine = lambda a, b: np.vstack([a, b])
            roi_00 = (0, 0, tw, th)
            roi_01 = (0, th, tw, th)
        else:
            raise ValueError("layout must be 'hstack' or 'vstack'")
    else:
        out_size = (tw, th)
        combine = lambda a, b: a
        roi_00 = (0, 0, tw, th)
        roi_01 = (0, 0, tw, th)

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, out_size)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_mp4}. Try codec='avc1' or a different path.")

    # Text drawing helpers
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_counter(img, text, x, y):
        # outline for readability
        cv2.putText(img, text, (x, y), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    written = 0
    os.makedirs(out_folder, exist_ok=True)
    
    for seq, idx in enumerate(common_idxs):
        a = maybe_resize(read_frame("00", idx), target_hw)
        b = maybe_resize(read_frame("10", idx), target_hw)

        if output_mode.lower() == "both":
            out = combine(a, b)
        else:
            out = a
        
        cv2.imwrite(os.path.join(out_folder, f"frame_{idx:04d}.png"), out)

        if counter_mode == "index":
            label = f"{idx}"
        elif counter_mode == "sequence":
            label = f"{seq}"
        else:
            raise ValueError("counter_mode must be 'index' or 'sequence'")

        # Choose where to draw
        if draw_on == "combined":
            x, y = 10, 20
            draw_counter(out, label, x, y)
        elif draw_on == "00":
            x0, y0, w0, h0 = roi_00
            draw_counter(out, label, x0 + 20, y0 + 50)
        elif draw_on == "10":
            x1, y1, w1, h1 = roi_01
            draw_counter(out, label, x1 + 20, y1 + 50)
        else:
            raise ValueError("draw_on must be 'combined', '00', or '10'")

        vw.write(out)
        written += 1

    vw.release()
    print(f"Done. Wrote {written} frames to: {out_mp4}")

if __name__ == "__main__":
    print(os.getcwd())
    # pip install opencv-python numpy
    #frames_dir = r"D:/scamp/scamp5d_host/bin/record_20251205_1653_15"
    #frames_dir = r"D:/scamp/scamp5d_host/bin/record_20251205_1655_54"
    #frames_dir = r"D:/scamp/scamp5d_host/bin/record_20251205_1656_47"
    frames_dir = r"D:/scamp/scamp5d_host/bin/record_20251205_1726_56"
    
    
    
    vid_prefix = "vid03"
    
    make_single_mp4_from_two_displays_with_counter(
        frames_dir,
        out_mp4=os.path.join("paper-figures", "scamp5_videos", f"{vid_prefix}.mp4"),
        output_mode="00",
        out_folder = os.path.join("paper-figures", "scamp5_videos", vid_prefix),
        fps=30.0,
        layout="vstack",        # 00 left, 01 right
        codec="mp4v",
        resize_mode="match_00",
        counter_mode="index",   # show filename index
        draw_on="combined"      # or "00"/"01"
    )
