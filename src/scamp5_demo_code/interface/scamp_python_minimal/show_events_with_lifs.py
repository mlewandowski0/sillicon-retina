"""SCAMP-5d silicon retina viewer with V_LIF membranes + DVS events.

Same DVS-style colouring as show_events.py (ON = red, OFF = blue), but the
window also shows the two LIF membrane voltages as grayscale panels above
the events panel:

    +---------------------+---------------------+
    |     ON  V_LIF       |     OFF V_LIF       |   <- AOUT  (SCAMP5_AOUT)
    +---------------------+---------------------+
    |              DVS events (red + blue)      |   <- DOUT  (SCAMP5_DOUT)
    +-------------------------------------------+

If you only want events without the membranes, run show_events.py instead.

Channel mapping
---------------
Each `vs_gui_add_display(...)` on the SCAMP side returns a `vs_handle` that
becomes the packet's `channel` field. The default ids below assume the
display widgets are registered in the order written in setup_gui_controls()
of scamp5_main.cpp:

    v_lif_on  -> channel 0     (AOUT, ON  V_LIF)
    v_lif_off -> channel 1     (AOUT, OFF V_LIF)
    evn_on    -> channel 2     (DOUT, ON  events)
    evn_off   -> channel 3     (DOUT, OFF events)

If your firmware uses different ids, edit `CHAN_*` below or override on the
command line (`--on-evt N`, `--off-evt M`, `--on-lif K`, `--off-lif L`).
The script prints every channel id it sees on startup so you can confirm.

Keyboard
--------
    d   toggle exponential decay of past events (DVS-style trails)
    [   decrease decay timescale (faster fade)
    ]   increase decay timescale (slower fade)
    c   clear the accumulator
    f   cycle vertical flip (flipud / fliplr / rot180 / none)
    m   cycle V_LIF colour map (gray / jet / viridis / inferno)
    q / ESC   quit

Requires:
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

import scamp


# ----------------------------------------------------------------------
# Channel mapping
# ----------------------------------------------------------------------
# `vs_gui_add_display(...)` on the SCAMP side returns a `vs_handle` that
# becomes the packet's `channel` field. The SDK does NOT guarantee these
# are 0..3 — on some boards the first widget gets channel 100+ instead.
#
# By default we AUTO-DISCOVER: the first two SCAMP5_DOUT channels we see
# are assigned to (ON, OFF) events, and the first two SCAMP5_AOUT channels
# are assigned to (ON, OFF) V_LIFs, both in arrival order. Pass --on-evt /
# --off-evt / --on-lif / --off-lif to override.
#
# Set to None to leave open for auto-discovery.

CHAN_ON_LIF:     int | None = None
CHAN_OFF_LIF:    int | None = None
CHAN_ON_EVENTS:  int | None = None
CHAN_OFF_EVENTS: int | None = None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

USB_SERIAL = "0"
POLL_INTERVAL_S = 0.0001

WINDOW_NAME = "SCAMP retina - V_LIF + DVS events"
DISPLAY_SCALE = 3                 # nearest-neighbour upscale (logical pixels
                                  # are 128x128 after 2x2-superpixel downsample)
OVERLAY_HEIGHT = 22
PANEL_LABEL_HEIGHT = 16
PANEL_GAP_PX = 2                   # black gap between panels

ON_COLOR_BGR  = (0,   0, 255)     # red
OFF_COLOR_BGR = (255, 0,   0)     # blue

ORIENTATIONS = [
    ("flipud", lambda im: np.flipud(im)),
    ("none",   lambda im: im),
    ("fliplr", lambda im: np.fliplr(im)),
    ("rot180", lambda im: np.rot90(im, 2)),
]

DECAY_TAU_S = [None, 0.05, 0.10, 0.20, 0.40, 0.80]

LIF_COLORMAPS = [
    ("gray",     None),
    ("jet",      cv2.COLORMAP_JET),
    ("viridis",  cv2.COLORMAP_VIRIDIS),
    ("inferno",  cv2.COLORMAP_INFERNO),
]


# ----------------------------------------------------------------------
# Runtime state
# ----------------------------------------------------------------------

class State:
    def __init__(self):
        self.orient_idx = 0
        self.decay_idx  = 0
        self.cmap_idx   = 0
        self.accumulator: np.ndarray | None = None   # (H, W, 2) float32, ch 0 = ON, 1 = OFF
        self.last_decay_time: float | None = None
        self.seen_channels: dict[int, str] = {}
        self.frame_times: deque[float] = deque(maxlen=120)
        self.frame_lc: int = -1
        self.on_event_count = 0
        self.off_event_count = 0

    @property
    def orient_name(self) -> str:
        return ORIENTATIONS[self.orient_idx][0]

    @property
    def orient_fn(self):
        return ORIENTATIONS[self.orient_idx][1]

    @property
    def decay_label(self) -> str:
        tau = DECAY_TAU_S[self.decay_idx]
        return "off" if tau is None else f"{tau:.2f}s"

    @property
    def cmap_name(self) -> str:
        return LIF_COLORMAPS[self.cmap_idx][0]

    @property
    def cmap_id(self):
        return LIF_COLORMAPS[self.cmap_idx][1]


STATE = State()


# ----------------------------------------------------------------------
# 2x2-superpixel downsample (128x128 effective resolution)
# ----------------------------------------------------------------------
# The retina firmware uses a 2x2 superpixel layout: only the PE cell of
# every 2x2 block carries the data; the other three cells are either
# replicas (V_LIF, via fill_A) or zeros (events DREG, because the spike
# map is AND'd with DREG_PE). So we keep only one pixel out of every 4 in
# the raw 256x256 buffer, giving a 128x128 logical image.
#
# Auto-detect the PE phase from event packets: in the four (row%2, col%2)
# offsets, the one with the most non-zero pixels is the PE phase. Once
# detected the same offset is used for the V_LIF panels (since fill_A
# replicates the same value across all 4 cells, any phase works for them).

PE_OFFSET: tuple[int, int] | None = None    # (row_offset, col_offset)


def extract_pe_cells(arr: np.ndarray, is_event: bool) -> np.ndarray:
    """Downsample a raw HxW buffer to (H/2)x(W/2) by taking only PE cells.
    Auto-detects the PE phase from event (DOUT) packets; V_LIF (AOUT)
    packets use whatever phase was detected (any phase works for them)."""
    global PE_OFFSET
    if PE_OFFSET is None and is_event:
        counts = [
            (r, c, int((arr[r::2, c::2] > 0).sum()))
            for r in (0, 1) for c in (0, 1)
        ]
        best = max(counts, key=lambda t: t[2])
        if best[2] > 0:
            PE_OFFSET = (best[0], best[1])
            print(f"[auto] PE_OFFSET = {PE_OFFSET}  "
                  f"({best[2]} non-zero pixels in that 2x2 phase)")
    r, c = PE_OFFSET if PE_OFFSET is not None else (0, 0)
    return arr[r::2, c::2]


# ----------------------------------------------------------------------
# Packet decoders
# ----------------------------------------------------------------------

def packet_to_binary(packet) -> np.ndarray:
    """SCAMP5_DOUT (event) packet -> uint8 (H/2)x(W/2) in {0, 1}."""
    w = packet["width"]
    h = packet["height"]
    arr = np.frombuffer(packet["buffer"], dtype=np.uint8).reshape((h, w))
    arr = extract_pe_cells(arr, is_event=True)   # 256x256 -> 128x128
    arr = STATE.orient_fn(arr)
    return np.ascontiguousarray((arr > 0).astype(np.uint8))


def packet_to_analog(packet) -> np.ndarray:
    """SCAMP5_AOUT (V_LIF) packet -> uint8 (H/2)x(W/2) (raw bytes, host
    already quantized to 0..255 inside the wrapper)."""
    w = packet["width"]
    h = packet["height"]
    arr = np.frombuffer(packet["buffer"], dtype=np.uint8).reshape((h, w))
    arr = extract_pe_cells(arr, is_event=False)  # 256x256 -> 128x128
    arr = STATE.orient_fn(arr)
    return np.ascontiguousarray(arr)


# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------

def colorize_lif(img_u8: np.ndarray) -> np.ndarray:
    """Apply the current LIF colour map to a uint8 HxW image."""
    cmap = STATE.cmap_id
    if cmap is None:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(img_u8, cmap)


def decayed_event_image(on_map: np.ndarray | None,
                        off_map: np.ndarray | None,
                        shape_hint: tuple[int, int] | None) -> np.ndarray:
    """Return an HxWx3 BGR event image, possibly using exponential decay."""
    # Pick a shape we can work with
    if on_map is not None:
        h, w = on_map.shape
    elif off_map is not None:
        h, w = off_map.shape
    elif shape_hint is not None:
        h, w = shape_hint
    else:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    if STATE.decay_idx == 0:
        # No decay: just colour the latest frame
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if on_map is not None:
            img[on_map > 0] = ON_COLOR_BGR
        if off_map is not None:
            img[off_map > 0] = OFF_COLOR_BGR
        return img

    # Decay path: separate accumulators for ON and OFF
    if (STATE.accumulator is None
            or STATE.accumulator.shape != (h, w, 2)):
        STATE.accumulator = np.zeros((h, w, 2), dtype=np.float32)
        STATE.last_decay_time = time.perf_counter()

    now = time.perf_counter()
    tau = DECAY_TAU_S[STATE.decay_idx]
    if STATE.last_decay_time is not None and tau is not None:
        dt = now - STATE.last_decay_time
        decay = float(np.exp(-dt * 0.693147 / tau))
        STATE.accumulator *= decay
    STATE.last_decay_time = now

    if on_map is not None:
        STATE.accumulator[..., 0] = np.maximum(
            STATE.accumulator[..., 0], on_map.astype(np.float32)
        )
    if off_map is not None:
        STATE.accumulator[..., 1] = np.maximum(
            STATE.accumulator[..., 1], off_map.astype(np.float32)
        )

    on_acc  = np.clip(STATE.accumulator[..., 0], 0.0, 1.0)
    off_acc = np.clip(STATE.accumulator[..., 1], 0.0, 1.0)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 2] = (on_acc  * 255).astype(np.uint8)   # red   = ON
    img[..., 0] = (off_acc * 255).astype(np.uint8)   # blue  = OFF
    return img


def label_panel(img: np.ndarray, text: str,
                fg=(220, 220, 220)) -> np.ndarray:
    """Paste a small label bar on top of the panel image (in place copy)."""
    h, w = img.shape[:2]
    out = np.zeros((h + PANEL_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    out[PANEL_LABEL_HEIGHT:, :, :] = img
    cv2.putText(out, text, (4, PANEL_LABEL_HEIGHT - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)
    return out


def render_frame(on_evt: np.ndarray | None,
                 off_evt: np.ndarray | None,
                 on_lif: np.ndarray | None,
                 off_lif: np.ndarray | None) -> None:

    # Decide the panel size from whichever map we have first.
    sample = on_evt if on_evt is not None else off_evt
    if sample is None:
        sample = on_lif if on_lif is not None else off_lif
    if sample is None:
        return
    h, w = sample.shape[:2]

    # ---- Top row: ON V_LIF | OFF V_LIF -------------------------------
    blank = np.zeros((h, w), dtype=np.uint8)
    on_panel  = label_panel(colorize_lif(on_lif  if on_lif  is not None else blank),
                            "ON  V_LIF")
    off_panel = label_panel(colorize_lif(off_lif if off_lif is not None else blank),
                            "OFF V_LIF")
    gap_col = np.zeros((on_panel.shape[0], PANEL_GAP_PX, 3), dtype=np.uint8)
    top_row = np.hstack([on_panel, gap_col, off_panel])

    # ---- Bottom: DVS events spanning full width ----------------------
    events = decayed_event_image(on_evt, off_evt, (h, w))
    # Repeat the events image horizontally so it occupies the same total
    # width as the two top panels (so the columns line up visually).
    events_wide = np.hstack([events, gap_col, events])  # same width as top_row
    events_panel = label_panel(events_wide, "DVS events  (ON=red  OFF=blue)")

    gap_row = np.zeros((PANEL_GAP_PX, top_row.shape[1], 3), dtype=np.uint8)
    body = np.vstack([top_row, gap_row, events_panel])

    overlay = make_overlay(body.shape[1])
    composite = np.vstack([overlay, body])

    if DISPLAY_SCALE != 1:
        composite = cv2.resize(
            composite,
            (composite.shape[1] * DISPLAY_SCALE,
             composite.shape[0] * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imshow(WINDOW_NAME, composite)


def make_overlay(width: int) -> np.ndarray:
    bar = np.zeros((OVERLAY_HEIGHT, width, 3), dtype=np.uint8)
    if len(STATE.frame_times) >= 2:
        fps = (len(STATE.frame_times) - 1) / (STATE.frame_times[-1] - STATE.frame_times[0])
    else:
        fps = 0.0
    pe_str = (f"PE={PE_OFFSET[0]},{PE_OFFSET[1]}"
              if PE_OFFSET is not None else "PE=?")
    text = (
        f"fps={fps:5.1f}  "
        f"lc={STATE.frame_lc}  "
        f"{pe_str}  "
        f"orient={STATE.orient_name}  "
        f"decay={STATE.decay_label}  "
        f"cmap={STATE.cmap_name}  "
        f"ON={STATE.on_event_count}  OFF={STATE.off_event_count}"
    )
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)
    return bar


# ----------------------------------------------------------------------
# Per-frame buffering: wait until all 4 panels (or a new lc) before render
# ----------------------------------------------------------------------

def maybe_auto_assign_channel(ch: int, datatype: str) -> None:
    """Auto-discover channels: first two SCAMP5_DOUT become (ON, OFF) event
    channels, first two SCAMP5_AOUT become (ON, OFF) V_LIF channels. Order
    matches what `output_displays()` emits (ON before OFF within each type)."""
    global CHAN_ON_EVENTS, CHAN_OFF_EVENTS, CHAN_ON_LIF, CHAN_OFF_LIF
    if datatype == "SCAMP5_DOUT":
        if CHAN_ON_EVENTS is None and ch != CHAN_OFF_EVENTS:
            CHAN_ON_EVENTS = ch
            print(f"[auto] CHAN_ON_EVENTS  = {ch}")
        elif CHAN_OFF_EVENTS is None and ch != CHAN_ON_EVENTS:
            CHAN_OFF_EVENTS = ch
            print(f"[auto] CHAN_OFF_EVENTS = {ch}")
    elif datatype == "SCAMP5_AOUT":
        if CHAN_ON_LIF is None and ch != CHAN_OFF_LIF:
            CHAN_ON_LIF = ch
            print(f"[auto] CHAN_ON_LIF     = {ch}")
        elif CHAN_OFF_LIF is None and ch != CHAN_ON_LIF:
            CHAN_OFF_LIF = ch
            print(f"[auto] CHAN_OFF_LIF    = {ch}")


class FrameAssembler:
    def __init__(self):
        self.lc: int | None = None
        self.on_evt:  np.ndarray | None = None
        self.off_evt: np.ndarray | None = None
        self.on_lif:  np.ndarray | None = None
        self.off_lif: np.ndarray | None = None

    def ingest(self, packet) -> None:
        ch = packet["channel"]
        lc = packet["loopcounter"]
        dt = packet.get("datatype", "")

        if ch == CHAN_ON_EVENTS  and dt == "SCAMP5_DOUT":
            decoded, slot = packet_to_binary(packet), "on_evt"
        elif ch == CHAN_OFF_EVENTS and dt == "SCAMP5_DOUT":
            decoded, slot = packet_to_binary(packet), "off_evt"
        elif ch == CHAN_ON_LIF  and dt == "SCAMP5_AOUT":
            decoded, slot = packet_to_analog(packet), "on_lif"
        elif ch == CHAN_OFF_LIF and dt == "SCAMP5_AOUT":
            decoded, slot = packet_to_analog(packet), "off_lif"
        else:
            return

        if self.lc is not None and lc != self.lc:
            # Loop counter advanced before we collected everything from the
            # previous frame. Render with whatever we have — if the user
            # toggled a panel off on the firmware side it'll just stay
            # blank, but we don't block waiting for it forever.
            self._flush()

        self.lc = lc
        setattr(self, slot, decoded)

    def _flush(self) -> None:
        if self.lc is None:
            return
        STATE.on_event_count  = int(self.on_evt.sum())  if self.on_evt  is not None else 0
        STATE.off_event_count = int(self.off_evt.sum()) if self.off_evt is not None else 0
        STATE.frame_lc = self.lc
        STATE.frame_times.append(time.perf_counter())
        render_frame(self.on_evt, self.off_evt, self.on_lif, self.off_lif)
        # Reset for the next frame
        self.lc = None
        self.on_evt = None
        self.off_evt = None
        self.on_lif = None
        self.off_lif = None


ASSEMBLER = FrameAssembler()


def show_placeholder(width: int = 384, height: int = 256) -> None:
    """Pop the window immediately so the user sees the script is alive."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, "waiting for SCAMP packets...", (10, height // 2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    parts = []
    parts.append(f"ON evt={CHAN_ON_EVENTS}"  if CHAN_ON_EVENTS  is not None else "ON evt=auto")
    parts.append(f"OFF evt={CHAN_OFF_EVENTS}" if CHAN_OFF_EVENTS is not None else "OFF evt=auto")
    parts.append(f"ON lif={CHAN_ON_LIF}"     if CHAN_ON_LIF     is not None else "ON lif=auto")
    parts.append(f"OFF lif={CHAN_OFF_LIF}"   if CHAN_OFF_LIF    is not None else "OFF lif=auto")
    cv2.putText(img, "  ".join(parts), (10, height // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "(if no V_LIF panels: enable display_v_lif_on/off on firmware)",
                (10, height // 2 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, img)


# ----------------------------------------------------------------------
# Packet dispatch
# ----------------------------------------------------------------------

def handle(packet) -> None:
    if packet.get("type") != "data":
        return

    dt = packet.get("datatype", "")
    ch = packet.get("channel", -1)

    if ch not in STATE.seen_channels:
        STATE.seen_channels[ch] = dt
        print(f"[discover] first packet on channel {ch}  datatype={dt}")
        maybe_auto_assign_channel(ch, dt)

    if dt == "TEXT":
        text = packet.get("text", "")
        if text:
            print(f"[lc={packet.get('loopcounter', -1)}] {text!r}")
        return

    ASSEMBLER.ingest(packet)


def drain_packets(max_packets: int = 1000) -> int:
    n = 0
    while n < max_packets:
        packet = scamp.get_packet()
        if packet is None:
            break
        handle(packet)
        n += 1
    return n


# ----------------------------------------------------------------------
# Keyboard
# ----------------------------------------------------------------------

def handle_key(key: int) -> bool:
    if key in (ord("q"), 27):
        return False
    if key == ord("f"):
        STATE.orient_idx = (STATE.orient_idx + 1) % len(ORIENTATIONS)
        print(f"[toggle] orient = {STATE.orient_name}")
    elif key == ord("d"):
        STATE.decay_idx = 2 if STATE.decay_idx == 0 else 0
        print(f"[toggle] decay = {STATE.decay_label}")
    elif key == ord("["):
        STATE.decay_idx = max(1, STATE.decay_idx - 1)
        print(f"[toggle] decay = {STATE.decay_label}")
    elif key == ord("]"):
        STATE.decay_idx = min(len(DECAY_TAU_S) - 1, STATE.decay_idx + 1)
        print(f"[toggle] decay = {STATE.decay_label}")
    elif key == ord("c"):
        STATE.accumulator = None
        print("[action] cleared event accumulator")
    elif key == ord("m"):
        STATE.cmap_idx = (STATE.cmap_idx + 1) % len(LIF_COLORMAPS)
        print(f"[toggle] cmap = {STATE.cmap_name}")
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--on-evt",  type=int, default=None,
                   help="channel id for ON  events DREG  (default: auto-discover "
                        "from first SCAMP5_DOUT packet)")
    p.add_argument("--off-evt", type=int, default=None,
                   help="channel id for OFF events DREG  (default: auto-discover "
                        "from second SCAMP5_DOUT packet)")
    p.add_argument("--on-lif",  type=int, default=None,
                   help="channel id for ON  V_LIF AREG   (default: auto-discover "
                        "from first SCAMP5_AOUT packet)")
    p.add_argument("--off-lif", type=int, default=None,
                   help="channel id for OFF V_LIF AREG   (default: auto-discover "
                        "from second SCAMP5_AOUT packet)")
    p.add_argument("--serial",  default=USB_SERIAL,
                   help=f"USB serial selector (default {USB_SERIAL!r})")
    p.add_argument("--pe-offset", default=None,
                   help="Force the 2x2-superpixel PE phase as 'R,C' with "
                        "R,C in {0,1}. Default: auto-detect from the first "
                        "event packet with any pixels set.")
    return p.parse_args()


def main() -> int:
    global CHAN_ON_EVENTS, CHAN_OFF_EVENTS, CHAN_ON_LIF, CHAN_OFF_LIF, PE_OFFSET
    args = parse_args()
    if args.on_evt  is not None: CHAN_ON_EVENTS  = args.on_evt
    if args.off_evt is not None: CHAN_OFF_EVENTS = args.off_evt
    if args.on_lif  is not None: CHAN_ON_LIF     = args.on_lif
    if args.off_lif is not None: CHAN_OFF_LIF    = args.off_lif
    if args.pe_offset is not None:
        r, c = (int(x) for x in args.pe_offset.split(","))
        PE_OFFSET = (r, c)
        print(f"[cli ] PE_OFFSET = {PE_OFFSET}  (forced via --pe-offset)")

    print(f"opening USB connection to SCAMP-5d serial={args.serial!r}...")
    rc = scamp.open_usb(args.serial)
    if rc != 0:
        print(f"open_usb failed rc={rc}. Is the board plugged in?", file=sys.stderr)
        return 1
    print("connected.")
    print(f"channels:")
    print(f"   ON  events = {CHAN_ON_EVENTS  if CHAN_ON_EVENTS  is not None else 'auto'}   "
          f"OFF events = {CHAN_OFF_EVENTS if CHAN_OFF_EVENTS is not None else 'auto'}")
    print(f"   ON  V_LIF  = {CHAN_ON_LIF     if CHAN_ON_LIF     is not None else 'auto'}   "
          f"OFF V_LIF  = {CHAN_OFF_LIF    if CHAN_OFF_LIF    is not None else 'auto'}")
    print("hotkeys: d=decay  [ / ] = shorter / longer tau  "
          "f=flip  m=cmap  c=clear  q/ESC=quit")

    scamp.send_message(scamp.VS_MSG_HOST_ON, 0, 0)

    # Pop the window immediately so user knows the script is alive even
    # before the first complete frame is assembled. cv2.waitKey() also needs
    # an active window to actually process keys.
    show_placeholder()

    try:
        while True:
            scamp.routine()
            n = drain_packets()

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not handle_key(key):
                break

            if n == 0:
                time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        scamp.send_message(scamp.VS_MSG_HOST_DC, 0, 0)
        time.sleep(0.05)
        scamp.close()
        cv2.destroyAllWindows()
        print("closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
