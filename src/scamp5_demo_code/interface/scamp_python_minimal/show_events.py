"""SCAMP-5d silicon retina event viewer (DVS-style, no V_LIF panels).

Reads the two binary spike maps the SCAMP firmware sends out each frame
(ON  = FIRING_AND_IS_PE,  S5  on the chip)
(OFF = FIRING_NEG_AND_IS_PE, S6 on the chip)
and renders them in classic DVS colours:

    ON  events  -> RED
    OFF events  -> BLUE
    both/none   -> BLACK

Channel mapping
---------------
Each `vs_gui_add_display(...)` on the SCAMP side returns a `vs_handle` that
becomes the packet's `channel` field. The default ids below assume the
display widgets are registered in the order written in `output_displays()`:

    v_lif_on  -> channel 0     (AOUT, ignored here)
    v_lif_off -> channel 1     (AOUT, ignored here)
    evn_on    -> channel 2     (DOUT, ON events)
    evn_off   -> channel 3     (DOUT, OFF events)

If your firmware uses different ids, either edit `CHAN_*` below or pass them
on the command line (`--on N --off M`). The script also prints every new
channel id it sees so you can tell what the firmware is actually emitting.

Keyboard
--------
    d   toggle exponential decay of past events (DVS-style trails)
    f   cycle vertical flip (flipud / fliplr / rot180 / none)
    [   decrease decay timescale (faster fade)
    ]   increase decay timescale (slower fade)
    c   clear the accumulator
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
# are 0..3 — on my board the first two displays get channels 102 and 103.
#
# By default we AUTO-DISCOVER: the first two SCAMP5_DOUT channels we see
# are assigned to (ON, OFF) events in arrival order, which matches the
# order `output_displays()` emits them. Pass `--on N --off M` to override.
#
# Set to None to leave open for auto-discovery.

CHAN_ON_EVENTS:  int | None = None
CHAN_OFF_EVENTS: int | None = None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

USB_SERIAL = "0"
POLL_INTERVAL_S = 0.0001

WINDOW_NAME = "SCAMP retina - DVS events"
DISPLAY_SCALE = 5                 # nearest-neighbour upscale (logical pixels are
                                  # 128x128 after 2x2-superpixel downsample)
OVERLAY_HEIGHT = 22

ON_COLOR_BGR  = (0,   0, 255)     # red
OFF_COLOR_BGR = (255, 0,   0)     # blue

# Vertical flip choice (SCAMP delivers the image with the top row at the
# bottom of the buffer; minimal.py uses np.flipud, matching that).
ORIENTATIONS = [
    ("flipud", lambda im: np.flipud(im)),
    ("none",   lambda im: im),
    ("fliplr", lambda im: np.fliplr(im)),
    ("rot180", lambda im: np.rot90(im, 2)),
]

# Decay timescales (seconds-to-half-brightness). The "off" entry disables
# decay entirely and the display shows only the most recent frame.
DECAY_TAU_S = [None, 0.05, 0.10, 0.20, 0.40, 0.80]


# ----------------------------------------------------------------------
# Runtime state
# ----------------------------------------------------------------------

class State:
    def __init__(self):
        self.orient_idx = 0
        self.decay_idx  = 0      # index into DECAY_TAU_S, 0 = no decay
        self.accumulator: np.ndarray | None = None
        self.last_decay_time: float | None = None
        self.seen_channels: dict[int, str] = {}  # channel id -> datatype
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


STATE = State()


# ----------------------------------------------------------------------
# 2x2-superpixel downsample (128x128 effective resolution)
# ----------------------------------------------------------------------
# The retina firmware uses a 2x2 superpixel layout: only the PE cell of
# every 2x2 block carries the actual data; the other three cells of every
# block are either replicas (V_LIF, via fill_A) or zeros (events DREG,
# because FIRING_AND_IS_PE is AND'd with DREG_PE). So we keep only one
# pixel out of every 4 in the raw 256x256 buffer, giving a 128x128 view.
#
# Which pixel of the 2x2 IS the PE depends on the firmware's PE-mask
# pattern. We auto-detect the phase from the first event packet that has
# any pixels set: of the four possible (row%2, col%2) offsets, the one
# with the most non-zero pixels wins.

PE_OFFSET: tuple[int, int] | None = None   # (row_offset, col_offset)


def extract_pe_cells(arr: np.ndarray) -> np.ndarray:
    """Downsample a raw HxW buffer to (H/2)x(W/2) by taking only PE cells."""
    global PE_OFFSET
    if PE_OFFSET is None:
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
# Packet -> ndarray
# ----------------------------------------------------------------------

def packet_to_binary(packet) -> np.ndarray:
    """SCAMP5_DOUT packet -> uint8 (H/2)x(W/2) with values in {0, 1}."""
    w = packet["width"]
    h = packet["height"]
    arr = np.frombuffer(packet["buffer"], dtype=np.uint8).reshape((h, w))
    arr = extract_pe_cells(arr)              # 256x256 -> 128x128
    arr = STATE.orient_fn(arr)
    return np.ascontiguousarray((arr > 0).astype(np.uint8))


# ----------------------------------------------------------------------
# Compose the visualisation
# ----------------------------------------------------------------------

def colorize_events(on_map: np.ndarray | None,
                    off_map: np.ndarray | None) -> np.ndarray:
    """Compose a HxWx3 BGR image from two binary maps."""
    if on_map is not None:
        h, w = on_map.shape
    elif off_map is not None:
        h, w = off_map.shape
    else:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    if on_map is not None:
        img[on_map > 0] = ON_COLOR_BGR
    if off_map is not None:
        # OFF on top of ON when both fire -> blue wins (matches typical DVS
        # debug colouring where the more-recent polarity is shown).
        img[off_map > 0] = OFF_COLOR_BGR
    return img


def update_accumulator(new_frame: np.ndarray) -> np.ndarray:
    """Apply exponential decay if enabled, then OR the new frame on top."""
    tau = DECAY_TAU_S[STATE.decay_idx]
    now = time.perf_counter()

    if STATE.accumulator is None or STATE.accumulator.shape != new_frame.shape:
        STATE.accumulator = np.zeros_like(new_frame, dtype=np.float32)
        STATE.last_decay_time = now

    if tau is not None and STATE.last_decay_time is not None:
        dt = now - STATE.last_decay_time
        # half-life decay:  e^(-dt * ln2 / tau)
        decay = float(np.exp(-dt * 0.693147 / tau))
        STATE.accumulator *= decay

    STATE.last_decay_time = now

    # Take max of decayed accumulator and new frame (so a fresh event always
    # shows at full brightness even if the decayed history is dim).
    STATE.accumulator = np.maximum(STATE.accumulator, new_frame.astype(np.float32))
    return np.clip(STATE.accumulator, 0.0, 1.0)


def render_frame(on_map: np.ndarray | None,
                 off_map: np.ndarray | None) -> None:
    """Build the BGR composite, add the overlay, and imshow it."""
    if STATE.decay_idx == 0:
        # No decay: just colour the latest frame
        body = colorize_events(on_map, off_map)
    else:
        # Decay: keep two separate accumulators so ON and OFF fade
        # independently. We hijack STATE.accumulator: index 0 = ON, 1 = OFF.
        # Simplest path: stack them into a (H, W, 2) float accumulator.
        if on_map is not None:
            h, w = on_map.shape
        else:
            h, w = off_map.shape

        if (STATE.accumulator is None
                or STATE.accumulator.shape != (h, w, 2)):
            STATE.accumulator = np.zeros((h, w, 2), dtype=np.float32)
            STATE.last_decay_time = time.perf_counter()

        tau = DECAY_TAU_S[STATE.decay_idx]
        now = time.perf_counter()
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

        body = np.zeros((h, w, 3), dtype=np.uint8)
        body[..., 2] = (on_acc  * 255).astype(np.uint8)   # red   = ON
        body[..., 0] = (off_acc * 255).astype(np.uint8)   # blue  = OFF

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
        f"ON={STATE.on_event_count}  OFF={STATE.off_event_count}"
    )
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)
    return bar


# ----------------------------------------------------------------------
# Per-frame buffering
# ----------------------------------------------------------------------

def maybe_auto_assign_channel(ch: int, datatype: str) -> None:
    """If channels were left as None for auto-discovery, fill them in
    based on the first two SCAMP5_DOUT packets seen (arrival order = the
    order `output_displays()` emits them = ON first, then OFF)."""
    global CHAN_ON_EVENTS, CHAN_OFF_EVENTS
    if datatype != "SCAMP5_DOUT":
        return
    if CHAN_ON_EVENTS is None and ch != CHAN_OFF_EVENTS:
        CHAN_ON_EVENTS = ch
        print(f"[auto] CHAN_ON_EVENTS  = {ch}")
    elif CHAN_OFF_EVENTS is None and ch != CHAN_ON_EVENTS:
        CHAN_OFF_EVENTS = ch
        print(f"[auto] CHAN_OFF_EVENTS = {ch}")


class FrameAssembler:
    """Collects ON + OFF event maps that share the same loop_counter, then
    renders once both arrive (or once the loop counter advances)."""

    def __init__(self):
        self.frame_lc: int | None = None
        self.on_map: np.ndarray | None = None
        self.off_map: np.ndarray | None = None

    def ingest(self, packet) -> bool:
        """Returns True if we just consumed a complete frame and rendered it."""
        ch = packet["channel"]
        lc = packet["loopcounter"]
        dt = packet.get("datatype", "")

        if dt != "SCAMP5_DOUT":
            return False
        if ch not in (CHAN_ON_EVENTS, CHAN_OFF_EVENTS):
            return False

        binary = packet_to_binary(packet)

        if self.frame_lc is not None and lc != self.frame_lc:
            # New frame started before the previous one finished assembling;
            # flush what we have so we never block forever waiting on a
            # panel the firmware isn't emitting this iteration.
            self._flush()

        self.frame_lc = lc
        if ch == CHAN_ON_EVENTS:
            self.on_map = binary
        else:
            self.off_map = binary

        if self.on_map is not None and self.off_map is not None:
            self._flush()
            return True

        return False

    def _flush(self) -> None:
        if self.frame_lc is None:
            return
        STATE.on_event_count  = int(self.on_map.sum())  if self.on_map  is not None else 0
        STATE.off_event_count = int(self.off_map.sum()) if self.off_map is not None else 0
        STATE.frame_lc = self.frame_lc
        STATE.frame_times.append(time.perf_counter())
        render_frame(self.on_map, self.off_map)
        self.frame_lc = None
        self.on_map = None
        self.off_map = None


ASSEMBLER = FrameAssembler()


def show_placeholder(width: int = 256, height: int = 256) -> None:
    """Pop the window immediately so the user knows the script is alive."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, "waiting for SCAMP packets...", (10, height // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    msg = (
        f"ON ch={CHAN_ON_EVENTS}  OFF ch={CHAN_OFF_EVENTS}"
        if (CHAN_ON_EVENTS is not None or CHAN_OFF_EVENTS is not None)
        else "auto-discovering channels (need 2 SCAMP5_DOUT)"
    )
    cv2.putText(img, msg, (10, height // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, img)


# ----------------------------------------------------------------------
# Packet dispatch
# ----------------------------------------------------------------------

def handle(packet) -> None:
    if packet.get("type") != "data":
        return

    dt = packet.get("datatype", "")
    ch = packet.get("channel", -1)

    # Record every new (channel, datatype) so the user can map them.
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
    """Returns False if the user wants to quit."""
    if key in (ord("q"), 27):
        return False
    if key == ord("f"):
        STATE.orient_idx = (STATE.orient_idx + 1) % len(ORIENTATIONS)
        print(f"[toggle] orient = {STATE.orient_name}")
    elif key == ord("d"):
        # Cycle decay: toggle between off and the first non-None entry.
        if STATE.decay_idx == 0:
            STATE.decay_idx = 2
        else:
            STATE.decay_idx = 0
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
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--on",  type=int, default=None,
                   help="channel id for ON  events DREG (default: auto-discover "
                        "from first SCAMP5_DOUT packet)")
    p.add_argument("--off", type=int, default=None,
                   help="channel id for OFF events DREG (default: auto-discover "
                        "from second SCAMP5_DOUT packet)")
    p.add_argument("--serial", default=USB_SERIAL,
                   help=f"USB serial selector (default {USB_SERIAL!r})")
    p.add_argument("--pe-offset", default=None,
                   help="Force the 2x2-superpixel PE phase as 'R,C' with "
                        "R,C in {0,1}. Default: auto-detect from the first "
                        "event packet with any pixels set.")
    return p.parse_args()


def main() -> int:
    global CHAN_ON_EVENTS, CHAN_OFF_EVENTS, PE_OFFSET
    args = parse_args()
    if args.on is not None:
        CHAN_ON_EVENTS = args.on
    if args.off is not None:
        CHAN_OFF_EVENTS = args.off
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
    if CHAN_ON_EVENTS is None and CHAN_OFF_EVENTS is None:
        print("channels: AUTO (assign first 2 SCAMP5_DOUT in arrival order)")
    else:
        print(f"channels: ON events = {CHAN_ON_EVENTS}, OFF events = {CHAN_OFF_EVENTS}")
    print("hotkeys: d=toggle decay  [ / ] = shorter / longer tau  "
          "f=flip  c=clear  q/ESC=quit")

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
