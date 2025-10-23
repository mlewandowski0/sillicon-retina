import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

def _get_ffprobe():
    try:
        from imageio_ffmpeg import get_ffprobe_exe
        return get_ffprobe_exe()
    except Exception:
        return "ffprobe"  # hope it's on PATH

def ffprobe_metadata(path: Path) -> Tuple[Optional[Dict], Optional[str]]:
    """Return (metadata as dict, error) by running ffprobe -show_streams -show_format."""
    import subprocess, shlex
    ffprobe = _get_ffprobe()
    cmd = f'"{ffprobe}" -v error -print_format json -show_streams -show_format "{path}"'
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        meta = json.loads(out.decode("utf-8", errors="ignore"))
        return meta, None
    except subprocess.CalledProcessError as e:
        return None, f"ffprobe failed: {e.output.decode(errors='ignore')[:200]}"
    except Exception as e:
        return None, f"ffprobe error: {repr(e)}"

def pick_video_file(folder: Path) -> Optional[Path]:
    """Prefer your standard output names; otherwise first .mp4 in the folder."""
    candidates = [
        folder / "video_upsampled_using_optical_flow.mp4",
    ]
    # Also allow the fixed-fps naming style
    for p in folder.glob("video_upsampled_fps_eq_*.mp4"):
        candidates.append(p)
    # Fallback: any mp4
    for p in folder.glob("*.mp4"):
        candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return None

def parse_meta(path: Path, meta: Dict) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[int], Optional[int], Optional[str]]:
    """
    Return (nb_frames, duration_sec, fps, width, height, warn)
    Try to use stream[video] entries; fallback to format.duration.
    """
    warn = None
    streams = meta.get("streams", [])
    fmt = meta.get("format", {})

    vstream = None
    for s in streams:
        if s.get("codec_type") == "video":
            vstream = s
            break
    if vstream is None:
        return None, None, None, None, None, "no video stream"

    width = vstream.get("width")
    height = vstream.get("height")

    # FPS: prefer avg_frame_rate
    fps = None
    afr = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate")
    if afr and afr != "0/0":
        try:
            num, den = afr.split("/")
            num, den = float(num), float(den)
            fps = num / den if den != 0 else None
        except Exception:
            warn = (warn or "") + " bad_fps_str"

    # Frames
    nb_frames = None
    if "nb_frames" in vstream and vstream["nb_frames"]:
        try:
            nb_frames = int(vstream["nb_frames"])
        except Exception:
            warn = (warn or "") + " bad_nb_frames"

    # Duration
    duration = None
    cand = vstream.get("duration") or fmt.get("duration")
    if cand:
        try:
            duration = float(cand)
        except Exception:
            warn = (warn or "") + " bad_duration"

    # If frames missing but we have fps & duration, estimate
    if nb_frames is None and fps and duration:
        nb_frames = int(round(fps * duration))

    # If fps missing but we have frames & duration, estimate
    if fps is None and nb_frames and duration and duration > 0:
        fps = nb_frames / duration

    return nb_frames, duration, fps, width, height, warn

def fallback_imageio_count(path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Count frames with imageio (slower, but robust if nb_frames is missing)."""
    try:
        import imageio.v3 as iio
        reader = iio.imopen(path, "r")
        n = reader.get_length()
        try:
            reader.close()
        except Exception:
            pass
        if n is None or n < 0:
            return None, "imageio_length_unknown"
        return int(n), None
    except Exception as e:
        return None, f"imageio_fallback_failed: {repr(e)}"

def human_mb(nbytes: int) -> float:
    return round(nbytes / (1024.0 * 1024.0), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing video_* folders")
    ap.add_argument("--pattern", default="*", help="Glob for sample folders (default: video_*)")
    ap.add_argument("--csv", required=True, help="Output CSV path")
    ap.add_argument("--fallback_imageio", action="store_true", help="Use imageio to count frames if ffprobe nb_frames missing")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    folders = sorted(root.glob(args.pattern))

    print(f"[scan] root={root} pattern={args.pattern} found={len(folders)}")

    import csv
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        idx = folder.name.split("_")[-1]
        video = pick_video_file(folder)
        if video is None:
            rows.append({
                "index": idx,
                "folder": str(folder.relative_to(root)),
                "video": "",
                "size_mb": "",
                "width": "", "height": "",
                "fps": "", "duration_sec": "",
                "frames": "",
                "status": "missing_video"
            })
            continue

        size_mb = human_mb(video.stat().st_size)
        meta, err = ffprobe_metadata(video)
        nb_frames = duration = fps = width = height = None
        status = "ok"
        warn = None

        if err:
            status = "ffprobe_error"
            warn = err[:200]
        else:
            nb_frames, duration, fps, width, height, warn = parse_meta(video, meta)
            if nb_frames is None and args.fallback_imageio:
                nb_frames, e2 = fallback_imageio_count(video)
                if e2:
                    warn = ((warn + " ") if warn else "") + e2
            if duration is None and fps and nb_frames:
                duration = nb_frames / fps

        rows.append({
            "index": idx,
            "folder": str(folder.relative_to(root)),
            "video": str(video.relative_to(root)),
            "size_mb": size_mb,
            "width": width or "",
            "height": height or "",
            "fps": round(fps, 5) if isinstance(fps, float) else "",
            "duration_sec": round(duration, 3) if isinstance(duration, float) else "",
            "frames": nb_frames if isinstance(nb_frames, int) else "",
            "status": status if status != "ok" else (warn or "ok")
        })

    # Sort by duration (desc) then frames (desc)
    def _dur_key(r):
        d = r.get("duration_sec")
        try:
            return float(d)
        except Exception:
            return -1.0
    rows.sort(key=_dur_key, reverse=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index","folder","video","size_mb",
            "width","height","fps","duration_sec","frames","status"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Quick console summary
    n_ok = sum(1 for r in rows if (r["status"] == "ok"))
    n_bad = len(rows) - n_ok
    print(f"[done] wrote {len(rows)} rows to {out_path} | ok={n_ok} bad={n_bad}")

if __name__ == "__main__":
    main()