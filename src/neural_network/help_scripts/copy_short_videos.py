#!/usr/bin/env python3
"""
Copy samples whose video has fewer than N frames.

Examples:
  # Copy whole folders where the video has < 900 frames
  python tools/copy_short_videos.py \
      --root src/neural_network/data/saliency_dataset_upsampled \
      --dest out/short_videos \
      --min-frames 900

  # Copy only the video files (not whole folders), keeping the folder structure
  python tools/copy_short_videos.py \
      --root src/neural_network/data/saliency_dataset_upsampled \
      --dest out/short_video_files \
      --min-frames 900 \
      --what video
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict

def _get_ffprobe():
    try:
        from imageio_ffmpeg import get_ffprobe_exe
        return get_ffprobe_exe()
    except Exception:
        return "ffprobe"  # assume on PATH

def ffprobe_metadata(path: Path) -> Tuple[Optional[Dict], Optional[str]]:
    import subprocess, shlex
    ffprobe = _get_ffprobe()
    cmd = f'"{ffprobe}" -v error -print_format json -show_streams -show_format "{path}"'
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        return json.loads(out.decode("utf-8", errors="ignore")), None
    except subprocess.CalledProcessError as e:
        return None, e.output.decode(errors="ignore")
    except Exception as e:
        return None, repr(e)

def parse_nb_frames(meta: Dict) -> Optional[int]:
    streams = meta.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        return None
    nb = v.get("nb_frames")
    if nb is not None:
        try:
            return int(nb)
        except Exception:
            return None
    # fallback: estimate from duration * fps
    afr = v.get("avg_frame_rate") or v.get("r_frame_rate")
    dur = v.get("duration") or meta.get("format", {}).get("duration")
    try:
        if afr and afr != "0/0" and dur:
            num, den = afr.split("/")
            fps = float(num) / float(den) if float(den) != 0 else None
            if fps and float(dur) > 0:
                return int(round(fps * float(dur)))
    except Exception:
        pass
    return None

def pick_video_file(folder: Path) -> Optional[Path]:
    # prefer your known names, then any mp4
    preferred = [
        folder / "video_upsampled_using_optical_flow.mp4",
    ]
    preferred += list(folder.glob("video_upsampled_fps_eq_*.mp4"))
    for p in preferred:
        if p.exists():
            return p
    for p in folder.glob("*.mp4"):
        return p
    return None

def count_frames(video_path: Path, fallback_imageio: bool) -> Optional[int]:
    meta, err = ffprobe_metadata(video_path)
    nb = parse_nb_frames(meta) if meta else None
    if nb is not None or not fallback_imageio:
        return nb
    # fallback: imageio
    try:
        import imageio.v3 as iio
        r = iio.imopen(video_path, "r")
        n = r.get_length()
        try:
            r.close()
        except Exception:
            pass
        return int(n) if n and n > 0 else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with sample folders (e.g., video_00000, ...)")
    ap.add_argument("--dest", required=True, help="Destination directory to copy into")
    ap.add_argument("--pattern", default="*", help="Folder glob under root (default: video_*)")
    ap.add_argument("--max-frames", type=int, default=900, help="Copy if frames < this value (default: 900)")
    ap.add_argument("--what", choices=["folder", "video"], default="folder",
                    help="Copy whole folder or only the video file (default: folder)")
    ap.add_argument("--fallback-imageio", action="store_true",
                    help="If ffprobe lacks nb_frames, try imageio (slower)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be copied, don’t actually copy")
    args = ap.parse_args()

    root = Path(args.root)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    folders = sorted(root.glob(args.pattern))
    copied, skipped, missing, errors = 0, 0, 0, 0

    print(f"[scan] root={root} pattern={args.pattern} folders={len(folders)} threshold={args.max_frames} frames")

    for folder in folders:
        video = pick_video_file(folder)
        if video is None:
            missing += 1
            print(f"[miss] no video in {folder}")
            continue

        nb = count_frames(video, args.fallback_imageio)
        if nb is None:
            errors += 1
            print(f"[err ] unable to determine frames for {video}")
            continue

        if nb < args.max_frames:
            rel = folder.relative_to(root)
            if args.what == "folder":
                out = dest / rel
                if args.dry_run:
                    print(f"[copy] {folder}  (frames={nb})  ->  {out}")
                else:
                    # copytree with dirs_exist_ok for py3.8+
                    shutil.copytree(folder, out, dirs_exist_ok=True)
                    copied += 1
            else:  # video
                # mirror folder structure, copy only the file
                out_dir = dest / rel
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / video.name
                if args.dry_run:
                    print(f"[copy] {video}  (frames={nb})  ->  {out_file}")
                else:
                    shutil.copy2(video, out_file)
                    copied += 1
        else:
            skipped += 1

    print(f"\n[done] copied={copied} skipped={skipped} missing={missing} errors={errors}")
    print(f"[dest] {dest.resolve()}")

if __name__ == "__main__":
    main()
