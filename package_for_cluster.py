#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DEFAULT_EXCLUDED_EXTS = {
    ".zip", ".npy", ".npz", ".mp4", ".avi",
    # add more here if you like:
    ".mkv", ".mov", ".jpg", ".jpeg", ".png", ".gif",
    ".bmp", ".tif", ".tiff", ".wav", ".mp3", ".flac",
    ".pth", ".pt", ".ckpt"
}

DEFAULT_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    ".venv", ".vs",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ipynb_checkpoints", "build", "dist", "venv", ".venv", "data", "logs",
    "bin", "models"
}

def should_skip_file(p: Path, excluded_exts: set[str]) -> bool:
    if p.is_symlink():
        return True
    return p.suffix.lower() in excluded_exts

def should_skip_dir(dirname: str, excluded_dirs: set[str]) -> bool:
    return dirname in excluded_dirs or dirname.startswith("._")

def stage_project(root: Path, stage_dir: Path,
                  excluded_exts: set[str], excluded_dirs: set[str]) -> list[Path]:
    """Copy selected files from root into stage_dir, preserving relative paths."""
    root = root.resolve()
    staged = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d, excluded_dirs)]
        for fname in filenames:
            src = Path(dirpath) / fname
            if should_skip_file(src, excluded_exts):
                continue
            rel = src.relative_to(root)
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copy file metadata too
            shutil.copy2(src, dst)
            staged.append(rel)
    return staged

def zip_from_dir(src_dir: Path, out_zip: Path) -> int:
    """Zip everything under src_dir (no extra exclusions; it's already staged)."""
    count = 0
    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(src_dir):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                arcname = fpath.relative_to(src_dir)
                zf.write(fpath, arcname)
                count += 1
    return count

def parse_ext_list(exts: list[str]) -> set[str]:
    norm = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        norm.add(e)
    return norm

def ensure_clean_dir(path: Path, force: bool):
    if path.exists():
        # if exists and not empty
        if any(path.iterdir()):
            if not force:
                raise SystemExit(
                    f"Stage dir '{path}' exists and is not empty. "
                    f"Use --force to clear it."
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(
        description="Stage then zip a project while excluding large/binary artifacts."
    )
    ap.add_argument("root", type=Path, help="Path to the project root directory")
    ap.add_argument("-o", "--output", type=Path, default=Path("project_bundle.zip"),
                    help="Output zip file path (default: project_bundle.zip)")
    ap.add_argument("--extra-exclude-ext", nargs="*", default=[],
                    help="Additional extensions to exclude (e.g. --extra-exclude-ext pkl safetensors)")
    ap.add_argument("--include-images", action="store_true",
                    help="Include common image files (overrides default exclusion).")
    ap.add_argument("--stage-dir", type=Path, default=None,
                    help="Optional path to a persistent staging directory.")
    ap.add_argument("--keep-stage", action="store_true",
                    help="Keep the temporary staging directory for inspection.")
    ap.add_argument("--only-stage", action="store_true",
                    help="Only stage files; do not create the zip.")
    ap.add_argument("--force", action="store_true",
                    help="If --stage-dir exists and is non-empty, clear it first.")
    args = ap.parse_args()

    excluded_exts = set(DEFAULT_EXCLUDED_EXTS)
    if args.include_images:
        excluded_exts -= {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}
    excluded_exts |= parse_ext_list(args.extra_exclude_ext)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    # Decide staging directory
    temp_ctx = None
    if args.stage_dir:
        stage_dir = args.stage_dir.resolve()
        ensure_clean_dir(stage_dir, force=args.force)
    elif args.keep_stage:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_dir = (root / f".bundle_stage_{ts}").resolve()
        ensure_clean_dir(stage_dir, force=False)
    else:
        # ephemeral temp dir that will be auto-removed if not kept
        temp_ctx = tempfile.TemporaryDirectory(prefix="bundle_stage_")
        stage_dir = Path(temp_ctx.name).resolve()

    # 1) Stage files
    staged = stage_project(root, stage_dir, excluded_exts, excluded_dirs)
    print(f"Staged {len(staged)} files into: {stage_dir}")
    if not staged:
        print("Nothing to stage with current exclusions.")
    else:
        # show a short preview
        preview = "\n".join(str(p) for p in staged[:20])
        if preview:
            print("Preview of first 20 staged entries:\n" + preview)
        if len(staged) > 20:
            print(f"... and {len(staged) - 20} more.")

    print("Excluded extensions:", ", ".join(sorted(excluded_exts)) or "(none)")
    print("Excluded directories:", ", ".join(sorted(excluded_dirs)) or "(none)")

    if args.only_stage:
        # Keep stage if user requested it or provided a custom dir; otherwise it will auto-clean via temp_ctx
        if args.stage_dir or args.keep_stage:
            print(f"Staging kept at: {stage_dir}")
        else:
            print("Staging is temporary and will be cleaned up on exit.")
        return

    # 2) Zip staged content
    files_added = zip_from_dir(stage_dir, args.output)
    print(f"Created: {args.output}  |  Files added from stage: {files_added}")

    # Clean up temp staging if ephemeral
    if temp_ctx is not None:
        # context manager will clean automatically on exit
        pass
    else:
        # if user asked to keep, it's already persistent
        if args.stage_dir or args.keep_stage:
            print(f"Staging preserved at: {stage_dir}")

if __name__ == "__main__":
    main()
