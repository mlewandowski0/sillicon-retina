#!/usr/bin/env bash
# Build the standalone CUDA retina, run it on a folder of PNG frames and tile
# every intermediate stage into output_<date>.mp4.
# Run from this directory:  bash run_and_stich.sh [frames_dir]
set -euo pipefail
FRAMES=${1:-frames}
make
./main "$FRAMES" 0          # mode 0 = save every stage to out/<stage>/<t>.npy
python stich_together_outputs.py
