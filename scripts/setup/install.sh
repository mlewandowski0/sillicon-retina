#!/usr/bin/env bash
# Install the Python environment used for the event-generation and training
# pipeline. Run from the repository root:  bash scripts/setup/install.sh
#
# Requires an NVIDIA GPU and a CUDA toolkit (nvcc) whose major version matches
# the PyTorch wheel below (cu130 = CUDA 13.x; use e.g. cu128 for CUDA 12.8).
set -euo pipefail

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# DVS simulator (ESIM) + silicon-retina CUDA kernels, compiled against the
# installed PyTorch. Provides the `esim_torch` package.
pip install -e src/neural_network/reusing_videos/esim_torch --no-build-isolation
