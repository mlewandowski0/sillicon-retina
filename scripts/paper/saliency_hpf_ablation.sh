#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/<this script>
# Requires the prepared dataset (see README "Data and models") and the
# environment from scripts/setup/install.sh. Metrics are logged to Weights & Biases
# (set WANDB_MODE=offline to keep them local).
set -euo pipefail
C=src/neural_network/configs
RETINA="python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"
DVS="python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2"

# Fig. "hpf_ablation_bars": Simple (mode 0) vs IIR (mode 1) high-pass filter at the
# OPL (HPF1) and ganglion (HPF2) stages, at two event-rate regimes.
#            (HPF1, HPF2)   low rate p2=1.0       high rate p2=1.2
#            (0, 0)         best_t1.json          best_t1a.json
#            (1, 1)         best_t2.json          best_t2a.json
#            (0, 1)         best_t3.json          best_t3a.json
#            (1, 0)         best_t4.json          best_t4a.json
for t in t1 t2 t3 t4 t1a t2a t3a t4a; do
    $RETINA -c $C/saliency/retina/christmas_run/best_$t.json
done
