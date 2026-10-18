#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/<this script>
# Requires the prepared dataset (see README "Data and models") and the
# environment from scripts/setup/install.sh. Metrics are logged to Weights & Biases
# (set WANDB_MODE=offline to keep them local).
set -euo pipefail
C=src/neural_network/configs
RETINA="python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"
DVS="python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2"

# Fig. "spatial_kernel_ablation_final": size/type of the contrast-gain-control spatial kernel.
# W&B runs in paper-figures/analysis/kernel_sweep_{train,val}_loss.csv (best_k_eq_<k>_<type>_*).
for k in 1 3 5 7 9; do
    for kind in gauss uniform; do
        $RETINA -c $C/saliency/retina/kernel_sweep/best_k_eq_${k}_${kind}.json
    done
done
