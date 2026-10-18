#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/<this script>
# Requires the prepared dataset (see README "Data and models") and the
# environment from scripts/setup/install.sh. Metrics are logged to Weights & Biases
# (set WANDB_MODE=offline to keep them local).
set -euo pipefail
C=src/neural_network/configs
RETINA="python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"
DVS="python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2"

# Fig. "sparsity_performance_sweep_annotated": saliency loss vs. event-grid occupancy.
# Retina: sweep of the ganglion gain p2 (W&B runs Fire_saliency_best1_b*).
#   best_b1..b5 hold p2 = 0.9, 1.05, 1.1, 1.2, 1.3. The figure also contains
#   p2 = 1.35, 1.4, 1.5 runs, all logged as "best1_b5"; they were obtained by
#   editing p2 in best_b5.json between runs.
for b in b1 b2 b3 b4 b5; do
    $RETINA -c $C/saliency/retina/christmas_run/best_$b.json
done
# DVS: contrast-threshold sweep (W&B runs dvs_threshold=<thr>). Configs are
# provided for thresholds 0.15-0.475; higher thresholds in the figure used the
# same config with contrast_threshold_{pos,neg} and wandb_name changed.
for f in $C/saliency/events/thr_sweep/*.json; do
    $DVS -c "$f"
done
# Event occupancy of a retina configuration (no training):
# python -m src.neural_network.experiment.compute_occupancy_retina -c $C/occupancy_plots/occ.json -o occupancy_results
