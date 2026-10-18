#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/<this script>
# Requires the prepared dataset (see README "Data and models") and the
# environment from scripts/setup/install.sh. Metrics are logged to Weights & Biases
# (set WANDB_MODE=offline to keep them local).
set -euo pipefail
C=src/neural_network/configs
RETINA="python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"
DVS="python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2"

# Fig. "feedback_comparison_detailed": no / |V| (absolute) / V^2 (quadratic) contrast-gain feedback.
# W&B runs in paper-figures/analysis/feedback_{train,val}_loss.csv.
$RETINA -c $C/saliency/retina/christmas_run/best_no_feedback.json   # Fire_saliency_no_fb
$RETINA -c $C/saliency/retina/christmas_run/best_quad_abs_fb.json   # Fire_saliency_abs_fb  (q_feedback_mode=1, |V|)
$RETINA -c $C/saliency/retina/christmas_run/best_quad_fb.json       # Fire_saliency_quad_fb (q_feedback_mode=0, V^2)
