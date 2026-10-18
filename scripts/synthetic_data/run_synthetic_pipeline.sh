#!/usr/bin/env bash
# Run the paper's training / analysis entry points end-to-end on a tiny synthetic
# dataset (2 train + 1 val clips, 2 epochs), to check the full pipeline without
# downloading DHF1K or ECOCO.  Takes about a minute on one GPU; nothing is sent to W&B.
#
# Run from the repository root:  bash scripts/synthetic_data/run_synthetic_pipeline.sh
#
# The synthetic clips are written where the real datasets live
# (src/neural_network/data/{saliency_dataset_ready,upsampled_train,upsampled_validation}),
# so the script refuses to run if any of those folders already exists.
# Outputs (logs, checkpoints, result JSONs) go to synthetic_test_out/.
set -euo pipefail

D=src/neural_network/data
for d in $D/saliency_dataset_ready $D/upsampled_train $D/upsampled_validation; do
    if [ -e "$d" ]; then
        echo "Refusing to run: $d exists (a real dataset?). Move it away first." >&2
        exit 1
    fi
done

CFG=scripts/synthetic_data/configs
OUT=synthetic_test_out
mkdir -p $OUT
export WANDB_MODE=disabled

python scripts/synthetic_data/make_synthetic_dataset.py --layout ready
python scripts/synthetic_data/make_synthetic_dataset.py --layout ecoco --H 180 --W 240 --fps 27

run () {
    local name=$1; shift
    echo "=== $name"
    if python -m "$@" > $OUT/$name.log 2>&1; then
        grep -E "Best validation loss|Grids occupancy" $OUT/$name.log | tail -1
    else
        echo "FAILED - see $OUT/$name.log"; exit 1
    fi
}
E=src.neural_network.experiment
run saliency_retina   $E.saliency_using_FIREnetwork_silicon_retina               -c $CFG/retina_best.json
run saliency_dvs      $E.saliency_using_FIREnetwork_events_v2                    -c $CFG/events_seq50.json
run occupancy_retina  $E.compute_occupancy_retina                                -c $CFG/occ.json -o $OUT/occupancy
run intensity_retina  $E.reconstruct_intensity_using_FIREnetwork_silicon_retina_v2 -c $CFG/retina_intensity.json
run intensity_dvs     $E.reconstruct_intensity_using_FIREnetwork_events          -c $CFG/events_intensity.json

echo "All pipeline entry points ran successfully."
echo "Remove the synthetic data with:"
echo "  rm -rf $D/saliency_dataset_ready $D/upsampled_train $D/upsampled_validation $OUT"
