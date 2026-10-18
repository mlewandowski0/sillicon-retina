#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/hyperparameter_search.sh
# Bayesian optimisation (Optuna/TPE) of the retina parameters.
# Studies from the paper are stored in results/hyperparameter_search/.
set -euo pipefail
# Saliency objective (base config src/neural_network/configs/saliency_retina_BO_base.json;
# writes retina_sweep.db and best_optuna_params.json to the working directory):
python -m src.neural_network.experiment.saliency_BO_sweep
# Intensity-reconstruction objective (see --help for all options):
# python -m src.neural_network.experiment.retina_parameter_search_using_BO --enable-wandb --trials 150 --seed 41 \
#     --epochs 50 --final-epochs 100 --train-limit 200 --val-limit 50 --max-grid-occupancy 0.75
