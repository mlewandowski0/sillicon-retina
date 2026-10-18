#!/usr/bin/env bash
python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina -c src/neural_network/configs/FIREnetwork_retina_saliency_p2_eq_1.0.json
python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina -c src/neural_network/configs/FIREnetwork_retina_saliency_p2_eq_1.1.json
python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina -c src/neural_network/configs/FIREnetwork_retina_saliency_p2_eq_1.25.json