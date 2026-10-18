#!/usr/bin/env bash
# Download the FILM frame-interpolation weights (redistributed by the vid2e
# authors) into models/film_net/, which is where the upsampling code expects
# them (models/film_net/Style/saved_model).
#
# Run from the repository root:  bash scripts/setup/download_model.sh
set -euo pipefail

mkdir -p models
wget https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip -O models/temp.zip
unzip -q models/temp.zip -d models/
# The archive unpacks to models/pretrained_models/film_net/{L1,Style,VGG}
mv models/pretrained_models/film_net models/film_net
rm -rf models/pretrained_models   # also holds an unused vgg/*.mat
rm models/temp.zip
echo "FILM weights installed to models/film_net/"
