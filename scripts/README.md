# `scripts`

All scripts are run **from the repository root**.

| Path | Purpose |
|---|---|
| [`setup/install.sh`](setup/install.sh) | Installs PyTorch, `requirements.txt` and builds the `esim_torch` CUDA extension |
| [`setup/download_model.sh`](setup/download_model.sh) | Downloads the FILM interpolation weights to `models/film_net/` (needed only for dataset upsampling) |
| [`setup/requirements-full-freeze.txt`](setup/requirements-full-freeze.txt) | `pip freeze` of an original experiment environment, for reference |
| [`smoke_test.py`](smoke_test.py) | End-to-end check on synthetic data: DVS + retina simulation, voxel encoding, one FireNet training step. Run `python -m scripts.smoke_test` |
| [`synthetic_data/`](synthetic_data/) | `run_synthetic_pipeline.sh` generates tiny synthetic DHF1K-/ECOCO-format datasets (`make_synthetic_dataset.py`) and runs every training / analysis entry point on them for 2 epochs with small configs (`configs/`). It refuses to run if a real dataset is present |
| [`paper/`](paper/) | One script per paper experiment. Each lists the configs used and the W&B run names that appear in `paper-figures/analysis/*.csv` |
| [`package_for_cluster.py`](package_for_cluster.py) | Zips the code (without data, media or checkpoints) for transfer to a compute cluster: `python scripts/package_for_cluster.py . -o bundle.zip` |
| [`history/`](history/) | The job scripts as they were originally run (dates in file names), kept verbatim as a record of the experiment history |

## `history/`

These files are unchanged copies of the shell scripts used during the project:

* `runners/`: saliency experiments, Nov 2025 – Jan 2026;
* `jobs/`, `root/`: early intensity-reconstruction and saliency runs, and the parameter search (`root/run_BO.sh`).

Some of them reference configs that were later renamed or edited in place, so they are not guaranteed to run as-is. Use `paper/` to re-run experiments.
