# `results`

Small result artefacts written by the experiment code. The code writes them to the working directory; they were moved here for clarity.

## `hyperparameter_search/`

| File | Produced by | Contents |
|---|---|---|
| `retina_sweep.db` | `experiment/saliency_BO_sweep.py` | Optuna study `silicon_retina_optimization` (171 trials), saliency objective |
| `best_optuna_params.json` | `experiment/saliency_BO_sweep.py` | Best retina parameters of that study |
| `retina_bo.sqlite3` | `experiment/retina_parameter_search_using_BO.py` (`scripts/history/root/run_BO.sh`) | Optuna studies `retina_bo_20250909_*` (156 trials), intensity-reconstruction objective |
| `result.json`, `optuna_result.json`, `most_recent_BO_result.json` | `result_file` output of the training scripts | Best loss and event-occupancy statistics of the last run of each kind |

Inspect a study with:

```python
import optuna
study = optuna.load_study(study_name="silicon_retina_optimization",
                          storage="sqlite:///results/hyperparameter_search/retina_sweep.db")
print(study.best_trial.params, study.best_value)
```

To *resume* a study, copy the database back to the repository root, which is where the scripts look by default.

## `dataset_curation/`

Histograms of DHF1K clip lengths, before and after the train/validation split, made by `src/neural_network/data_help/partition_saliency.py`.
