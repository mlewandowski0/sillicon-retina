# `paper-figures`: figures and the data behind them

The notebooks re-create the result figures from Weights & Biases exports, without re-training. **Run them with `paper-figures/` as the working directory.** They need `pandas`, `matplotlib`, `seaborn` and `SciencePlots` (all in the root `requirements.txt`).

```bash
cd paper-figures
jupyter nbconvert --to notebook --execute analysis_saliency.ipynb --output analysis_saliency.ipynb
jupyter nbconvert --to notebook --execute analysis_intensity.ipynb --output analysis_intensity.ipynb
```

Both notebooks were verified to run top-to-bottom and regenerate every PNG listed below.

## Result figures

`analysis/*.csv` are raw W&B exports. Column names are `<run name> - <metric>`, where the run name is the `wandb_prefix` / `wandb_name` of a config in `src/neural_network/configs/` (see [`scripts/paper/`](../scripts/paper/)).

| Output | Notebook (cell index, 0-based) | Input data |
|---|---|---|
| `intensity_training_loss.png`, `intensity_validation_loss.png`, `intensity_convergence_curves.png` | `analysis_intensity.ipynb` | `analysis/intensity-{train,val}.csv` |
| `saliency_training_loss.png`, `saliency_validation_loss.png` | `analysis_saliency.ipynb` (6, 11) | `analysis/retina_{train,val}_saliency.csv` |
| `feedback_comparison_detailed.png` | `analysis_saliency.ipynb` (16) | final losses and occupancies of the three feedback runs, entered as arrays in the cell (the CSVs `analysis/feedback_{train,val}_loss.csv` hold the full curves) |
| `feedback_comparison_detailed_3.png` | `analysis_saliency.ipynb` (17) | `analysis/feedback_{train,val}_loss.csv` |
| `sparsity_performance_sweep.png` | `analysis_saliency.ipynb` (20) | losses entered in the cell |
| `sparsity_performance_sweep_robust.png`, `sparsity_performance_sweep_annotated.png` | `analysis_saliency.ipynb` (22, 23) | losses from `analysis/bandwidth_{train,val}.csv` and `analysis/dvs_sweep_{train,val}.csv` (top-5 mean); grid occupancies entered in the cell |
| `hpf_ablation_bars.png` | `analysis_saliency.ipynb` (26) | final losses and occupancies of the eight HPF runs, entered in the cell |
| `spatial_kernel_ablation_final.png` | `analysis_saliency.ipynb` (29–30) | `analysis/kernel_sweep_val_loss.csv` |
| `upsampling_convergence_curves_fixed.png`, `upsampling_comparison_boxplot_{train,val}.png`, `{train,val}_stats_*.csv` | `analysis_saliency.ipynb` (33–37) | `analysis/comparison_{train,val}_loss.csv` |

Event-occupancy statistics are produced by the training scripts (`result_file` JSON) and by `src/neural_network/experiment/compute_occupancy_retina.py`. Stored sweep results are in `src/neural_network/configs/occupancy_sweeps/results/`.

## Qualitative figures and diagrams

| Output | Source |
|---|---|
| Visual abstract, panels (d)/(e) | `visualization_figures/run_whole.py` (or `run_whole.ipynb`) on `high_speed_output.mp4`, then composed in `retina-figs*.pptx` |
| Retina pipeline figure (`retina-pipeline.png`) | `visualization_figures/abstract_and_digital_output.ipynb` on `high_speed_output.mp4` (writes every stage to `retina_visualization/`), then composed |
| `Tasks.png` (experiment overview) | `Tasks.drawio` |
| Superpixel diagram | `retina-figs.pptx`, slide 1 |
| `scamp5.png`, `scamp5-host.png` | Screenshots of the SCAMP-5 host app. `render_col_scamp5.py` turns host-app recordings into videos. Its input folder is set at the bottom of the script |

`high_speed_output.mp4` is a VD55G0 recording made with `src/rt_visualization/windows/record_hs_video.py`. The `visualization_figures` notebooks need a CUDA GPU and the `esim_torch` package (see the root README).
