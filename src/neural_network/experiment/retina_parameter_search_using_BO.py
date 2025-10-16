#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.samplers import TPESampler
import shutil


from esim_torch.silicon_retina_torch import SiliconRetinaConfig

from src.neural_network.dataloaders.Encodings import EventRepresentation
from src.neural_network.utils.util import make_config_dictionary


def occupancy_penalty(g, p, lo_g, hi_g, lo_p, hi_p, base=1e3, w=1e4):
    def hinge(v, lo, hi):
        if v < lo: return lo - v
        if v > hi: return v - hi
        return 0.0
    return float(base + w*hinge(g, lo_g, hi_g)**2 + w*hinge(p, lo_p, hi_p)**2)

def suggest_silicon_retina_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    p = {}
    p["alpha1"] = trial.suggest_float("alpha1", 0.1, 0.9)
    # p["alpha2"] = trial.suggest_float("alpha2", 0.1, 0.9)
    p["alpha2"] = p["alpha1"]
    
    p["V_BIP_LPF_alpha"] = trial.suggest_float("V_BIP_LPF_alpha", 0.05, 0.95)
    p["lambda_OPL"] = trial.suggest_float("lambda_OPL", 2.0, 5.0)
    p["G_A_zero"] = trial.suggest_float("G_A_zero", 0.0, 1.5)
    p["lambda_A"] = trial.suggest_float("lambda_A", 0.0, 2.0)
    p["p1"] = trial.suggest_float("p1", 0.0, 1.0)
    p["I_bias_on"] = trial.suggest_float("I_bias_on", 1e-4, 0.5)
    p["p2"] = trial.suggest_float("p2", 0.2, 3.0)
    p["p3"] = trial.suggest_float("p3", 0.1, 2.0)
    p["U_rest_on"]  = trial.suggest_float("U_rest_on",  -0.1, 0.1)
    p["U_th_on"]    = trial.suggest_float("U_th_on",    0.01, 0.2)
    p["U_reset_on"]  = trial.suggest_float("U_reset_on",  0, 0.2)
    p["HPF2_param"] = trial.suggest_float("HPF2_param", 0.0, 1.0)

    p["HPF1_mode"]  = trial.suggest_categorical("HPF1_mode", [0, 1])
    p["HPF2_mode"]  = trial.suggest_categorical("HPF2_mode", [0, 1])
  
    p["I_bias_off"] = p["I_bias_on"]
    p["U_rest_off"] = p["U_rest_on"]
    p["U_th_off"]   = p["U_th_on"]
    p["U_reset_off"] = p["U_reset_on"]
    p["HPF1_param"] = p["alpha1"]

    # p["apply_log_to_input"] = trial.suggest_categorical("apply_log_to_input", [False, True])
    return p


def run_training(module: str, config_path: Path, result_path: Path,
                 cuda: Optional[str], timeout_s: Optional[int]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)

    cmd = [sys.executable, "-m", module, "-c", str(config_path)]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def read_result_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def objective_factory(args, study_root: Path):
    # Constant pieces that your make_config_dictionary will fill
    event_representation = EventRepresentation[args.event_repr]
    event_representation_params = json.loads(args.event_repr_params) if args.event_repr_params else {}

    def objective(trial: optuna.trial.Trial) -> float:
        trial_dir = study_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sample retina params and instantiate cfg
        params = suggest_silicon_retina_params(trial)
        se_cfg = SiliconRetinaConfig().from_dict(params)

        # per-trial result file (unique to avoid races)
        result_file = trial_dir / "result.json"
        prefix = f"{args.prefix}_trial{trial.number:04d}"

        # Build config for a quick-but-meaningful trial
        cfg = make_config_dictionary(
            num_epochs=args.epochs,
            burn_in_steps=args.burn_in_steps,
            sequence_length=args.sequence_length,
            force_recomputing=True,                # ensure fresh events per trial
            use_wandb=args.enable_wandb,  # keep quiet by default here
            log_input=args.log_input,
            event_type=event_representation,
            event_representation_params=event_representation_params,
            silicon_retina_config=se_cfg,
            prefix=prefix,
            result_file=str(result_file),
            check_dataset=True,
            min_per_pixel_occupancy=args.min_per_pixel_occupancy,
            max_per_pixel_occupancy=args.max_per_pixel_occupancy,
            min_grid_occupancy=args.min_grid_occupancy,
            max_grid_occupancy=args.max_grid_occupancy,
            val_on_train=args.val_on_train,
            train_limit=args.train_limit if args.train_limit >= 0 else None,
            val_limit=args.val_limit if args.val_limit >= 0 else None,
            wandb_project_name = "e2vid-ecoco-retina-BO",
            wandb_prefix = "BO_FIREnetwork",
        )

        # Write the config.json we’ll pass to the runner
        config_path = trial_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Launch training
        proc = run_training(args.module, config_path, result_file, args.cuda, args.timeout_s)
        print(proc)

        # Try to read the result file (preferred)
        result = read_result_file(result_file)
 
        # Handle crashes/timeouts / missing results
        if result is None:
            print("*** Trial failed! *****")
            print(result) 
                   
            trial.set_user_attr("returncode", proc.returncode)
            tail = (proc.stderr or "")[-2000:]
            trial.set_user_attr("stderr_tail", tail)
            raise optuna.TrialPruned("Runner failed or result.json missing.")

        # Extract metrics
        best_value = float(result.get("best_value", float("inf")))
        pixel_occ = float(result.get("pixel_occupancy", float("nan")))
        grid_occ = float(result.get("grid_occupancy", float("nan")))

        # Attach for dashboarding
        trial.set_user_attr("pixel_occupancy", pixel_occ)
        trial.set_user_attr("grid_occupancy", grid_occ)
        trial.set_user_attr("used_params", params)

        # turn your "invalid" sentinel into a numeric penalty
        if best_value == -1:
            return occupancy_penalty(
                grid_occ, pixel_occ,
                args.min_grid_occupancy, args.max_grid_occupancy,
                args.min_per_pixel_occupancy, args.max_per_pixel_occupancy
            )

        # if the process crashed *after* writing a valid result, still accept the metric
        if not math.isfinite(best_value):
            raise optuna.TrialPruned("Non-finite loss.")

        return best_value
    
    return objective


def save_best_artifacts(study: optuna.Study, args, study_root: Path):
    best = study.best_trial
    print(f"\n=== Best trial ===\n# {best.number}  val={best.value:.6f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save just the retina params
    (study_root / "best_params.json").write_text(json.dumps(best.params, indent=2))

    # Build a full config for a “proper” training run (restore your larger budgets)
    se_cfg = SiliconRetinaConfig().from_dict(best.params)
    event_representation = EventRepresentation[args.event_repr]
    event_representation_params = json.loads(args.event_repr_params) if args.event_repr_params else {}

    best_cfg = make_config_dictionary(
        num_epochs=args.final_epochs,
        burn_in_steps=args.burn_in_steps,
        sequence_length=args.sequence_length,
        force_recomputing=False,                # reuse any caches you keep
        use_wandb=not args.enable_wandb,
        log_input=args.log_input,
        event_type=event_representation,
        event_representation_params=event_representation_params,
        silicon_retina_config=se_cfg,
        prefix=f"{args.prefix}_best",
        result_file=str(study_root / "best_result.json"),
        check_dataset=True,
        min_per_pixel_occupancy=args.min_per_pixel_occupancy,
        max_per_pixel_occupancy=args.max_per_pixel_occupancy,
        min_grid_occupancy=args.min_grid_occupancy,
        max_grid_occupancy=args.max_grid_occupancy,
        val_on_train=args.val_on_train,
        train_limit=None if args.final_train_limit < 0 else args.final_train_limit,
        val_limit=None if args.final_val_limit < 0 else args.final_val_limit,
    )

    (study_root / "best_full_config.json").write_text(json.dumps(best_cfg, indent=2))
    print(f"\nSaved:\n  {study_root / 'best_params.json'}\n  {study_root / 'best_full_config.json'}")


def main():
    ap = argparse.ArgumentParser()
    # Runner module (use your correct path; override if needed)
    ap.add_argument("--module", default="src.neural_network.experiment.reconstruct_intensity_using_FIREnetwork_silicon_retina_v2",
                    help="Python -m path to your training runner.")
    ap.add_argument("--storage", default="sqlite:///retina_bo.sqlite3", help="Optuna storage for resume/parallel.")
    ap.add_argument("--study-name", default=f"retina_bo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1, help="Optuna parallel workers in this process.")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g., '0').")
    ap.add_argument("--timeout-s", type=int, default=None, help="Per-trial timeout (seconds).")
    ap.add_argument("--enable-wandb", action="store_true", help="Opt into wandb logging.")
    ap.add_argument("--prefix", default="retina_BO_FIRENetwork")

    # Quick trial budget
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--final-epochs", type=int, default=20)
    ap.add_argument("--burn-in-steps", type=int, default=10)
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--train-limit", type=int, default=4)
    ap.add_argument("--val-limit", type=int, default=4)
    ap.add_argument("--final-train-limit", type=int, default=-1)
    ap.add_argument("--final-val-limit", type=int, default=-1)
    ap.add_argument("--log-input", action="store_true")

    # Event representation
    ap.add_argument("--event-repr", dest="event_repr", default="EVENTS",
                    choices=[e.name for e in EventRepresentation])
    ap.add_argument("--event-repr-params", default="",
                    help='JSON string (e.g. \'{"num_bins":2}\')')

    # Dataset guardrails (mirrors your validator)
    ap.add_argument("--min-grid-occupancy", type=float, default=0.01)
    ap.add_argument("--max-grid-occupancy", type=float, default=0.7)
    ap.add_argument("--min-per-pixel-occupancy", type=float, default=0.0)
    ap.add_argument("--max-per-pixel-occupancy", type=float, default=1.0)
    ap.add_argument("--val-on-train", action="store_true")

    ap.add_argument("--study-root", default=None, help="Where to store per-trial configs/results.")
    args = ap.parse_args()

    # Trial artifacts root
    if args.study_root is None:
        # Put beside this script in a dated folder
        base = Path(__file__).resolve().parent
        date_str = datetime.now().strftime("%d_%m_%Y_%H")
        study_root = base / f"BO_configs_started={date_str}"
    else:
        study_root = Path(args.study_root)
    study_root.mkdir(parents=True, exist_ok=True)

    # Create/reuse study
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True, n_startup_trials=10)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )
    
    study.enqueue_trial({
        "alpha1": 0.5, 
        "alpha2": 0.5, 
        "V_BIP_LPF_alpha": 0.5, 
        "lambda_OPL": 3.0,
        "G_A_zero": 0.5, 
        "lambda_A": 0.5, 
        "p1": 0.25, 
        "p2": 1.0, 
        "p3": 0.2,
        "I_bias_on": 0.01,
        "I_bias_off": 0.01,
        "U_rest_on": -0.02,
        "U_rest_off": -0.02,
        "U_th_on": 0.05, 
        "U_th_off": 0.05,
        "U_reset_on": 0.0, "U_reset_off": 0.0,
        "HPF1_mode": 0, 
        "HPF2_mode": 0,
        "HPF1_param": 0.5, 
        "HPF2_param": 0.5,
        "apply_log_to_input": False,
    })

    objective = objective_factory(args, study_root)

    try:
        study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    save_best_artifacts(study, args, study_root)


if __name__ == "__main__":
    main()
