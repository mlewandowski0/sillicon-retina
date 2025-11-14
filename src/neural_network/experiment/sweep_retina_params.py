import json 
import os 
import copy 
import subprocess
from pathlib import Path
import csv 


# config 
BASE_CONFIG_PATH = Path("src/neural_network/configs/occupancy_plots/occ.json")

# Where to write all generated configs 
SWEEP_CONFIG_DIR = Path("src/neural_network/configs/occupancy_sweeps/configs/")

# where compute_occupancy_retina will write its result JSONs
RESULT_DIR = Path("src/neural_network/configs/occupancy_sweeps/results/")

# Module to call 
OCCUPANCY_MODULE = "src.neural_network.experiment.compute_occupancy_retina"

# visualization folder 
RENDER_FOLDER = Path("src/neural_network/configs/occupancy_sweeps/visualizations")

# visualizations per run 
RENDER_NUMBER = 30

PARAM_SWEEPS = {
    "p2": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5],
    "p3" : [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lambda_OPL" : [2.0, 3.0, 4.0, 5.0],
    "U_reset_on" : [0.01, 0.02, 0.03, 0.04, 0.05],
    
}

# If you want to tie "ON" and "OFF" versions for some params:
TIE_ON_OFF = {
    "U_th_on": "U_th_off",
    "I_bias_on": "I_bias_off",
    "U_reset_on" : "U_reset_off"
}

# path to CSV with summary of sweeps 
CSV_SUMMARY_PATH = RESULT_DIR / "sweep_summary.csv"

def load_base_config():
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = json.load(f)
    return base_config

def run_single_config(cfg_path, run_name):
    """
        Run : 
            python -m src.neural_network.experiment.compute_occupancy_retina -c cfg_path
    and return True/False depending on success.
    """
    
    out_folder = RENDER_FOLDER / run_name
    out_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = ['python', '-m', OCCUPANCY_MODULE, '-c', str(cfg_path),
           "-o", str(out_folder), 
           "-n", str(RENDER_NUMBER)]
    print("Running : ", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"  -> ERROR: return code {result.returncode}")
        return False
    return True


def read_result_json(result_path):
    """
    Read the JSON written by write_optuna_result / compute_occupancy_retina.

    Adjust the keys if your result JSON has different names.
    """
    if not result_path.exists():
        print(f"  -> WARNING: result file {result_path} not found")
        return None

    with open(result_path, "r") as f:
        data = json.load(f)

    # Try a few common key names
    pixel_keys = ["total_pixel_occupancy", "pixel_occupancy"]
    grid_keys = ["total_grid_occupancy", "grid_occupancy"]

    pixel_occ = None
    grid_occ = None

    for k in pixel_keys:
        if k in data:
            pixel_occ = data[k]
            break

    for k in grid_keys:
        if k in data:
            grid_occ = data[k]
            break

    if pixel_occ is None or grid_occ is None:
        print(f"  -> WARNING: could not find occupancy keys in {result_path}")
        print("     Keys:", list(data.keys()))

    return {
        "pixel_occupancy": pixel_occ,
        "grid_occupancy": grid_occ,
        "raw": data,
    }



def main():
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config()
    all_records = []

    for param_name, values in PARAM_SWEEPS.items():
        print("=" * 80)
        print(f"Sweeping parameter: {param_name}")
        print("=" * 80)

        for idx, val in enumerate(values):
            cfg = copy.deepcopy(base_cfg)

            # Set the parameter in silicon_retina_config
            se_cfg = cfg.setdefault("silicon_retina_config", {})
            se_cfg[param_name] = float(val)

            # Optionally tie OFF param to ON param
            if param_name in TIE_ON_OFF:
                tied_name = TIE_ON_OFF[param_name]
                se_cfg[tied_name] = float(val)
                print(f"  Tied {tied_name} = {val}")

            # Give each run its own result file
            base_name = f"result_{param_name}_{idx:03d}"
            result_file_name = f"{base_name}.json"
            result_path = RESULT_DIR / result_file_name
            cfg["result_file"] = str(result_path)

            # Also, disable wandb if it's in base cfg (just in case)
            if "use_wandb" in cfg:
                cfg["use_wandb"] = False

            # Write this config to disk
            cfg_name = f"occ_{param_name}_{idx:03d}.json"
            cfg_path = SWEEP_CONFIG_DIR / cfg_name
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=4)

            print(f"\n[{param_name} sweep] idx={idx}, value={val}")
            print(f"  Config: {cfg_path}")
            print(f"  Result file: {result_path}")

            # Run the occupancy script as a separate process
            ok = run_single_config(cfg_path, run_name=base_name)
            if not ok:
                print("  -> Skipping result read because run failed.")
                continue

            # Read back occupancy
            res = read_result_json(result_path)
            if res is None:
                continue

            pixel_occ = res["pixel_occupancy"]
            grid_occ = res["grid_occupancy"]

            print(
                f"  -> pixel_occupancy = {pixel_occ:.4%} "
                f"grid_occupancy = {grid_occ:.4%}"
                if (pixel_occ is not None and grid_occ is not None)
                else "  -> (could not parse occupancies)"
            )

            all_records.append({
                "param": param_name,
                "value": val,
                "pixel_occupancy": pixel_occ,
                "grid_occupancy": grid_occ,
                "result_file": str(result_path),
                "config_file": str(cfg_path),
            })

    # Print a small summary
    print("\n\nSUMMARY OF SWEEPS:")
    for rec in all_records:
        print(
            f"{rec['param']:>12} = {rec['value']:>8} | "
            f"pixel_occ = {rec['pixel_occupancy']:.4%} | "
            f"grid_occ = {rec['grid_occupancy']:.4%}"
        )

    # Optionally save to CSV
    if CSV_SUMMARY_PATH is not None:
        with open(CSV_SUMMARY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "param",
                    "value",
                    "pixel_occupancy",
                    "grid_occupancy",
                    "result_file",
                    "config_file",
                ],
            )
            writer.writeheader()
            for rec in all_records:
                writer.writerow(rec)

        print(f"\nSaved CSV summary to {CSV_SUMMARY_PATH}")


if __name__ == "__main__":
    main()