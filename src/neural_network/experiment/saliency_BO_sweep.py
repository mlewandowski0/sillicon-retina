import optuna
import json
import os
import subprocess
import sys
import copy
import tempfile
import shutil

# --- Configuration ---
# Path to your base configuration file (the template)
BASE_CONFIG_PATH = "src/neural_network/configs/saliency_retina_BO_base.json"

# The python module to run
MODULE_NAME = "src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"

# Max grid occupancy allowed (0.2 = 20%)
MAX_GRID_OCCUPANCY = 0.20

# Where to store the optuna database (allows resuming if the sweep crashes)
STORAGE_URL = "sqlite:///retina_sweep.db"
STUDY_NAME = "silicon_retina_optimization"

def objective(trial):
    # 1. Load the base config
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # 2. Define the Hyperparameter Search Space
    # We modify the 'silicon_retina_config' section of the JSON
    
    # --- OPL Layer ---
    config["silicon_retina_config"]["alpha1"] = trial.suggest_float("alpha1", 0.1, 0.9)
    config["silicon_retina_config"]["alpha2"] = trial.suggest_float("alpha2", 0.1, 0.9)
    config["silicon_retina_config"]["lambda_OPL"] = trial.suggest_float("lambda_OPL", 1.0, 10.0)
    
    # --- Bipolar / Gain Control ---
    config["silicon_retina_config"]["V_BIP_LPF_alpha"] = trial.suggest_float("V_BIP_LPF_alpha", 0.1, 0.9)
    config["silicon_retina_config"]["G_A_zero"] = trial.suggest_float("G_A_zero", 0.0, 1.0)
    config["silicon_retina_config"]["lambda_A"] = trial.suggest_float("lambda_A", 0.1, 5.0)
    config["silicon_retina_config"]["p1"] = trial.suggest_float("p1", 0.05, 1.0) # Bipolar update rate
    
    # --- Ganglion / Spiking ---
    config["silicon_retina_config"]["p2"] = trial.suggest_float("p2", 0.1, 2.0) # Ganglion update rate
    config["silicon_retina_config"]["p3"] = trial.suggest_float("p3", 0.05, 1.0) # Current scaler
    
    # Thresholds (Assuming symmetric ON/OFF for simplicity, can be split if needed)
    u_th = trial.suggest_float("U_th", 0.005, 0.1)
    config["silicon_retina_config"]["U_th_on"] = u_th
    config["silicon_retina_config"]["U_th_off"] = u_th
    
    u_reset = trial.suggest_float("U_reset", 0.01, 0.15)
    config["silicon_retina_config"]["U_reset_on"] = u_reset
    config["silicon_retina_config"]["U_reset_off"] = u_reset
    
    # --- Other Settings ---
    # Ensure dataset checking is ON so we can fail early on bad parameters
    config["check_dataset"] = True
    
    # 3. Setup Temporary Files for this Trial
    # We need a unique config file and a unique result file for this process
    trial_id = trial.number
    temp_config_path = f"temp_config_trial_{trial_id}.json"
    temp_result_path = f"temp_result_trial_{trial_id}.json"
    
    # Inject the result file path so the subprocess knows where to write
    config["result_file"] = temp_result_path
    
    # Save the modified config
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=4)

    try:
        # 4. Run the Training Script as a Subprocess
        # This isolates memory usage. 
        cmd = [sys.executable, "-m", MODULE_NAME, "-c", temp_config_path]
        
        print(f"\n[Optuna] Starting Trial {trial_id}...")
        
        # We capture stdout/stderr to prevent clutter, or remove capture_output=True to see logs
        # Using check=False because the script might exit(1) on dataset validation failure
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False)

        # 5. Read the Result
        if not os.path.exists(temp_result_path):
            print(f"[Optuna] Trial {trial_id} failed: No result file generated.")
            raise optuna.TrialPruned()

        with open(temp_result_path, 'r') as f:
            metrics = json.load(f)

        # 6. Apply Constraints & Return Objective
        best_val_loss = metrics.get("best_value", -1)
        grid_occupancy = metrics.get("total_grid_occupancy", 1.0)
        
        # Case A: The script explicitly reported failure (e.g. invalid dataset)
        if best_val_loss == -1:
            print(f"[Optuna] Trial {trial_id} Pruned: Invalid Dataset (Occupancy: {grid_occupancy:.2%})")
            raise optuna.TrialPruned()
            
        # Case B: The optimizer constraint (Double check logic)
        if grid_occupancy > MAX_GRID_OCCUPANCY:
            print(f"[Optuna] Trial {trial_id} Pruned: Grid Occupancy too high ({grid_occupancy:.2%})")
            # We can either prune, or return a very high loss to discourage this area
            # Returning infinity is safer for Bayesian Optimization than pruning sometimes
            return float('inf')

        # Case C: Success
        print(f"[Optuna] Trial {trial_id} Success. Loss: {best_val_loss}, Occupancy: {grid_occupancy:.2%}")
        
        # Report the occupancy as a custom user attribute for analysis later
        trial.set_user_attr("grid_occupancy", grid_occupancy)
        
        return best_val_loss

    except Exception as e:
        print(f"[Optuna] Exception in trial {trial_id}: {e}")
        raise optuna.TrialPruned()

    finally:
        # 7. Cleanup Temporary Files
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if os.path.exists(temp_result_path):
            os.remove(temp_result_path)

if __name__ == "__main__":
    # Check if DB exists to resume or create new
    print(f"Initializing Optuna Study: {STUDY_NAME}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="minimize", # We want to minimize Validation Loss
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # Optional: helps if you report intermediate steps
    )

    # Run the optimization
    # n_trials=50 is a good start. Since it's expensive, you might want fewer.
    try:
        study.optimize(objective, n_trials=25)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("\n" + "="*50)
    print("Study Statistics")
    print("="*50)
    print(f"Best Trial ID: {study.best_trial.number}")
    print(f"Best Value (Loss): {study.best_value}")
    print(f"Best Params: ")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best params to a file for easy loading
    with open("best_optuna_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
