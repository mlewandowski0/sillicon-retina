import os 
import sys
from esim_torch import SiliconRetinaConfig
from src.neural_network.dataloaders.SiliconRetinaEventDatasetFromUpsampled import SiliconRetinaEventDatasetFromUpsampled
from src.neural_network.dataloaders.Encodings import EventRepresentation
from src.neural_network.utils.util import make_config_dictionary
import torch
import json
from datetime import datetime

file = sys.argv[0]
dirname = os.path.dirname(file)
date_str = datetime.now().strftime("%d_%m_%Y_%H")
dirname = os.path.join(dirname, f"BO_configs_started={date_str}")
os.makedirs(dirname, exist_ok=True)

""" 
dataclass
class SiliconRetinaConfig:
    alpha1 : float = 0.5
    alpha2 : float = 0.5
    V_BIP_LPF_alpha : float = 0.5
    lambda_OPL : float = 3.0
    G_A_zero : float = 0.5
    lambda_A : float = 0.5
    p1 : float = 0.25
    I_bias_on : float = 0.01
    I_bias_off : float = 0.01
    p2 : float = 1.0
    p3 : float = 0.2
    U_rest_on : float = -0.02
    U_rest_off : float = -0.02

    U_th_on : float = 0.05
    U_th_off : float = 0.05

    U_reset_on : float = 0.0
    U_reset_off : float = 0.0
    
    HPF1_mode : int = 0
    HPF1_param : float = 0.5
    
    HPF2_mode : int = 0 
    HPF2_param : float = 0.5

    apply_log_to_input : bool = False
    
"""
se_cfg = SiliconRetinaConfig()
# se_cfg.alpha1 = propose_float(0.01, 1.0)
# ...

i = 0

event_representation = EventRepresentation.EVENTS   
event_representation_params = {}

result_file = "most_recent_BO_result.json"
config = make_config_dictionary(num_epochs=2,
                                burn_in_steps=10,
                                sequence_length=30,
                                force_recomputing=True,
                                use_wandb=False,
                                log_input=False,
                                event_type=event_representation,
                                event_representation_params=event_representation_params,
                                silicon_retina_config=se_cfg,
                                prefix="retina_BO_FIRENetwork",
                                result_file=result_file,
                                check_dataset=True,
                                min_per_pixel_occupancy=0.0,
                                max_per_pixel_occupancy=1.0,
                                min_grid_occupancy=0.01,
                                max_grid_occupancy=0.6,
                                val_on_train=True,
                                train_limit = 4,
                                val_limit=4)

current_config_name = os.path.join(dirname, f"config_{i}.json")
print(f"Writing config to {current_config_name}")

with open(current_config_name, "w") as f:
    json.dump(config, f, indent=4)
    
print(f"running the experiment with config {current_config_name}")
print(f"python -m src.neural_network.experiment.reconstruct_intensity_using_FIREnetwork_silicon_retina_v2 -c {current_config_name}")

with open(result_file, "r") as f:
    result = json.loads(f.read())
    
best_value = result["best_value"]
pixel_occupancy = result["pixel_occupancy"]
grid_occupancy = result["grid_occupancy"]

if best_value == -1:
    print("dataset was not OK, skipping these parameters")
    
print(f"dataset had loss of {best_value} and the pixel_occupancy={pixel_occupancy}, grid_occupancy={grid_occupancy}")
