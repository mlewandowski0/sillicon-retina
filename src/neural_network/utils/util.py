import json 
from typing import Dict
from src.neural_network.dataloaders.Encodings import EventRepresentation
from esim_torch import SiliconRetinaConfig
from tqdm import tqdm
import torch

def update_json(json_path, kvs : Dict):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    for k, v in kvs.items():
        data[k] = v
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
        
def make_config_dictionary(num_epochs : int,
                           burn_in_steps : int,
                           sequence_length : int,
                           force_recomputing : bool,
                           use_wandb : bool,
                           log_input : bool, 
                           event_type : EventRepresentation,
                           silicon_retina_config : SiliconRetinaConfig,
                           event_representation_params : Dict,
                           result_file : str,
                           wandb_project_name : str = "e2vid-ecoco",
                           wandb_prefix : str = "FIREnetwork",
                           prefix : str = "test",
                           check_dataset : bool = True,
                           min_per_pixel_occupancy : float = 0.0,
                           max_per_pixel_occupancy : float = 1.0,
                           min_grid_occupancy : float = 0.01,
                           max_grid_occupancy : float = 0.6,
                           val_on_train : bool = False,
                           
                           neural_network_params = {"hid1" : 32, "ks1" : 3,     # assume firenetwork
                                                    "add_bn" : False,
                                                    "out_sigmoid" : False,}, 
                           use_amp=True,
                           amp_dtype="fp16",
                           train_limit = None,
                           val_limit = None,
                           batch_size = 2,                           
                           num_bins=5,
                           num_prev_images = 3
                           ):
    
    
    log_str = ""
    if log_input:
        log_str = "log_input"
    else:
        log_str = "no_log_input"
        
    event_type_str = ""
    if event_type == EventRepresentation.EVENTS:
        event_type_str = "events"
    elif event_type == EventRepresentation.EVENTS_TIME_PLUS_POS: 
        event_type_str = "events_pos"
    elif event_type == EventRepresentation.SPIKE_TRACE_BANKS5:
        event_type_str = "tau_filters"   
    
    folder_name = f"{prefix}_length={sequence_length}_burnin={burn_in_steps}_{log_str}_{event_type_str}"
    
    basic_config = {
        "num_bins" : num_bins,
        "num_prev_images" : num_prev_images,
        "burn_in_steps" : 10, 
        "sequence_length" : sequence_length,
        "keep_last_n_checkpoints" : 5,
        
        
        "use_amp" : use_amp,
        "amp_dtype" : amp_dtype,
        
        "force_recomputing" : force_recomputing,
        "log_input" : False,
        
        "data_root": "src/neural_network/data/ecoco_depthmaps_test",
        "cache_folder": "src/neural_network/data/ecoco_cache",
        
        "train_limit" : train_limit,
        "val_limit" : val_limit,
        "skip_frames" : 1,
        
        "batch_size" : batch_size,
        "num_epochs" : num_epochs,
        "learning_rate" : 1e-4,
        "weight_decay" : 1e-4,
        "lr_step_size" : 10,
        "lr_gamma" : 0.9,
        "prefetch_factor" : 2,
        "dropout_rate" : 0.2, 
        "augment" : True,
        "skip_type" : "sum",
        
        "debug_save" : True,
        "debug_iters" : 200,
        "debug_folder": f"src/neural_network/debug/{folder_name}",

        "use_lpips" : True, 
        
        "checkpoint_dir": f"src/neural_network/checkpoints/{folder_name}",
        "save_every" : 5,
        "keep_last_n_checkpoints]" : 5,
        
        "num_workers" : 10,
        "use_wandb" : use_wandb,
        
        "result_file": result_file,
        "check_dataset": check_dataset,
        "min_grid_occupancy" : min_grid_occupancy,
        "max_grid_occupancy" : max_grid_occupancy,
        "min_per_pixel_occupancy" : min_per_pixel_occupancy,
        "max_per_pixel_occupancy" : max_per_pixel_occupancy,
        "val_on_train" : val_on_train,
        
        "wandb_project_name" : wandb_project_name,
        "wandb_prefix" : wandb_prefix,
        
        "silicon_retina_config" : silicon_retina_config.to_dict(),
        "event_representation_class" : event_type.value,
        "event_representation_params" : event_representation_params
    } 
    
    config = basic_config | neural_network_params
    return config

def statistic_from_grids(grids, eps=1e-6):
    # grids: [T, B, H, W]
    A = grids.abs()

    # grid occupancy: fraction of non-zero voxels
    vox_nonzero = (A > eps).sum().item()
    vox_total   = A.numel()
    grids_occupancy = vox_nonzero / vox_total

    # per-pixel occupancy across time/bins: fraction of pixels that ever fired
    # collapse over T and B → [H,W]
    per_pix_sum = A.sum(dim=(0,1))
    per_pixel_non_zero = (per_pix_sum > eps).sum().item()
    WH = per_pix_sum.numel()
    per_pixel_occupancy = per_pixel_non_zero / WH

    return grids_occupancy, per_pixel_occupancy

def write_optuna_result(json_path, best_value, pixel_occupancy, grid_occupancy):
    with open(json_path, "w") as f:
        json.dump({"best_value" : best_value,
                   "pixel_occupancy" : pixel_occupancy,
                   "grid_occupancy" : grid_occupancy}, f, indent=4)

def check_dataset_occupancy(ds, min_grid_occupancy=0.01, max_grid_occupancy=0.6, 
                         min_per_pixel_occupancy=0., max_per_pixel_occupancy=1.0):    
    
    
    avg_grid_occupancy = 0.0
    avg_per_pixel_occupancy = 0.0
    
    for datapoint in tqdm(ds):
        grids = datapoint["grids"]
        grids_occupancy, per_pixel_occupancy = statistic_from_grids(grids)

        avg_grid_occupancy += grids_occupancy
        avg_per_pixel_occupancy += per_pixel_occupancy

        if grids_occupancy < min_grid_occupancy or grids_occupancy > max_grid_occupancy or per_pixel_occupancy < min_per_pixel_occupancy or per_pixel_occupancy > max_per_pixel_occupancy or per_pixel_occupancy > max_per_pixel_occupancy:
             print(f"Invalid dataset! Grids occupancy: {grids_occupancy*100:.4f} %, Per pixel occupancy: {per_pixel_occupancy*100:.4f} %")
             return False, grids_occupancy, per_pixel_occupancy
        

    avg_grid_occupancy /= len(ds)
    avg_per_pixel_occupancy /= len(ds)
    return True, avg_grid_occupancy, avg_per_pixel_occupancy

def validat_occupancy_dataset(ds, min_grid_occupancy=0.01, max_grid_occupancy=0.6, 
                         min_per_pixel_occupancy=0., max_per_pixel_occupancy=1.0,
                         result_file="optuna_result.json"):    
    
    valid, grids_occupancy, per_pixel_occupany = check_dataset_occupancy(ds, 
                                                                          min_grid_occupancy=min_grid_occupancy, 
                                                                          max_grid_occupancy=max_grid_occupancy,
                                                                          min_per_pixel_occupancy=min_per_pixel_occupancy,
                                                                          max_per_pixel_occupancy=max_per_pixel_occupancy)
    if not valid:
        print("The dataset is NOT valid! Exiting.")
        write_optuna_result(result_file, best_value=-1, pixel_occupancy=per_pixel_occupany, grid_occupancy=grids_occupancy)
        exit(1)
    else:
        print("The dataset is valid!")    
        print(f"Val dataset - avg grids occupancy: {grids_occupancy*100:.4f} %, avg per pixel occupancy: {per_pixel_occupany*100:.4f} %")

    return valid, grids_occupancy, per_pixel_occupany