import argparse
import json
import wandb
from datetime import datetime
import torch.optim as optim
import torch
import os
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity, schedule


from torch.utils.data import DataLoader
from tqdm import tqdm

from src.neural_network.networks.FIRE_network import FIRE_network
from src.neural_network.networks.SaliencyLoss import SaliencyLoss
from src.neural_network.dataloaders.SaliencyDatasetWrapper import SaliencyDatasetWrapper
from src.neural_network.dataloaders.reconstruction_utils import save_checkpoint  
from src.neural_network.dataloaders.SiliconRetinaEventDatasetFromUpsampledSaliency import SiliconRetinaEventDatasetFromUpsampledSaliency 
from esim_torch.silicon_retina_torch import SiliconRetinaConfig,SiliconRetinaFloatCUDAOnline
from src.neural_network.dataloaders.Encodings import EventRepresentation
from src.neural_network.utils.util import validat_occupancy_dataset

from torch.optim.lr_scheduler import StepLR

def write_optuna_result(json_path, best_value, total_pixel_occupancy, total_grid_occupancy,
                        val_pixel_occupancy = -1, val_grid_occupancy = -1,
                        train_pixel_occupancy = -1, train_grid_occupancy = -1):
    with open(json_path, "w") as f:
        json.dump({"best_value" : best_value,
                   "total_pixel_occupancy" : total_pixel_occupancy,
                   "total_grid_occupancy" : total_grid_occupancy,
                   "val_pixel_occupancy" :  val_pixel_occupancy,
                   "val_grid_occupancy" : val_grid_occupancy,
                   "train_pixel_occupancy" : train_pixel_occupancy,
                   "train_grid_occupancy" : train_grid_occupancy}, f, indent=4)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to json config file")
    args = ap.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = json.load(f)

    print("E2VID Training on ECOCO Dataset")
    print("================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    INP = config.get("num_bins", 5)
    # get result file
    result_file = config.get("result_file", "result.json")
    config["num_bins"] = INP

    model = FIRE_network(in_ch=INP,
                         hid1=config["hid1"],
                         ks1=config["ks1"],
                         add_bn=config["add_bn"],
                         out_mode=config["out_mode"]).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Silicon Retina Dataset parameters
    use_upsampled_from_OF = config.get("use_upsampled_from_OF", False)
    config["use_upsampled_from_OF"] = use_upsampled_from_OF

    # verbose flag
    verbose = config.get("verbose", False)
    config["verbose"] = verbose

    # Silicon Retina parameters
    se_cfg = SiliconRetinaConfig().from_dict(config["silicon_retina_config"])

    event_representation = config["event_representation_class"]
    event_representation = EventRepresentation(event_representation)
    event_representation_params = config["event_representation_params"]

    event_representation_params["num_bins"] = config.get("num_bins", 5)
    config["num_bins"] = event_representation_params["num_bins"]
    

    force_recomputing = config.get("force_recomputing", False)
    config["force_recomputing"] = force_recomputing

    log_input = config.get("log_input", False)

    # parameters for dataset checking
    min_grid_occupancy = config.get("min_grid_occupancy", 0.00)
    max_grid_occupancy = config.get("max_grid_occupancy", 1.0)
    min_per_pixel_occupancy = config.get("min_per_pixel_occupancy", 0.0)
    max_per_pixel_occupancy = config.get("max_per_pixel_occupancy", 1.0) 

    config["log_input"] = log_input

    skip_frames = config.get("skip_frames", False)
    config["skip_frames"] = skip_frames

    use_lpips = config.get('use_lpips', False)
    config["use_lpips"] = use_lpips

    # print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    train_base = SiliconRetinaEventDatasetFromUpsampledSaliency(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "train"),
                                                                config=se_cfg,
                                                                prediction_video_name="outputs.npz",
                                                                preprocessing_params=event_representation_params,
                                                                preprocessing_type=event_representation,
                                                                log_scale=log_input,
                                                                force_recomputing=force_recomputing,
                                                                limit=config["train_limit"],
                                                                use_upsampled_from_OF=use_upsampled_from_OF,
                                                                verbose=verbose)
    # compute the occupancy of the training dataset 
    valid, grids_occupancy_train, per_pixel_occupany_train = validat_occupancy_dataset(train_base, 
                                                                                min_grid_occupancy=min_grid_occupancy, 
                                                                                max_grid_occupancy=max_grid_occupancy,
                                                                                min_per_pixel_occupancy=min_per_pixel_occupancy,
                                                                                max_per_pixel_occupancy=max_per_pixel_occupancy,
                                                                                result_file=result_file,
                                                                                truncate=False)
    
    
    # use train or validation for validation
    val_on_train = config.get("val_on_train", False)


    val_base = SiliconRetinaEventDatasetFromUpsampledSaliency(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "val"),
                                                                config=se_cfg,
                                                                prediction_video_name="outputs.npz",
                                                                preprocessing_params=event_representation_params,
                                                                preprocessing_type=event_representation,
                                                                log_scale=log_input,
                                                                force_recomputing=force_recomputing,
                                                                limit=config["val_limit"],
                                                                use_upsampled_from_OF=use_upsampled_from_OF,
                                                                verbose=verbose)

    # compute the occupancy of the validation dataset 
    valid, grids_occupancy_val, per_pixel_occupany_val = validat_occupancy_dataset(val_base, 
                                                                                min_grid_occupancy=min_grid_occupancy, 
                                                                                max_grid_occupancy=max_grid_occupancy,
                                                                                min_per_pixel_occupancy=min_per_pixel_occupancy,
                                                                                max_per_pixel_occupancy=max_per_pixel_occupancy,
                                                                                result_file=result_file,
                                                                                truncate=False)


    # compute average occupancy
    total = len(train_base) + len(val_base)
    grids_occupancy = (grids_occupancy_train * len(train_base) + grids_occupancy_val * len(val_base)) / total
    per_pixel_occupany = (per_pixel_occupany_train * len(train_base) + per_pixel_occupany_val * len(val_base)) / total

    config["grids_occupancy"] = grids_occupancy
    config["per_pixel_occupany"] = per_pixel_occupany
    
    print(f"Average occupancy over train and val datasets: Grids occupancy: {grids_occupancy*100:.4f} %, Per pixel occupancy: {per_pixel_occupany*100:.4f} %")
    print(f"Occupancy train dataset: Grids occupancy: {grids_occupancy_train*100:.4f} %, Per pixel occupancy: {per_pixel_occupany_train*100:.4f} %")
    print(f"Occupancy val dataset: Grids occupancy: {grids_occupancy_val*100:.4f} %, Per pixel occupancy: {per_pixel_occupany_val*100:.4f} %")
    
    write_optuna_result(result_file, best_value=-1, 
                        total_pixel_occupancy=per_pixel_occupany, total_grid_occupancy=grids_occupancy,
                        val_pixel_occupancy = per_pixel_occupany_val, val_grid_occupancy = grids_occupancy_val,
                        train_pixel_occupancy = per_pixel_occupany_train, train_grid_occupancy= grids_occupancy_train)
