import os 
import shutil
import tqdm


inp = os.path.join("src", "neural_network", "data", "saliency_dataset_upsampled")
output_train = os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "train")
output_val = os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "val")

if __name__ == "__main__":
    
    inputs_dir = os.listdir(inp)
        
    train_dir = os.listdir(output_train)
    val_dir = os.listdir(output_val)
        
    for el in tqdm.tqdm(inputs_dir):
        
        if not os.path.isdir(os.path.join(inp, el)):
            continue
        
        if el in train_dir:
            inp_full_path = os.path.join(inp, el)
            output_full_path = os.path.join(output_train, el)
            
            if os.path.exists(output_full_path):               
                shutil.copy(os.path.join(inp_full_path, "video_upsampled_using_optical_flow.mp4"), os.path.join(output_full_path, "video_upsampled_using_optical_flow.mp4")) 
                shutil.copy(os.path.join(inp_full_path, "params.json"), os.path.join(output_full_path, "params.json")) 
            
            

        elif el in val_dir:
            inp_full_path = os.path.join(inp, el)
            output_full_path = os.path.join(output_val, el)
            
            if os.path.exists(output_full_path):               
                shutil.copy(os.path.join(inp_full_path, "video_upsampled_using_optical_flow.mp4"), os.path.join(output_full_path, "video_upsampled_using_optical_flow.mp4")) 
                shutil.copy(os.path.join(inp_full_path, "params.json"), os.path.join(output_full_path, "params.json")) 
                
            
        else:
            raise Exception(f"skipping {el} (not in train or val)")