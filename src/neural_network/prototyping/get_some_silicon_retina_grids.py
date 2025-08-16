import os
import shutil

limit = 100

out_folder = "silicon_retina_grids"
root = os.path.join("src", "neural_network", "data", "upsampled_train")

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

vids = sorted([os.path.join(root, el) for el in os.listdir(root) if os.path.isdir(os.path.join(root, el))])
vids = vids[:limit]


for i,vid in enumerate(vids):
    shutil.copy(os.path.join(vid, "cache_B=5_retina_grids.npy"), os.path.join(out_folder,f"{i}.npy"))

shutil.make_archive(out_folder, 'zip', out_folder)