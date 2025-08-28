import json 
from typing import Dict


def update_json(json_path, kvs : Dict):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    for k, v in kvs.items():
        data[k] = v
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        