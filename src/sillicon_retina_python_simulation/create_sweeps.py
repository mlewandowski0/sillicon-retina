import random
import json
import sqlite3
import subprocess
import os
import tqdm

num_new_needed = 500

values = {
    "input_path" : ["datasets/recording/"],
    "save_path" : ["videos/sweeps"],
    "total_bits" : [5],
    "frac_bits" : [4],
    "g0" : [-20, -10, 0, 20, 40],
    "lambda_A" : [1, 2, 4, 8, 16, 31],
    "alpha1" : [1, 2, 4, 8],
    "alpha2" : [1, 2, 4, 8],
    "alpha3" : [1, 2, 4, 8],
    "alpha4" : [1, 2, 4, 8],
    "alpha5" : [1, 2, 4, 8],
    "alpha" : [1, 2, 4, 8, 16, 31],
    "I_bias" : [0, 15, 30, 45],
    "p1" : [1, 4, 8],
    "p2" : [1, 2, 4, 8, 16],
    "V_rest" : [-30, -60, 0],
    "threshold" : [0, 30, 60],
    "sub" : [30, 60]
}

db_file = "tested_combinations.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the table if it doesn't exist.
cursor.execute("""
    CREATE TABLE IF NOT EXISTS combinations (
        combination_key TEXT PRIMARY KEY,
        combination_json TEXT
    )
""")
conn.commit()

def combination_to_key(comb):
    return "|".join(f"{k}:{comb[k]}" for k in sorted(comb))

def generate_random_combination(param_values):
    return {key: random.choice(vals) for key, vals in param_values.items()}

new_combinations = []


while len(new_combinations) < num_new_needed:
    print("*",len(new_combinations), num_new_needed)
    comb = generate_random_combination(values)
    comb_key = combination_to_key(comb)

    # check if the combination key already exists in the database
    cursor.execute("SELECT 1 FROM combinations WHERE combination_key = ?", (comb_key,))
    if not cursor.fetchone():
        command = ["python", "paper_impl_SCAMP_sweep.py"]
        for param, value in comb.items():
            command.extend([f"--{param}", str(value)])

        result = subprocess.run(command)

        if result.returncode >= 0:
            new_combinations.append(comb)
            comb_json = json.dumps(comb)
            cursor.execute("INSERT INTO combinations (combination_key, combination_json) VALUES (?, ?)",
                           (comb_key, comb_json))
            conn.commit()  # Commit after each new insertion.
