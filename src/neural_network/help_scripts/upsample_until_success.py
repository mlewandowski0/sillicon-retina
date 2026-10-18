# supervisor_resume.py
import os, json, subprocess, sys, time
from pathlib import Path

OUT = "src/neural_network/data/saliency_dataset_upsampled"
IN = "src/neural_network/data/saliency_dataset_truncated"
# forwarded to the worker; unset = optical-flow upsampling, e.g. UPSAMPLE_FPS=200 = fixed-rate upsampling
FIXED_FPS = int(os.environ.get("UPSAMPLE_FPS", "0"))

def is_done(i: int) -> bool:
    folder = os.path.join(OUT, f"video_{i:05d}")
    if not os.path.isdir(folder): return False
    if not os.path.exists(os.path.join(folder, "params.json")): return False
    video = f"video_upsampled_fps_eq_{FIXED_FPS}.mp4" if FIXED_FPS else "video_upsampled_using_optical_flow.mp4"
    return os.path.exists(os.path.join(folder, video))

def find_next_start():
    n = len([d for d in os.listdir(IN) if os.path.isdir(os.path.join(IN, d))])
    for i in range(n):
        if not is_done(i):
            return i, n
    return n, n

def run_pass(start_idx: int) -> int:
    env = os.environ.copy()
    env["START_IDX"] = str(start_idx)
    # call the worker module that configures TF GPU paths itself
    return subprocess.call([sys.executable, "-m", "src.neural_network.help_scripts.upsample_worker_gpu"], env=env)

def main():
    try:
        while True:
            i, n = find_next_start()
            if i >= n:
                print("All done.")
                return
            print(f"Resuming at index {i}/{n}")
            rc = run_pass(i)
            if rc == 0:
                # worker finished (maybe processed many items); loop to find next
                continue
            else:
                # killed/crashed — pause and retry; worker is idempotent
                print(f"[warn] worker exited code {rc}; retrying in 5s…")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user; exiting.")
        sys.exit(130)

if __name__ == "__main__":
    main()
