# supervisor_resume.py
import os, json, subprocess, sys, time, textwrap

OUT = "src/neural_network/data/saliency_dataset_upsampled"

def is_done(i: int) -> bool:
    folder = os.path.join(OUT, f"video_{i:05d}")
    if not os.path.isdir(folder): return False
    if not os.path.exists(os.path.join(folder, "params.json")): return False
    return os.path.exists(os.path.join(folder, "video_upsampled_using_optical_flow.mp4"))

def find_next_start():
    meta = json.load(open(os.path.join(OUT, "metadata.json")))
    n = int(meta["number_of_videos"])
    for i in range(n):
        if not is_done(i):
            return i, n
    return n, n

def run_pass(start_idx: int) -> int:
    env = os.environ.copy()
    env["START_IDX"] = str(start_idx)
    code = textwrap.dedent(r"""
        import os
        from src.neural_network.dataloaders.VideoDatasetUpsampling import (
            VideoDatasetUpsampling, VideoDatasetUpsamplingCacheToFolder
        )
        from src.neural_network.dataloaders.VideoDatasetFromPairs import VideoDatasetFromVideoPairGlobal

        OUT = "src/neural_network/data/saliency_dataset_upsampled"
        start_index = int(os.environ["START_IDX"])

        video_val = VideoDatasetFromVideoPairGlobal(
            dirpath=os.path.join("src","neural_network","data","saliency_dataset_truncated"),
            split=None,
            global_fps=30,
            input_video="input.AVI",
            output_video="maps.mp4",
            suffix=None
        )
        upsampling_ds_val = VideoDatasetUpsampling(
            video_val,
            model_path=os.path.join("models","film_net","Style","saved_model"),
            upsample_to_events=True,
            new_framerate=False,
            output_name="maps",
            debug=False
        )

        # override=False → skip already done items
        VideoDatasetUpsamplingCacheToFolder(
            upsamplingDataset=upsampling_ds_val,
            output_path=OUT,
            start_index=start_index,
            override=False, save_optical_flow = False
        )
    """).lstrip("\n")
    return subprocess.call([sys.executable, "-c", code], env=env)

def main():
    meta_path = os.path.join(OUT, "metadata.json")
    if not os.path.exists(meta_path):
        print("metadata.json missing; run your upsampling script once to create it.")
        sys.exit(1)

    try:
        while True:
            i, n = find_next_start()
            if i >= n:
                print("All done.")
                return
            print(f"Resuming at index {i}/{n}")
            rc = run_pass(i)
            if rc == 0:
                continue
            else:
                print(f"[warn] pass crashed with code {rc}; retrying in 5s…")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user; exiting.")
        sys.exit(130)

if __name__ == "__main__":
    main()
