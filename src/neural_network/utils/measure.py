import time 
import cProfile, pstats
import tqdm

def profile_dataset(dataset, data_points=20):
    print("Profiling the code : time required to iterate over the whole dataset")
    start_time = time.time()
    for i in tqdm.tqdm(range(len(dataset))):
        _ = dataset[i]
    end_time = time.time()
    print("Time taken : ", end_time - start_time, " seconds for ", len(dataset), " items")
    
    pr = cProfile.Profile()
    pr.enable()
    for i in range(data_points):
        i = dataset[i]
        _ = i["frames"]
        _ = i["grids"]
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)    # top 10 slowest calls