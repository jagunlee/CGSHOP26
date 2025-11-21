import sys
import os
from data import *
import time
from multiprocessing import Process, Pool

def find_dt_center(inp):
    dt = Data(os.path.join(inp))
    dt.find_center_np()
    dt.WriteData()

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "data/benchmark_instances/random_instance_440_160_20.json"
    if "json" in inp:
        start = time.time()
        dt = Data(inp)
        dt.find_center()
        end = time.time()
        print(f"total time: {end-start}s")
        dt.WriteData()
        # dt.random_move()
    else:
        json_list = os.listdir(inp)
        rirs_list = []
        for inp1 in json_list:
            # if "json" not in inp1:
            #     continue
            if "rirs" in inp1:
                continue
            # if "-20-" in inp1:
            #     continue
            rirs_list.append(os.path.join(inp,inp1))
            # dt = Data(os.path.join(inp,inp1))
            # dt.find_center()
            # dt.WriteData()
        
        pool = Pool(processes=10)
        pool.map(find_dt_center, rirs_list)
