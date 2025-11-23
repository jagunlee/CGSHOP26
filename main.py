import sys
import os
from data import *
import time

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "data/benchmark_instances/random_instance_132_80_20.json"
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
        for inp1 in json_list:
            if "json" not in inp1:
                continue
            dt = Data(os.path.join(inp,inp1))
            dt.find_center()
            dt.WriteData()
