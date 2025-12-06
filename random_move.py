import sys
import os
from data import *
from multiprocessing import Process
import time

def make_random_move(inp):
    dt = Data(inp)
    dt.random_move()
    dt.WriteData()

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "data/benchmark_instances/random_instance_4_40_2.json"
    if "json" in inp:
        start = time.time()
        dt = Data(inp)
        # dt.find_center()
        end = time.time()
        # print(f"total time: {end-start}s")
        # dt.WriteData()
        dt.random_move()
    else:
        json_list = os.listdir(inp)
        # json_list.reverse()
        for i in range(len(json_list)):
            json_list[i] = os.path.join(inp, json_list[i])
        pool = Pool(10)
        pool.map(make_random_move, json_list)
        # for inp1 in json_list:
        #     if "json" not in inp1:
        #         continue
        #     dt = Data(os.path.join(inp,inp1))
        #     dt.random_move()
        #     dt.WriteData()

