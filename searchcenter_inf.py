from data import *
from multiprocessing import Process, Pool
import os

def read_dt(d):
    dt = Data(d)
    prev = dt.dist
    dt.log = False
    count = 0
    while count < prev:
        dt.random_new_center()
        if dt.dist < prev:
            print(f"[{dt.instance_uid}] Improved: {prev} -> {dt.dist}")
            dt.WriteData()
            prev = dt.dist
        count += 1

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "opt/rirs-500-20-5e21448d.solution.json"
    if "json" in inp:
        read_dt(inp)
    else:
        inp_list = os.listdir(inp)
        # json_list.reverse()
        json_list = []
        for inp1 in inp_list:
            if "json" not in inp1:
                continue
            json_list.append(os.path.join(inp,inp1))

            # if "-20-" in inp1:
            #     continue
            # rirs_list.append(os.path.join(inp,inp1))
            # dt = Data(os.path.join(inp,inp1))
        print(json_list)
        pool = Pool(60)
        pool.map(read_dt, json_list)
