from th_data import *
from multiprocessing import Process, Pool
import os
import argparse

def read_dt(d):
    dt = Data(d)
    prev = dt.dist
    count = 0
    while dt.dist < prev * 1.1:
        print("count", count)
        dt.random_new_center()
        count += 1
        if dt.dist < prev:
            print(f"[{dt.instance_uid}] Improved: {prev} -> {dt.dist}")
            dt.WriteData()
            prev = dt.dist


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    inp = args.data
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
