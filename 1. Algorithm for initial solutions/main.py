import sys
import os
from data import *
import time
import argparse
import natsort

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=0)
    parser.add_argument('-l', type=str)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    instances = []
    userInput = False

    parent_directory = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    for file in os.listdir(parent_directory + '/data/benchmark_instances'):
        if 'pdf' == file[len(file)-3:]:
            pass
        else:
            instances.append('../data/benchmark_instances/'+file)

    instances = natsort.natsorted(instances)
    instance = instances[int(args.n)]

    log = open('log/'+args.l, 'w')
    sys.stdout = log

    #### 1. Read instance ####
    start = time.time()
    D = Data(instance)
    end = time.time()
    print(f"Read instance:{instance}")
    print(f"read takes {end - start:.5f} sec")

    #### 2. Generate Initial solution ####
    centerT = D.findCenterGlobal()
    D.WriteData()
    end = time.time()
    print(f"initial solution takes {end - start:.5f} sec")
