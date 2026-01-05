import json, copy, time, random
import fast_data as fast
import rirs_data as th2
from multiprocessing import Pool
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    Tris_path = '~/Experiment/PFD/hyeyun_git/data/benchmark_instances/'


    # read initial solution
    input_file = args.data
    dt = th2.Data(input_file)
    #dt = fast.FastData(input_file)
    #dt.ReadData()
    #best_dist = dt.dist
    #dt.inst_info()
    #print(f"\nNo flipped, dist: {dt.dist}")
    #pfd = [len(f) for f in dt.pFlips]
    #dt.pfd_distribution(pfd)

