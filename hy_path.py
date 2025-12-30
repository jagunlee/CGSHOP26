import json, copy, time, random
import tmp_data as tmp
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

    Tris_path = './data/benchmark_instances/'


    # read initial solution
    input_file = args.data
    dt = tmp.Data(input_file)
    best_dist = dt.dist
    dt.inst_info()
    print(f"\nNo flipped, dist: {dt.dist}")
    pfd = [len(f) for f in dt.pFlips]
    dt.pfd_distribution(pfd)

