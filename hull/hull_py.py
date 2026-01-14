import json, copy, time, random
import hull_fast_data as hull
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
    dt = hull.FastData(input_file)

