import json, copy, time, random
import fast_data as fast
import parallel_data as para
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

