import json, copy, time, random
import rirs_data as rirs
from multiprocessing import Pool
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser



def Tij_pfd(dt):
    # T_i, T_j pfd
    print("           ", end=' ')
    for i in range(len(dt.triangulations)):
        print(f"{i:02d}", end=' ')
    print(end='\n')
    for i in range(len(dt.triangulations)):
        print(f"from T_{i:02d} ", end=' ')
        for j in range(len(dt.triangulations)):
            if j<i: print("__", end=' ')
            else:
                pFlips_paired1 = dt.parallel_flip_path2(dt.triangulations[i], dt.triangulations[j])
                pFlips_paired2 = dt.parallel_flip_path2(dt.triangulations[j], dt.triangulations[i])
                print(f"{min(len(pFlips_paired1), len(pFlips_paired2)):02d}", end=' ')
        print(end='\n')

def Tij_diff(dt):
    # T_i, T_j pfd
    print("           ", end=' ')
    for i in range(len(dt.triangulations)):
        print(f"{i:02d}", end=' ')
    print(end='\n')
    for i in range(len(dt.triangulations)):
        print(f"from T_{i:02d} ", end=' ')
        for j in range(len(dt.triangulations)):
            if i!=j:
                diff_edges = len(dt.triangulations[i].edges - dt.triangulations[j].edges)
                print(f"{diff_edges:02d}", end=' ')
        print(end='\n')




if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    Tris_path = './data/benchmark_instances/'


    # read initial solution
    input_file = args.data
    dt = rirs.Data(input_file)
    best_dist = dt.dist
    dt.inst_info()
    print(f"\nNo flipped, dist: {dt.dist}")
    pfd = [len(f) for f in dt.pFlips]
    dt.pfd_distribution(pfd)

