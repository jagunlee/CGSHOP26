#####
# 4cgshop.py is generating better C.
####

#from jg_data import *
from th2_data import *
import os

import numpy as np
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.zip.zip_writer import ZipWriter

import torch
from makemoretokens import CharDataset, ModelConfig, Transformer, InfiniteDataLoader, generate, evaluate


if __name__ == '__main__':

    Tris_path = './data/benchmark_instances/'
    #inst_file = 'random_instance_110_15_3'
    #inst_file = 'random_instance_93_40_10'
    inst_file = 'random_instance_444_15_10'
    #inst_file = 'random_instance_552_320_20'
    #inst_file = 'random_instance_826_320_20'
    #inst_file = 'rirs-1500-50-49040875'
    #inst_file = 'rirs-1500-20-abcb179b'

    for j in range(1, 5):
        for i in range(1, 360):
            filename = inst_file + f".solution_{j}_{i}.json"
            with open('./same_inst_solutions/'+ filename, 'r') as f:
                root = json.load(f)
            dt = Data('data/benchmark_instances/'+inst_file+'.json') # Data class from th2_data.py

            flips = root["flips"]
            prev_Dsum = sum([len(f) for f in flips])

            firstT = copy.deepcopy(dt.triangulations[0])
            for pll_flip in flips[0]:
                for flip in pll_flip:
                    firstT.flip((flip[0], flip[1]))
                    #dt.flipDiagonal(firstT, [flip])
            centerT = copy.deepcopy(firstT)
            dt.computeDistanceSum(centerT)

            curr_Dsum = sum([len(dt.pFlips[i]) for i in range(len(dt.triangulations))])
            if prev_Dsum == 25:
                print(f"{j}_{i} solution: (prev_dsum) --> (curr_dsum): {prev_Dsum}, {curr_Dsum}")
            elif curr_Dsum < 25:
                print(f"{j}_{i} solution: (prev_dsum) --> (curr_dsum): {prev_Dsum}, {curr_Dsum}")


