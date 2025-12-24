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
    inst_file = 'random_instance_982_320_10'
    inst_file = 'random_instance_110_15_3'
    print(inst_file)
    with open(Tris_path+ inst_file +'.json', 'r') as f:
        root = json.load(f)
    print(f"number of vertices = {len(root["points_x"])}")
    print(f"number of triangulations = {len(root["triangulations"])}")
    f.close()

    #with open(Tris_path+ inst_file +'.json', 'r') as f:
    #    flips = root["flips"]
    #prev_Dsum = sum([len(f) for f in flips])



