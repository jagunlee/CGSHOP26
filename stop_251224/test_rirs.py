#####
# 4cgshop.py is generating better C.
####

from th2_data import *
import os

import numpy as np
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.zip.zip_writer import ZipWriter

import torch


if __name__ == '__main__':

    Tris_path = './data/benchmark_instances/'
    inst_file = 'rirs-12500-200-e8ac0d06'
    print(inst_file)
    with open(Tris_path+ inst_file +'.json', 'r') as f:
        root = json.load(f)
    f.close()
    numpt = len(root["points_x"])
    numtris = len(root["triangulations"])
    numedges = len(root["triangulations"][0])
    print(f"number of vertices = {numpt}")
    print(f"number of triangulations = {numtris}")
    print(f"number of edges = {numedges}")

    Tris = [set() for _ in range(numtris)]
    for i, tri in enumerate(root["triangulations"]):
        Tris[i] = set(list(map(tuple,tri)))

    intersect_edges=Tris[0]
    for i in range(1, numtris):
        intersect_edges = intersect_edges.intersection(Tris[i])
    numinter = len(intersect_edges)
    print(f"number of insertect edges / total edges = {numinter}/{numedges}")

    intersect_edges = [[] for _ in range(numtris)]
    max_intersect=0
    min_intersect=numedges
    for i in range(numtris-1):
        for j in range(i+1,numtris):
            edges_ = Tris[i].intersection(Tris[j])
            intersect_edges[i].append(len(edges_))
        max_edges = max(intersect_edges[i])
        min_edges = min(intersect_edges[i])
        if max_intersect < max_edges: max_intersect = max_edges
        if min_intersect > min_edges: min_intersect = min_edges
        idx_max = intersect_edges[i].index(max_edges)
        #print(f"number of max intersection edges with T{i}: {max_edges}, T{idx_max}")
    print(f"max_intersect {max_intersect}, min_intersect {min_intersect}")


    #with open(Tris_path+ inst_file +'.json', 'r') as f:
    #    flips = root["flips"]
    #prev_Dsum = sum([len(f) for f in flips])



