from matplotlib import pyplot as plt
from cgshop2026_pyutils.geometry import (
    FlippableTriangulation,
    draw_flips,
    Point,
    expand_edges_by_convex_hull_edges,
    is_triangulation,
)
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
#from cgshop2026_pyutils.visualize import create_instance_plot
#from origin_visualize import create_instance_plot
from visualize import create_instance_plot
from cgshop2026_pyutils.io import read_instance, read_solution

import json
import os
import argparse
import hull_fast_data as fast

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.data
    dt = fast.FastData(input_file)


