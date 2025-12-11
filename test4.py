from matplotlib import pyplot as plt
from cgshop2026_pyutils.visualize import create_instance_plot
from cgshop2026_pyutils.geometry import FlippableTriangulation, draw_flips, Point, draw_edges, is_triangulation
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
import json

from th_data import *

path1 = './data/benchmark_instances/'
#input_file = 'random_instance_110_15_3'
input_file = 'random_instance_444_15_10'

with open(path1+input_file+'.json', "r") as f:
    file = json.load(f)
    instance = CGSHOP2026Instance(
            instance_uid = file["instance_uid"],
            points_x = file["points_x"],
            points_y = file["points_y"],
            triangulations=file["triangulations"]
            )
    triangulations=file["triangulations"]

with open('./solutions/'+input_file+'.solution.json', 'r') as f:
    root = json.load(f)
dt = Data(root["meta"]["input"])

flip = [[5, 11], [6, 8], [2, 9], [6, 7], [5, 2], [6, 8]]

tmp = deepcopy(dt.triangulations[0])
print("flip = ", flip)
for edge in flip:
    edge = sorted(edge)
    print(tuple(edge))
    t = tmp.dict[tuple(edge)]
    dt.print_triangle(t)
    #dt.isFlippable(tmp, edge)
