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

print(len(tmp.dict.keys()), len(tmp.edges))

count=1
for E in tmp.edges:
    v1, v2 = E
    E_r = (v2,v1)

    if E in tmp.dict:
        face1 = tmp.dict[E]
        if E_r in tmp.dict:
            face2 = tmp.dict[E_r]
            v3 = int(list(set(face1.pts)-set(face2.pts))[0])
            v4 = int(list(set(face2.pts)-set(face1.pts))[0])
            print(count, ": two faces! ", E, " in :", face1.pts, face2.pts, "(v3, v4) = (", v3, v4, ")")
        else:
            print(count, ": hull ----- ", E)
    else:
        if E_r in tmp.dict:
            print(count, ": hull ----- r", E_r)
    count+=1

#count=1
#for E in tmp.dict.keys():
#    # which triangle share an edge E
#    face1 = tmp.dict[E]
#    found=False
#    for t in tmp.triangles- {face1}:
#        if len(set(t.pts) - set(E))==1:
#            print(count, ": ------------", E, "------------- ", t.pts)
#            found=True
#            break
#    if found==False:
#        print(count, ": hull = ", E)
#    count+=1

print()
print("flip = ", flip)
for edge in flip:
    edge = sorted(edge)
    print()
    print(tuple(edge), " is flippable?", end=" ")
    if dt.isFlippable(tmp, tuple(edge)):
        print("Flipped!")
    else:
        print("No!!")
