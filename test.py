from matplotlib import pyplot as plt
from cgshop2026_pyutils.visualize import create_instance_plot
from cgshop2026_pyutils.geometry import FlippableTriangulation, draw_flips, Point, draw_edges, is_triangulation
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
import json

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


    #create_instance_plot(instance)
    #plt.savefig('/Users/hyeyun/Desktop/'+input_file+'.pdf')

    #edges = [(3, 7), (4, 12), (3, 13), (5, 10), (0, 2), (8, 9), (8, 12), (9, 11), (1, 6), (0, 8), (2, 5), (1, 9), (2, 11), (2, 8), (6, 8), (6, 14), (7, 13), (4, 8), (3, 6), (10, 11), (10, 14), (1, 11), (1, 8), (1, 14), (6, 13), (4, 7), (3, 14), (5, 11), (0, 9), (8, 13), (2, 9), (2, 12), (1, 10), (7, 8)]
    #edges = [(0, 2), (0, 8), (0, 9), (1, 4), (1, 5), (2, 4), (2, 7), (2, 9), (3, 6), (3, 9), (4, 8), (5, 6), (5, 7), (6, 9), (7, 8), (7, 9)]
    edges = [[0, 1], [0, 5], [0, 8], [0, 9], [1, 3], [1, 4], [1, 5], [1, 6], [2, 8], [2, 9], [3, 6], [3, 9], [5, 7], [6, 9], [7, 8]]
    points = [Point(x,y) for x, y in zip(instance.points_x, instance.points_y)]
    print(points)
    #draw_edges(points, edges, show_indices=True)
    #plt.show()
    is_triangulation(points, edges, verbose=True)


#for triang in instance.triangulations:
#    points = [Point(x,y) for x, y in zip(instance.points_x, instance.points_y)]
#    assert is_triangulation(points, triang, verbose=False), f"Triangulation {triang} is not valid for the given points."


#path2 = './hy_opt/'
#with open(path2+input_file+'.solution.json', "r") as f:
#    file = json.load(f)
#    solution = CGSHOP2026Solution(
#            instance_uid = file["instance_uid"],
#            flips=file["flips"]
#            )
#
##errors = check_for_errors(instance, solution)
##assert not errors, f"Errors found in solution: {errors}"
#
#for flip in solution.flips:
#    print(flip)


