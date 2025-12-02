from matplotlib import pyplot as plt
from cgshop2026_pyutils.geometry import FlippableTriangulation, draw_flips, expand_edges_by_convex_hull_edges, is_triangulation, draw_edges
from cgshop2026_pyutils.geometry import Point as PP
from cgshop2026_pyutils.geometry import compute_triangles
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.visualize import create_instance_plot

from jg_data import *
import numpy as np



if __name__ == '__main__':
    inst_file = 'random_instance_444_15_10'
    path = './'
    input_file = 'transformer-output-decoded.txt'
    Tris_path = './data/benchmark_instances/'

    dt = Data(Tris_path + inst_file + '.json')
    N = len(dt.pts)

    #if 'decoded' in input_file:
    lines=[]
    with open(path+input_file,"r") as file:
        for line in file:
            lines.append(line.strip())
    with open(Tris_path+inst_file+'.json', "r") as f:
        file = json.load(f)
        instance = CGSHOP2026Instance(
                instance_uid = file["instance_uid"],
                points_x = file["points_x"],
                points_y = file["points_y"],
                triangulations=file["triangulations"]
                )

    E = len(instance.triangulations[0])
    # Read 'transformer-output-decoded.txt' and save it in C
    # Read Triangulations Ts
    num=1
    for obj in lines:
        print(num)
        num+=1
        adjmat = np.zeros((N,N))
        index=0
        C_edges=[]
        for i in range(N-1):
            for j in range(i+1,N):
                while obj[index] ==',':
                    index+=1
                adjmat[i,j] = int(obj[index])
                adjmat[j,i] = adjmat[i,j]
                #index+=1
                if adjmat[i,j] ==1:
                    C_edges.append([i,j])

        points = [PP(x,y) for x, y in zip(instance.points_x, instance.points_y)]

        #draw_edges(points, C_edges, show_indices=True)
        #plt.show()
        # Remove intersecting edges
        # C_edges의 엣지를 하나하나 추가하면서 intersect 하는 부분 지우기
        random.shuffle(C_edges)
        valid_edge=[]
        for e in C_edges:
            conflict = True
            if len(valid_edge)>0:
                for le in valid_edge:
                    if dt.intersect(e[0],e[1], le[0], le[1])==True:#no intersection
                        conflict = False
                    else:
                        conflict = True
                        break
                if conflict == False:
                    valid_edge.append(e)
            else:
                valid_edge.append(e)
        C_edges = valid_edge
        if len(C_edges) != E:
            print(C_edges)
            print("---------------------")
            continue
        result = compute_triangles(points, C_edges)
        print(result)
        draw_edges(points, C_edges, show_indices=True)
        plt.show()
        print("---------------------")
        exit(0)
