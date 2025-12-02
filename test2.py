from matplotlib import pyplot as plt
from cgshop2026_pyutils.geometry import FlippableTriangulation, draw_flips, expand_edges_by_convex_hull_edges, is_triangulation, draw_edges
from cgshop2026_pyutils.geometry import Point as PP
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
        print("hy:  center = ")
        print(C_edges)

        points = [PP(x,y) for x, y in zip(instance.points_x, instance.points_y)]
        nei_dict = dict()
        for i in range(N):
            nei_dict[i] = []

        #draw_edges(points, C_edges, show_indices=True)
        #plt.show()
        # Remove intersecting edges
        # C_edges의 엣지를 하나하나 추가하면서 intersect 하는 부분 지우기
        # !!! Degenerate case 지우기 ex) (6,7), (6, 14) 엣지 있는데 (7,14)가 두 엣지를 모두 포함하는 경우 (7,14)는 삭제되야 함
        valid_edge=[]
        test_edge=[]
        random.shuffle(C_edges)
        for e in C_edges:
            if len(valid_edge)>0:
                conflict = True
                print("e: ", e)
                print("valid_edge:", valid_edge)
                if e in valid_edge: continue
                for le in valid_edge:
                    if dt.intersect(e[0],e[1], le[0], le[1])==True:#no intersection
                    # !!! Degenerate case 지우기
                    # 1) (6, 7), (6, 14) 가 있을 때, (7, 14)가 추가되려고 할 때, (7,6,14)가 한 직선 위에 있는 경우
                        inter_nei = list(set(nei_dict[e[0]]) & set(nei_dict[e[1]])) #7과 14의 neigh 교집합
                        if len(inter_nei)>0:
                            p1 = dt.pts[e[0]] #7의 좌표
                            p2 = dt.pts[e[1]] #14의 좌표
                            for ine in inter_nei:
                                p3 = dt.pts[ine[0]] #6의 좌표
                                if (p2.y-p1.y)*(p3.x-p2.x) == (p3.y-p2.y)*(p2.x-p1.x):
                                    conflict=True
                                    break
                    ## 3) (7, 14)를 추가하려고 할 때, 6의 존재를 발견해서 (7, 14)대신 (6,7)과 (6, 14)가 추가되야 함 <- 재귀함수가 되어야 함..
                    #    p1 = dt.pts[e[0]] #7의 좌표
                    #    p2 = dt.pts[e[1]] #14의 좌표
                    #    min_x = min(p1.x, p2.x)
                    #    max_x = max(p1.x, p2.x)
                    #    min_y = min(p1.y, p2.y)
                    #    max_y = max(p1.y, p2.y)
                    #    for p3 in dt.pts:
                    #        if min_x < p3.x < max_x and min_y < p3.y < max_y:
                    #            #기울기 확인
                    #            if (p2.y-p1.y)*(p3.x-p2.x) == (p3.y-p2.y)*(p2.x-p1.x):
                    #                conflict=True
                    #                idx_p3 = (dt.pts).index(p3)
                    #                valid_edge.append([idx_pe,e[0]])
                    #                valid_edge.append([idx_pe,e[1]])
                    #                break
                    ## 2) (0,5)가 있을 때, (2, 13)이 추가되려고 할 때, (2, 0, 13)이 한 직선 위에 있는 경우
                    #    p1 = dt.pts[e[0]] #2
                    #    p2 = dt.pts[e[1]] #13
                    #    p3 = dt.pts[le[0]] #0
                    #    p4 = dt.pts[le[1]] #5
                    #    # (p
                if conflict == False:
                    valid_edge.append(e)
                    draw_edges(points, valid_edge, show_indices=True)
            else:
                valid_edge.append(e)
                nei_dict[e[0]].append(e[1])
                nei_dict[e[1]].append(e[0])

        C_edges = valid_edge
        print("edges = ")
        print(C_edges)
        is_triangulation(points, C_edges, verbose=True)
        if len(C_edges)!=E and is_triangulation(points, C_edges, verbose=False)==False:
            is_triangulation(points, C_edges, verbose=True)
            draw_edges(points, C_edges, show_indices=True)
            #plt.show()
            break

        #draw_edges(points, C_edges, show_indices=True)
        #plt.show()
        print("---------------------")
        #exit(0)
