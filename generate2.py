import json, copy, time, random
import th2_data as th2
from multiprocessing import Pool
import os
import argparse


# Do flip (num_flips) times
def perturb_center1(flip_rate:float, dt, C_edges, best_dist):
    initial_center = copy.deepcopy(dt.center)
    num_flips = int(flip_rate*0.01*len(C_edges))
    flipped=0
    for e in C_edges.keys():
        if dt.flippable(initial_center, e):
            initial_center.flip(e)
            flipped+=1
            if flipped == num_flips: break

    if flipped==0: return flipped, [], []
    # compute pfd sum
    all_pFlips=[]*len(dt.triangulations)
    for tri in dt.triangulations:
        pFlips=[]
        pFlips_paired1 = dt.parallel_flip_path2(tri, initial_center)
        pFlips_paired2 = dt.parallel_flip_path_rev2(initial_center, tri)
        pFlips_paired=[]
        if len(pFlips_paired1) < len(pFlips_paired2):
            pFlips_paired=pFlips_paired1
        else:
            pFlips_paired=pFlips_paired2
            pFlips_paired.reverse()
        for round_ in pFlips_paired:
            round_temp = [list(oneFlip) for oneFlip in round_]
            pFlips.append(round_temp)
        all_pFlips.append(pFlips)


    pfd = [len(f) for f in all_pFlips]
    if sum(pfd) <= best_dist:
        return flipped, pfd, all_pFlips
    return flipped, pfd, []


# Do flip edges with (flip_value) weight value
def perturb_center2(flip_value, dt, C_edges, best_dist):
    initial_center = copy.deepcopy(dt.center)
    flipped=0
    for e in C_edges.keys():
        if C_edges[e]==flip_value:
            if dt.flippable(initial_center, e):
                initial_center.flip(e)
                flipped+=1
        if C_edges[e]>flip_value: break

    if flipped==0: return flipped, [], []
    # compute pfd sum
    all_pFlips=[]*len(dt.triangulations)
    for tri in dt.triangulations:
        pFlips=[]
        pFlips_paired1 = dt.parallel_flip_path2(tri, initial_center)
        pFlips_paired2 = dt.parallel_flip_path_rev2(initial_center, tri)
        pFlips_paired=[]
        if len(pFlips_paired1) < len(pFlips_paired2):
            pFlips_paired=pFlips_paired1
        else:
            pFlips_paired=pFlips_paired2
            pFlips_paired.reverse()
        for round_ in pFlips_paired:
            round_temp = [list(oneFlip) for oneFlip in round_]
            pFlips.append(round_temp)
        all_pFlips.append(pFlips)


    pfd = [len(f) for f in all_pFlips]
    if sum(pfd) <= best_dist:
        return flipped, pfd, all_pFlips
    return flipped, pfd, []


# Do randomly flip edges with (flip_value) weight value
def C_update_perturb_center3(flip_value, dt, C_edges, best_dist):
    initial_center = copy.deepcopy(dt.center)
    flipped=0
    edge_list=[]
    for e in C_edges.keys():
        if C_edges[e]==flip_value:
            edge_list.append(e)
        if C_edges[e]>flip_value: break


    random.shuffle(edge_list)
    for e in edge_list:
        if dt.flippable(initial_center, e):
            initial_center.flip(e)
            flipped+=1

    if flipped==0: return flipped, [], []
    # compute pfd sum
    all_pFlips=[]*len(dt.triangulations)
    for tri in dt.triangulations:
        pFlips=[]
        pFlips_paired1 = dt.parallel_flip_path2(tri, initial_center)
        pFlips_paired2 = dt.parallel_flip_path_rev2(initial_center, tri)
        pFlips_paired=[]
        if len(pFlips_paired1) < len(pFlips_paired2):
            pFlips_paired=pFlips_paired1
        else:
            pFlips_paired=pFlips_paired2
        for round_ in pFlips_paired:
            round_temp = [list(oneFlip) for oneFlip in round_]
            pFlips.append(round_temp)
        all_pFlips.append(pFlips)


    pfd = [len(f) for f in all_pFlips]
    #C update!!!
    dt.center = initial_center
    if sum(pfd) <= best_dist:
        return flipped, pfd, all_pFlips
    return flipped, pfd, []





def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser


def Edge_weight1(dt):
    C_edges = dict.fromkeys(dt.center.edges,0)
    for E in C_edges.keys():
        for T in dt.triangulations:
            if E in T.edges: C_edges[E]+=1
    # ascending order
    C_edges = dict(sorted(C_edges.items(), key=lambda item: item[1]))
    #C_edges = dict(sorted(C_edges.items(), key=lambda item: item[1], reverse=True))
    return C_edges

def Edge_weight2(dt):
    list_E = list(dt.center.edges)
    random.shuffle(list_E)
    C_edges = dict.fromkeys(list_E,0)
    return C_edges


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
    dt = th2.Data(input_file)
    best_dist = dt.dist
    dt.inst_info()
    print(f"\nNo flipped, dist: {dt.dist}")
    pfd = [len(f) for f in dt.pFlips]
    dt.pfd_distribution(pfd)


    #Tij_diff(dt)

    # C edges weight
    C_edges = Edge_weight1(dt)
    #C_edges = Edge_weight2(dt)

    # perturb_center1
    #for rate in range(5, 110, 5):
    #    flipped, pfd, pFlips = perturb_center1(rate, dt, C_edges, best_dist)
    #    new_dist = sum(pfd)
    #    print(f"{flipped}({flipped*100/len(C_edges):.2f})% flipped, new_dist: {new_dist}")
    #    dt.pfd_distribution(pfd)
    #    if new_dist < best_dist:
    #        dt.pFlips = pFlips
    #        dt.WriteData()

    best_flips = dt.pFlips
    # perturb_center3
    for value in range(len(dt.triangulations)):
        flipped, pfd, pFlips = C_update_perturb_center3(value, dt, C_edges, best_dist)
        #C_edges = Edge_weight1(dt)
        if flipped==0: continue
        new_dist = sum(pfd)
        print(f"{flipped}({flipped*100/len(C_edges):.2f})% flipped, only edge with {value}-weight flipped,  new_dist: {new_dist}")
        dt.pfd_distribution(pfd)
        if flipped>0 and new_dist <= best_dist:
            if best_flips != pFlips:
                dt.pFlips = pFlips
                dt.WriteData()
                best_fips = pFlips
            else: print("Same flips, so no writedata")




    ##parallel
    #with Pool(processes=4) as pool:
    #    R_pfds = pool.starmap(perturb_center, [(rate, dt, C_edges, best_dist) for rate in range(10, 110, 10)])
    #for r_pfd in R_pfds:
    #    flipped= r_pfd[0]
    #    pfd = r_pfd[1]
    #    new_dist = sum(pfd)
    #    print(f"{flipped}({flipped*100/len(C_edges):.2f})% flipped, new_dist: {new_dist}")
    #    dt.pfd_distribution(pfd)
    #    if new_dist < best_dist:
    #        dt.pFlips = r_pfd[2]
    #        dt.WriteData()
