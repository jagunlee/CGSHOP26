import fast_data as fast
from multiprocessing import Pool
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser


def Edge_weight1(dt):
    C_edges = dict.fromkeys(dt.center.edges,0)
    for E in C_edges.keys():
        for T in dt.triangulations[:-1]:
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

def Edge_weight3(dt): # crossing 수
    C_edges = dict.fromkeys(dt.center.edges,0)
    center = dt.triangulations[-1].fast_copy()
    for E in C_edges.keys():
        if dt.flippable(center, E):
            for T in dt.triangulations[:-1]:
                score = dt.flip_score(center, T, E, 1)
                if score[0]>0:
                    C_edges[E] += score[0]

    C_edges = dict(sorted(C_edges.items(), key=lambda item: item[1]))
    return C_edges


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    Tris_path = './data/benchmark_instances/'


    # read initial solution
    input_file = args.data
    dt = fast.FastData(input_file)
    best_dist = dt.dist
    dt.inst_info()
    print(f"\nNo flipped, initial dist: {dt.dist}")
    pfd = [len(f) for f in dt.pFlips]
    dt.pfd_distribution(pfd)


    # C edges weight
    C_edges = Edge_weight3(dt)

    best_flips = dt.pFlips
    past = pfd
    #for value in range(len(dt.triangulations)-1):
    values = set(C_edges.values())
    for value in values:
        print(value)
        flipped, pfd, pFlips = dt.perturb_center3(value, C_edges, best_dist)
        if flipped==0: continue
        new_dist = sum(pfd)
        print(f"{flipped}({flipped*100/len(C_edges):.2f})% flipped, only edge with {value}-weight flipped,  new_dist: {new_dist}")
        if past == pfd: continue
        dt.pfd_distribution(pfd)
        if flipped>0 and new_dist < best_dist:
            #if best_flips != pFlips:
            dt.pFlips = pFlips
            dt.WriteData()
            best_fips = pFlips
            #else: print("Same flips, so no writedata")




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
