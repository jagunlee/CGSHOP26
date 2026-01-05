import json
import sys
import random
from rirs_Point import Point
from rirs_Triangulation import Triangle, Triangulation
import copy
# import cv2
import time
import os
import pandas as pd
import datetime
import numpy as np
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors

sys.setrecursionlimit(1000000)
SEARCH_DEPTH = 1
PAR_LEN = 2.5
PAR_CROSS = 1


class Data:
    def __init__(self, inp):
        self.input = inp
        # print('self.input:', self.input)
        self.triangulations = []
        self.ReadData()
        self.instance_uid = (inp.split('/')[-1]).split('.')[0]

    def ReadData(self):
        if "solution" not in self.input:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            # print(root)
            self.instance_name = root["instance_uid"]
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = []
            for i in range(len(self.pts_y)):
                self.pts.append(Point(self.pts_x[i], self.pts_y[i]))
            print(len(root["triangulations"]), ": ", end = " ")
            for t in root["triangulations"]:
                self.triangulations.append(self.make_triangulation(t))
                print(len(self.triangulations), end=", ")
            print()

            self.pFlips = [None] * len(self.triangulations)
        else:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.instance_name = root["instance_uid"]
            self.pFlips = root["flips"]
            self.dist = sum([len(x) for x in self.pFlips])
            org_input = '/Users/hyeyun/Experiment/PFD/hyeyun_git/data/benchmark_instances/'+self.instance_name+'.json'

            self.input = org_input
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = []
            for i in range(len(self.pts_y)):
                self.pts.append(Point(self.pts_x[i], self.pts_y[i]))
            for t in root["triangulations"]:
                self.triangulations.append(self.make_triangulation(t))
            print(f"num_edges = {len(root["triangulations"][0])}")

            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = copy.deepcopy(self.triangulations[min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip((flp[0], flp[1]))
            new_dist = 0
            new_pfp =[None]*len(self.triangulations)
            for i in range(len(self.triangulations)):
                new_pfp1=self.parallel_flip_path2(self.triangulations[i], self.center)
                new_pfp2=self.parallel_flip_path_rev2(self.center, self.triangulations[i])
                if len(new_pfp1) < len(new_pfp2):
                    new_pfp[i] = new_pfp1
                else:
                    new_pfp[i] = new_pfp2
                new_dist+=len(new_pfp[i])
            print(f"Original dist: {self.dist}, New dist: {new_dist}")
            if new_dist<self.dist:
                self.dist = new_dist
                self.pFlips = new_pfp
                self.WriteData()
                print("dist and pFlips are updated!\n")
    #hy
    def inst_info(self):
        print(f"_________ {self.instance_uid} info ___________")
        print(f"# of points: {len(self.pts_x)}")
        print(f"# of triangulations: {len(self.triangulations)}")
        print(f"# of edges in T: {len(self.triangulations[0].edges)}")
        pfd = [len(x) for x in self.pFlips]
        max_pfd = max(pfd)
        max_pfd_tid = pfd.index(max_pfd)
        min_pfd = min(pfd)
        min_pfd_tid = pfd.index(min_pfd)
        print(f"max pfd T_{max_pfd_tid}: {max_pfd}")
        print(f"min pfd T_{min_pfd_tid}: {min_pfd}")

    #hy
    def pfd_distribution(self,pfd):
        pfd_set = sorted(set(pfd))
        print("______ pfd distribution ________")
        for i, dist in enumerate(pfd_set):
            print(f"pfd {dist:03d} ({pfd.count(dist):03d}): ", end=' ')
            for _ in range(pfd.count(dist)):
                print("*", end='')
            all_indexs =[j for j in range(len(pfd)) if dist==pfd[j]]
            print(" (T:", end='')
            for idx in all_indexs:
                  print(idx, end=',')
            print(end=')\n')
        print()


    def make_triangulation(self, t: Triangulation):# t is not Triangulation...
        tri = Triangulation()
        tri.adj = [-1] * len(self.pts)
        graph = [[] for _ in range(len(self.pts))]
        for u, v in t:
            graph[u].append(v)
            graph[v].append(u)
            tri.adj[u] = v
            tri.adj[v] = u
            tri.edges.add((min(u, v), max(u, v)))

        for i in range(len(self.pts)):
            for j in range(len(graph[i])):
                v1 = graph[i][j]
                if v1 < i:
                    continue
                for k in range(j + 1, len(graph[i])):
                    v2 = graph[i][k]
                    if v2 < i:
                        continue
                    if v1 in graph[v2]:
                        if turn(self.pts[i], self.pts[v1], self.pts[v2]) > 0:
                            t = Triangle(i, v1, v2)
                        else:
                            t = Triangle(i, v2, v1)

                        flag = False
                        for l in range(len(graph[i])):
                            if l == j or l == k:
                                continue
                            v3 = graph[i][l]
                            smallflag = True
                            for m in range(3):
                                smallflag &= turn(self.pts[t.pt(m)], self.pts[t.pt(m + 1)], self.pts[v3]) >= 0
                            if smallflag:
                                flag = True
                                break
                        if flag:
                            del t
                            continue

                        for l in range(3):
                            tri.dict[t.pt(l), t.pt(l + 1)] = t
                            tt = tri.find_triangle(t.pt(l + 1), t.pt(l))
                            if tt:
                                tt.neis[tt.get_ind(t.pt(l + 1))] = t
                                t.neis[l] = tt
                        tri.triangles.add(t)
        return tri

    def find_triangle_containing(self, tri: Triangulation, con: tuple):
        q1, q2 = con
        if tri.find_triangle(q1, q2) or tri.find_triangle(q2, q1):
            return None
        p = tri.adj[q1]
        assert((min(p, q1), max(p, q1)) in tri.edges)
        t = tri.find_triangle(q1, p)
        if not t:
            t = tri.find_triangle(p, q1)
        # print(p, q1)
        assert(t)
        r1 = self.pts[q1]
        r4 = self.pts[q2]
        while True:
            i = t.get_ind(q1)
            r2 = self.pts[t.pt(i + 1)]
            r3 = self.pts[t.pt(i + 2)]
            if turn(r1, r2, r4) < 0:
                t = t.neis[i]
            elif turn(r1, r3, r4) > 0:
                t = t.nei(i+2)
            else:
                return t

    def count_cross(self, tri: Triangulation, con: tuple):
        t = self.find_triangle_containing(tri, con)
        if not t:
            return 0
        q1, q2 = con
        i = t.get_ind(q1)
        tt = t.nei(i + 1)
        j = tt.get_ind(t.pt(i + 1))
        cnt = 1
        while tt.pt(j + 1) != q2:
            cnt += 1
            t, i = tt, j
            if turn(self.pts[q1], self.pts[q2], self.pts[t.pt(i + 1)]) < 0:
                tt = t.nei(i + 1)
                j = tt.get_ind(t.pt(i + 1))
            else:
                tt = t.neis[i]
                j = tt.get_ind(t.pts[i])
        return cnt

    def parallel_flip_path(self, tri1:Triangulation, tri2:Triangulation):
        tri = copy.deepcopy(tri1)
        pfp = []
        while True:
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri2, e, SEARCH_DEPTH)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                tri.flip(e)
            pfp.append(flips)
        assert(tri.edges == tri2.edges)
        return pfp

    #jg
    def parallel_flip_path2(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri1.fast_copy()
        pfp = []
        while True:
            prev_flip = []
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    if e in prev_flip:
                        continue
                    score = self.flip_score(tri, tri2, e, 1)#hy 0-> 1
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                e1 = tri.flip(e)
                prev_flip.append(e1)
            pfp.append(flips)
        assert(tri.edges == tri2.edges)
        return pfp

    #hy
    def parallel_flip_path_rev2(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri1.fast_copy()
        rev_pfp=[]
        pfp=[]
        while True:
            prev_flip = []
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if e in prev_flip:
                    continue
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri2, e, 1)#hy 0-> 1
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips = []
            marked = set()
            for (p1, p3), _ in cand:
                t1 = tri.find_triangle(p1, p3)
                t2 = tri.find_triangle(p3, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips.append((p1, p3))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                e1 = tri.flip(e)
                prev_flip.append(e1)
            rev_pfp.append(prev_flip)
            pfp.append(flips)
        assert(tri.edges == tri2.edges)
        rev_pfp.reverse()
        return rev_pfp


    def flippable(self, tri:Triangulation, e:tuple):
        p1, p3 = e
        t1 = tri.find_triangle(p1, p3)
        t2 = tri.find_triangle(p3, p1)
        if (not t1) or (not t2):
            return False
        i = t1.get_ind(p3)
        p4 = t1.pt(i + 1)
        j = t2.get_ind(p1)
        p2 = t2.pt(j + 1)
        return (turn(self.pts[p2], self.pts[p3], self.pts[p4]) > 0) and (turn(self.pts[p2], self.pts[p1], self.pts[p4]) < 0)

    def flip_score(self, tri:Triangulation, tri_dest:Triangulation, e:tuple, depth:int):
        p1, p3 = e
        t1 = tri.find_triangle(p1, p3)
        t2 = tri.find_triangle(p3, p1)
        i = t1.get_ind(p3)
        p4 = t1.pt(i + 1)
        j = t2.get_ind(p1)
        p2 = t2.pt(j + 1)
        ori_cross = self.count_cross(tri_dest, e)
        new_cross = self.count_cross(tri_dest, (p2, p4))
        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        if depth == 1:
            return m_score
        tri.flip(e)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                nsc = self.flip_score(tri, tri_dest, pe, 1)#hy
                m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))

        #print("try flipping(2)",(p2, p4))
        tri.flip((p2, p4))
        return m_score

    def print_triangle(self, t: Triangle):
        print("Triangle :", end="")
        print(t)
        print(t.pts[0], ":", self.pts[t.pts[0]])
        print(t.pts[1], ":", self.pts[t.pts[1]])
        print(t.pts[2], ":", self.pts[t.pts[2]])
        print(t.neis[0])
        print(t.neis[1])
        print(t.neis[2])

    # w1:w2 내분점에서 가장 가까운 중간 triangulation을 반환
    def internal_division(self, T1: Triangulation, w1: int, T2: Triangulation, w2: int):

        T1copy = copy.deepcopy(T1)

        # parallel version
        pfp = self.parallel_flip_path(T1, T2)
        midVal = int(len(pfp) * (w2 / (w1 + w2)))
        #print('len(pfp):', len(pfp), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)

        # use_pfp: 사용되는 parallel flip들 모음
        use_pfp = pfp[:midVal]

        for flips in use_pfp:
            for flip in flips:
                T1copy.flip(flip)
        return T1copy

    def findCenter(self):

        start = time.time()

        wc = [(t, 1) for t in self.triangulations]
        while len(wc) > 1:
            print(len(wc), "triangles left")
            random.shuffle(wc)
            t1, w1 = wc.pop()
            t2, w2 = wc.pop()
            tc = self.internal_division(t1, w1, t2, w2)

            end = time.time()
            print('time:', f"{end - start:.5f} sec")
            wc.append((tc, w1 + w2))

        centerT = wc[0][0]

        with open("centers/" + self.instance_uid + ".json", "w", encoding="utf-8") as f:
            json.dump(list(centerT.edges), f, indent='\t')

        return centerT

    def findCenterGlobal(self):
        mtriangulations = [copy.deepcopy(t) for t in self.triangulations]
        pfps =[[] for _ in self.triangulations]
        while True:
            mscore = 0
            for i in len(mtriangulations):
                ncand = []
                nscore = 0
                tri = mtriangulations[i]
                edges = list(tri.edges)
                for e in edges:
                    escore = 0
                    for j in len(mtriangulations):
                        if i==j: continue
                        score = self.flip_score(tri, mtriangulations[j], e, SEARCH_DEPTH)
                        if score > 0:
                            escore += score
                    if escore > 0:
                        nscore += escore
                        ncand.append((e, escore))
                if nscore > mscore:
                    mscore = nscore
                    mt = tri
                    mcand = ncand
                    mi = i
            if mscore == 0:
                break
            mcand.sort(key=lambda x:x[1], reverse=True)
            marked = set()
            for (p1, p2), _ in mcand:
                t1 = mt.find_triangle(p1, p2)
                t2 = mt.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                # print(p1, p2)
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                mt.flip(e)
            pfps[mi].append(flips)



    def WriteData(self):

        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        #hy??
        self.instance_uid = self.instance_name
        inst["instance_uid"] = self.instance_uid


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": sum([len(pFlip) for pFlip in self.pFlips])} # , "input": self.input}

        folder = "hy_solutions"
        with open(folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        #verify
        f = open(self.input, "r", encoding="utf-8")
        root = json.load(f)
        f.close()

        instance = CGSHOP2026Instance(
            instance_uid=self.instance_uid,
            points_x=self.pts_x,
            points_y=self.pts_y,
            triangulations=root["triangulations"],
            )
        solution = CGSHOP2026Solution(
                instance_uid=self.instance_uid,
                flips=self.pFlips,
                )

        errors = check_for_errors(instance, solution)

        if errors != []:
            print(errors)
            exit(0)
        else: print("No errors")
        opt_folder = "hy_opt"
        opt_list = os.listdir(opt_folder)
        already_exist = False

        for sol in opt_list:
            if self.instance_uid+".solution.json" in sol:
                already_exist = True

                with open(opt_folder+"/"+sol, "r", encoding="utf-8") as ff:
                    root = json.load(ff)
                    try:
                        old_score = root["meta"]["dist"]
                    except:
                        old_flips = root["flips"]
                        old_score = len([len(x) for x in old_flips])

                if old_score>sum([len(pFlip) for pFlip in self.pFlips]): # self.dist:
                    os.remove(opt_folder+"/"+sol)
                    with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                        json.dump(inst, f, indent='\t')

        if not already_exist:
            with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                json.dump(inst, f, indent='\t')

    def computeDistanceSum(self, centerT):

        my_all_pFlips=[]
        for i in range(len(self.triangulations)):

            my_pFlips=[]

            pFlips_paired = self.parallel_flip_path(self.triangulations[i], centerT)

            for round_ in pFlips_paired:

                round_temp = []

                for oneFlip in round_:

                    (p1, p2) = oneFlip

                    oneFlip_temp = [p1, p2]

                    round_temp.append(oneFlip_temp)

                my_pFlips.append(round_temp)
            my_all_pFlips.append(my_pFlips)
        return my_all_pFlips


def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
