import json
import sys
import random
from Point import Point
from Triangulation import Triangle, Triangulation
import copy
# import cv2
import time
import os
import pandas as pd
import datetime
import numpy as np
from pathlib import Path
import pdb
# from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
# from cgshop2026_pyutils.geometry import FlippableTriangulation
# from cgshop2026_pyutils.verify import check_for_errors
# import cgshop2026_pyutils

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
        # print('self.instance_uid:', self.instance_uid)
        # self.distance = []
        # self.pFlips = []

    def ReadData(self):
        if "solution" not in self.input:
        # print("--------------------ReadData--------------------")
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            # print(root)
            self.instance_name = root["instance_uid"]
            self.instance_uid = self.instance_name
            print('self.instance_uid:', self.instance_uid)
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = []
            for i in range(len(self.pts_y)):
                self.pts.append(Point(self.pts_x[i], self.pts_y[i]))
            for t in root["triangulations"]:
                self.triangulations.append(self.make_triangulation(t))
                print(len(self.triangulations),"/",len(root["triangulations"]))

            #print(len(self.triangulations))
            # self.distance = [] * len(self.triangulations)
            self.pFlips = [None] * len(self.triangulations)
            self.center = self.triangulations[0].fast_copy()
            # print(len(self.pFlips))
        else:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.instance_name = root["instance_uid"]
            self.instance_uid = self.instance_name
            print('instance_uid:', self.instance_uid)
            self.pFlips = root["flips"]
            self.dist = sum([len(x) for x in self.pFlips])
            org_dist = self.dist
            org_input = root["meta"]["input"]

            # self.input = Path(org_input)
            self.input =  org_input.replace("\\", "/")
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = []
            for i in range(len(self.pts_y)):
                self.pts.append(Point(self.pts_x[i], self.pts_y[i]))
            for t in root["triangulations"]:
                self.triangulations.append(self.make_triangulation(t))
                print(len(self.triangulations),"/",len(root["triangulations"]))

            # print(f"num of pts: {len(self.pts)}")
            # print(f"num of triangulations: {len(self.triangulations)}")
            # print(f"Original dist: {self.dist}")

            # pdb.set_trace()
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.make_triangulation(root["triangulations"][min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip((flp[0], flp[1]))
            # self.computeDistanceSum(self.center)
            # print(f"New dist: {self.dist}")
            # if self.dist<org_dist:
            #     self.WriteData()
            print("---------------------------------")
            


    def make_triangulation(self, t: Triangulation):
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
        # tri = self.make_triangulation(tri1.return_edge())
        tri = tri1.fast_copy()
        pfp = []
        while True:
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    # 전에 뒤집은거 안뒤집게 해야할듯?
                    score = self.flip_score(tri, tri2, e, SEARCH_DEPTH)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            # print(cand)
            # print(len(cand))
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                # print(p1, p2)
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                tri.flip(e)
            pfp.append(flips)
            # print(len(flips))
        assert(tri.edges == tri2.edges)
        return pfp
    
    def parallel_flip_path2(self, tri1:Triangulation, tri2:Triangulation):
        # tri = self.make_triangulation(tri1.return_edge())
        tri = tri1.fast_copy()
        pfp = []
        while True:
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    # 전에 뒤집은거 안뒤집게 해야할듯?
                    score = self.flip_score(tri, tri2, e, 0)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            # print(cand)
            # print(len(cand))
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                # print(p1, p2)
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                tri.flip(e)
            pfp.append(flips)
            # print(len(flips))
        assert(tri.edges == tri2.edges)
        return pfp
            
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
        # print(depth, e)
        t1 = tri.find_triangle(p1, p3)
        t2 = tri.find_triangle(p3, p1)
        #self.print_triangle(t1)
        #self.print_triangle(t2)
        i = t1.get_ind(p3)
        p4 = t1.pt(i + 1)
        j = t2.get_ind(p1)
        p2 = t2.pt(j + 1)
        ori_cross = self.count_cross(tri_dest, e)
        if depth == 0:
            return (ori_cross, 0)
        new_cross = self.count_cross(tri_dest, (p2, p4))
        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        # return m_score
        if depth == 1:
            return m_score
        # self.print_triangle(t1)
        # self.print_triangle(t2)
        #print("try flipping(1)",e)
        tri.flip(e)
        # self.print_triangle(t1)
        # self.print_triangle(t2)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                # print(pe)
                nsc = self.flip_score(tri, tri_dest, pe, depth - 1)
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
        print('len(pfp):', len(pfp), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)

        # use_pfp: 사용되는 parallel flip들 모음
        use_pfp = pfp[:midVal]

        for flips in use_pfp:
            for flip in flips:
                
                T1copy.flip(flip)
        return T1copy
        
    def findCenter(self):

        start = time.time()

        # random.shuffle(self.triangulations)

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
        start = time.time()
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
        inst["instance_uid"] = self.instance_name


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": self.dist, "input": self.input, "center": self.center.return_edge()}
        
        folder = "solutions"
        with open(folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        opt_folder = "opt"
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

        fname = "result.csv"
        if not os.path.exists(fname):
            df_dict = dict()
            df_dict["date"] = datetime.date.today()
            df_dict[self.instance_uid] = [self.dist]
            df = pd.DataFrame(df_dict)
            df.to_csv("result.csv")

        else:
            df = pd.read_csv(fname, index_col = 0)
            col = df.columns
            if self.instance_uid not in col:
                df[self.instance_uid] = float("INF")
            today = datetime.date.today().isoformat()
            # pdb.set_trace()
            if df["date"].iloc[-1]!=today:
                df.loc[len(df)] = list(df.iloc[-1])
                df.loc[len(df.index)-1, "date"] = today

            df.loc[len(df.index)-1, self.instance_uid] = min(df.loc[len(df.index)-1, self.instance_uid], sum([len(pFlip) for pFlip in self.pFlips]))
            df.to_csv("result.csv")
            pass

    def computeDistanceSum(self, centerT):

        # start = time.time()

        # print(self.pFlips)
        # print(len(self.pFlips))
        # print(len(self.triangulations))
        tot_dist = 0
        for i in range(len(self.triangulations)):
            
            # list[list[list[int, int]]]
            self.pFlips[i] = []

            # list[list[list[tuple(int, int), tuple(int, int)]]]
            pFlips_paired = self.parallel_flip_path(self.triangulations[i], centerT)

            for round in pFlips_paired:

                round_temp = []

                for oneFlip in round:
                    
                    # (p1, p2), (p3, p4) = fs[i]
                    (p1, p2) = oneFlip
                
                    oneFlip_temp = [p1, p2]

                    round_temp.append(oneFlip_temp)

                self.pFlips[i].append(round_temp)
            tot_dist+=len(pFlips_paired)
        


            '''
            pfp = self.parallel_flip_path(self.triangulations[i], centerT)
            self.pFlips[i] = []

            # (a, b), (c, d) 를 (a, b)로 바꿔야 함
            for round in pfp:
                roundFlips = []
                for singleFlip in round:
                    roundFlips.append(singleFlip[0])
                self.pFlips[i].append(roundFlips)
            self.pFlips[i].reverse()
            '''

            # print('parallel flip distance from the center to T', i, ':', len(self.pFlips[i]))
            
            # end = time.time()
            # print('time:', f"{end - start:.5f} sec")
        self.dist = tot_dist
        print(f"New dist: {tot_dist}")

    def computePFDOnly(self, centerT):

        # start = time.time()

        # print(self.pFlips)
        # print(len(self.pFlips))
        # print(len(self.triangulations))
        tot_dist = 0
        pF = []
        for i in range(len(self.triangulations)):
            
            # list[list[list[int, int]]]
            pFi = []

            # list[list[list[tuple(int, int), tuple(int, int)]]]
            pFlips_paired = self.parallel_flip_path(self.triangulations[i], centerT)
            for round in pFlips_paired:

                round_temp = []

                for oneFlip in round:
                    
                    # (p1, p2), (p3, p4) = fs[i]
                    (p1, p2) = oneFlip
                
                    oneFlip_temp = [p1, p2]

                    round_temp.append(oneFlip_temp)

                pFi.append(round_temp)
            tot_dist+=len(pFlips_paired)
            pF.append(pFi)

        return pF, tot_dist

    def random_move(self):
        prev_len = self.dist
        total_best = prev_len
        T = self.center.fast_copy()

        print(f"Start with {prev_len}")

        step = 0
        total_step = 0
        end_step = 3 * len(self.triangulations) * len(self.pts)

        edges = list(T.edges)
        starting_edge_ind = 0
        random.shuffle(edges)

        while total_step < end_step:
            total_step += 1
            do_random = random.random() > 0.999 ** step

            if do_random or starting_edge_ind == len(edges):
                # print(f"{total_step}/{end_step} Shuffle")
                cand = []
                edges = list(T.edges)
                for e in edges:
                    if self.flippable(T, e):
                        cand.append(e)
                if not cand:
                    break
                random.shuffle(cand)
                random_choice = [random.random() for _ in range(len(cand))]
                e_list = [e for i, e in enumerate(cand) if random_choice[i] > 0.5]
                flips = []
                marked = set()
                for (p1, p2)in e_list:
                    t1 = T.find_triangle(p1, p2)
                    t2 = T.find_triangle(p2, p1)
                    if t1 in marked or t2 in marked:
                        continue
                    # print(p1, p2)
                    flips.append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                for e in flips:
                    T.flip((e[0], e[1]))

                edges = list(T.edges)
                random.shuffle(edges)
                starting_edge_ind = 0

                # quick length-only evaluation
                newpF, new_len = self.computePFDOnly(T)
                if total_best>new_len:
                    self.dist = new_len
                    self.pFlips = newpF
                    self.WriteData()
                total_best = min(total_best, new_len)
                prev_len = new_len
                step = 0
                continue

            # single flip try
            T1 = T.fast_copy()
            e = edges[starting_edge_ind]
            if not self.flippable(T1, e):
                starting_edge_ind += 1
                continue

            T1.flip(e)

            pF, new_len = self.computePFDOnly(T)
            if new_len <= prev_len:
                # print(f"[{self.instance_uid} {total_step}/{end_step}] {total_best}->{new_len}")
                step = 0
                T = T1
                edges = list(T.edges)
                random.shuffle(edges)

                if new_len < prev_len:
                    # only now compute full flip list
                    self.center = T.fast_copy()
                    self.dist = new_len
                    self.flip = pF
                    if new_len < total_best:
                        print(f"[{self.instance_uid} {total_step}/{end_step}] {total_best}->{new_len}")
                        total_best = new_len
                        self.WriteData()
                    prev_len = min(prev_len, new_len)

                starting_edge_ind = 0
            else:
                step += 1
                starting_edge_ind += 1

        return self.center
    # def computeDistanceSum(self, centerT):

    #     start = time.time()

    #     print(self.pFlips)
    #     print(len(self.pFlips))
    #     print(len(self.triangulations))

    #     for i in range(len(self.triangulations)):
            
    #         # list[list[list[int, int]]]
    #         self.pFlips[i] = []

    #         # list[list[list[tuple(int, int), tuple(int, int)]]]
    #         pFlips_paired = self.parallel_flip_path(self.triangulations[i], centerT)

    #         for round in pFlips_paired:

    #             round_temp = []

    #             for oneFlip in round:
                    
    #                 # (p1, p2), (p3, p4) = fs[i]
    #                 (p1, p2) = oneFlip
                
    #                 oneFlip_temp = [p1, p2]

    #                 round_temp.append(oneFlip_temp)

    #             self.pFlips[i].append(round_temp)


    #         '''
    #         pfp = self.parallel_flip_path(self.triangulations[i], centerT)
    #         self.pFlips[i] = []

    #         # (a, b), (c, d) 를 (a, b)로 바꿔야 함
    #         for round in pfp:
    #             roundFlips = []
    #             for singleFlip in round:
    #                 roundFlips.append(singleFlip[0])
    #             self.pFlips[i].append(roundFlips)
    #         self.pFlips[i].reverse()
    #         '''

    #         print('parallel flip distance from the center to T', i, ':', len(self.pFlips[i]))
            
    #         end = time.time()
    #         print('time:', f"{end - start:.5f} sec")
            
    def computeDistanceSum2(self, centerT):

        start = time.time()

        print(self.pFlips)
        print(len(self.pFlips))
        print(len(self.triangulations))

        for i in range(len(self.triangulations)):
            
            # list[list[list[int, int]]]
            self.pFlips[i] = []

            # list[list[list[tuple(int, int), tuple(int, int)]]]
            pFlips_paired = self.parallel_flip_path2(self.triangulations[i], centerT)

            for round in pFlips_paired:

                round_temp = []

                for oneFlip in round:
                    
                    # (p1, p2), (p3, p4) = fs[i]
                    (p1, p2), (p3, p4) = oneFlip
                
                    oneFlip_temp = [p1, p2]

                    round_temp.append(oneFlip_temp)

                self.pFlips[i].append(round_temp)


            '''
            pfp = self.parallel_flip_path(self.triangulations[i], centerT)
            self.pFlips[i] = []

            # (a, b), (c, d) 를 (a, b)로 바꿔야 함
            for round in pfp:
                roundFlips = []
                for singleFlip in round:
                    roundFlips.append(singleFlip[0])
                self.pFlips[i].append(roundFlips)
            self.pFlips[i].reverse()
            '''

            print('parallel flip distance from the center to T', i, ':', len(self.pFlips[i]))
            
            end = time.time()
            print('time:', f"{end - start:.5f} sec")
            
    def verify(self):
        pass

        # # Define points (square) and two triangulations that will be flipped to a common form
        # points_x = [0, 1, 0, 1]
        # points_y = [0, 0, 1, 1]

        # '''
        # triangulations = [  # Each triangulation is a list of interior edges
        #     [(0, 3)],        # diagonal 0-3
        #     [(1, 2)],        # diagonal 1-2 (the flip partner)
        # ]
        # '''

        # instance = CGSHOP2026Instance(
        #     instance_uid=self.instance_uid,
        #     points_x=self.pts_x,
        #     points_y=self.pts_y,

        #     # triangulations=[T.getEdges() for T in self.triangulations],
        #     triangulations=[T.edges for T in self.triangulations],
        # )

        # # A solution that flips the diagonal in the first triangulation to match the second.
        # # flips is: one list per triangulation -> sequence of parallel flip sets -> each set is a list of edges
        # solution = CGSHOP2026Solution(
        #     instance_uid=self.instance_uid,
        #     flips=self.pFlips
        #     # [ [[(0,3)]] , [] ]  # flip edge (0,3) in triangulation 0; triangulation 1 already in target form
        # )

        # errors = check_for_errors(instance, solution)
        # print("Errors:", errors or "None ✔")

# CCW라면 양수 반환, CW라면 음수 반환
# collinear라면 0 반환
# https://www.geeksforgeeks.org/dsa/orientation-3-ordered-points/

'''
def turn(p1: MyPoint, p2: MyPoint, p3: MyPoint):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
'''

def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
