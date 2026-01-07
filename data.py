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
import heapq
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
        print('self.instance_uid:', self.instance_uid)
        # self.distance = []
        # self.pFlips = []

    def ReadData(self):
        if "solution" not in self.input:
        # print("--------------------ReadData--------------------")
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            # print(root)
            self.instance_name = root["instance_uid"]
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
            # print(len(self.pFlips))
        else:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.instance_name = root["instance_uid"]
            self.pFlips = root["flips"]
            self.dist = sum([len(x) for x in self.pFlips])
            try:
                org_input = root["meta"]["input"]
                self.input = org_input
            except:
                self.input = 'data/benchmark_instances/' + self.instance_name + '.json'
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

            print(f"num of pts: {len(self.pts)}")
            print(f"num of triangulations: {len(self.triangulations)}")
            print(f"Original dist: {self.dist}")

            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = copy.deepcopy(self.triangulations[min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip((flp[0], flp[1]))
            new_dist = 0
            for i in range(len(self.triangulations)):
                p_i = self.parallel_flip_path(self.triangulations[i], self.center)
                new_dist+=len(p_i)
            print(f"Original dist: {new_dist}")
            if new_dist<self.dist:
                self.dist = new_dist


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
        tri = copy.deepcopy(tri1)
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
            #input()
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
        start = time.time()
        tri = copy.deepcopy(tri1)
        pfp = [[]]
        when = dict()
        info = dict()
        heap = []
        for t in tri.triangles:
            when[t] = 0
        for e in tri2.edges:
            if e in tri.edges:
                continue
            else:
                # print(e)
                tris, crossings = self.get_crossed_edges_and_triangles(tri, e)
                c, flips = self.get_edge_cost(tri, e, when, crossings, len(self.pts))
                info[e] = (tris, flips)
                heapq.heappush(heap, (-c, e))
        i = 0
        print('initialize done')
        end = time.time()
        print('time:', f"{end - start:.5f} sec")
        while heap:
            _, e = heapq.heappop(heap)
            if e not in info: continue
            else:
                flips = info[e][1]
                used = info[e][0]
                for flip, k in flips:
                    p0, p1 = flip
                    t1 = tri.find_triangle(p0, p1)
                    t2 = tri.find_triangle(p1, p0)
                    when[t1] = k + 1
                    when[t2] = k + 1
                    tri.flip(flip)
                    if k == len(pfp):
                        pfp.append([])
                        print('dist increased to', len(pfp))
                        print(len(info), 'edges remain')
                    pfp[k].append(flip)
                for e in list(info.keys()):
                    if info[e][0] & used:
                        tris, crossings = self.get_crossed_edges_and_triangles(tri, e)
                        c, flips = self.get_edge_cost(tri, e, when, crossings, len(self.pts))
                        if not flips:
                            del info[e]
                            continue
                        info[e] = (tris, flips)
                        heapq.heappush(heap, (-c, e))
            # print(i+1, 'th flip')
            # print('so far', len(pfp[i]), 'flips')
            
            #input()
        return pfp

    def parallel_flip_path3(self, tri1:Triangulation, tri2:Triangulation):
        start = time.time()
        tri = copy.deepcopy(tri1)
        pfp = [[]]
        when = dict()
        info = dict()
        heaps = [[]]
        for t in tri.triangles:
            when[t] = 0
        for e in tri2.edges:
            if e in tri.edges:
                continue
            else:
                # print(e)
                tris, crossings = self.get_crossed_edges_and_triangles(tri, e)
                c, flips = self.get_edge_cost(tri, e, when, crossings, len(self.pts))
                info[e] = (tris, flips)
                heapq.heappush(heaps[0], (-c, e))
        i = 0
        print('initialize done')
        end = time.time()
        print('time:', f"{end - start:.5f} sec")
        while True:
            if not heaps[i]:
                if i == len(heaps) - 1: break
                print(i+1,'th flip done.')
                print(len(pfp[i]), 'flips')
                print(len(info), 'edges remain')
                end = time.time()
                print('time:', f"{end - start:.5f} sec")
                i += 1
                pfp.append([])
            _, e = heapq.heappop(heaps[i])
            if e not in info or info[e][1][0][1] != i: continue
            else:
                flips = info[e][1]
                used = set()
                for flip, k in flips:
                    if k != i:
                        break
                    p0, p1 = flip
                    t1 = tri.find_triangle(p0, p1)
                    t2 = tri.find_triangle(p1, p0)
                    when[t1] = i + 1
                    when[t2] = i + 1
                    used.add(t1)
                    used.add(t2)
                    tri.flip(flip)
                    pfp[i].append(flip)
                if not used: continue
                for e in list(info.keys()):
                    if info[e][0] & used:
                        tris, crossings = self.get_crossed_edges_and_triangles(tri, e)
                        c, flips = self.get_edge_cost(tri, e, when, crossings, len(self.pts))
                        if not flips:
                            del info[e]
                            continue
                        info[e] = (tris, flips)
                        minc = flips[0][1]
                        if minc == len(heaps):
                            heaps.append([])
                        heapq.heappush(heaps[minc], (-c, e))
            # print(i+1, 'th flip')
            # print('so far', len(pfp[i]), 'flips')
            
            #input()
        return pfp


    def parallel_flip_path_rand(self, tri1:Triangulation, tri2:Triangulation, lim=10000):
        tri = copy.deepcopy(tri1)
        pfp = []
        when = dict()
        for t in tri.triangles:
            when[t] = 0
        edges = list(tri2.edges)
        random.shuffle(edges)
        for e in edges:
            if e in tri.edges:
                continue
            # print("making", e)
            if random.choice([True, False]): e = e[1], e[0]
            _, crossings = self.get_crossed_edges_and_triangles(tri, e)
            _, flips = self.get_edge_cost(tri, e, when, crossings, len(self.pts))
            for flip, k in flips:
                if k == lim:
                    return []
                if len(pfp) == k:
                    pfp.append([])
                pfp[k].append(flip)
                p0, p1 = flip
                t1 = tri.find_triangle(p0, p1)
                t2 = tri.find_triangle(p1, p0)
                tri.flip(flip)
                when[t1] = k + 1
                when[t2] = k + 1
        return pfp
            # (1) e와의 cross가 없어지는 flip / (2) e와 cross가 있는 edge를 flippable하게 만들어주는 flip
            # flippability를 저장하는 boolean list 만들기
            # False->True, True->False 중 flippable하게 해주는 친구 찾기 << 두단계 건너뛰어서 있을 수도 있네...
            # (1)과 (2) 중 가장 빨리 flip 할 수 있는거 flip

    def get_crossed_edges_and_triangles(self, tri:Triangulation, e:tuple):
        if e in tri.edges:
            return set(), []
        crossings = []
        affected = set()
        t = self.find_triangle_containing(tri, e)
        i = t.get_ind(e[0])
        affected.add(t)
        crossings = [(t.pt(i+1), t.pt(i+2))]
        t = t.nei(i+1)
        while True:
            affected.add(t)
            p1, p2 = crossings[-1]
            i = t.get_ind(p1)
            q = t.pt(i + 1)
            if q == e[1]: break
            if turn(self.pts[e[0]], self.pts[e[1]], self.pts[q]) > 0:
                crossings.append((p1, q))
                t = t.neis[i]
            else:
                crossings.append((q, p2))
                t = t.nei(i+1)
        return affected, crossings

    def get_edge_cost(self, tri:Triangulation, e:tuple, when:dict, crossings:list, lim:int):
        # parallel flip 가장 늦은거, 총 flip하는 수 tuple 형태로
        # edge에 cost랑 affected 저장
        # print(crossings)
        if not crossings:
            return (0, [])
        resolver = [False] * len(crossings)
        flippability = []
        sameside = []
        ccs = []
        for i in range(len(crossings)):
            c = crossings[i]
            flippability.append(self.flippable(tri,c))
            t1 = tri.find_triangle(c[0], c[1])
            t2 = tri.find_triangle(c[1], c[0])
            cc = max(when[t1], when[t2])
            ccs.append(cc)
            if i == 0: prev = e[0]
            else:
                p1, p2 = crossings[i-1]
                if p1 in c: prev = p2
                else: prev = p1
            if i == len(crossings) - 1: nex = e[1]
            else:
                p1, p2 = crossings[i+1]
                if p1 in c: nex = p2
                else: nex = p1
            sameside.append(turn(self.pts[e[0]], self.pts[e[1]], self.pts[prev]) * turn(self.pts[e[0]], self.pts[e[1]], self.pts[nex]) >= 0)
        
        for i in range(len(crossings)):
            if not flippability[i]:
                c = crossings[i]
                t = tri.find_triangle(c[0], c[1])
                ind = t.get_ind(c[0])
                hf, tmf = self.to_make_flippable(tri, t, ind)
                if i + hf < len(crossings):
                    if crossings[i + hf] == tmf:
                        resolver[i + hf] = True
                t = tri.find_triangle(c[1], c[0])
                ind = t.get_ind(c[1])
                hf, tmf = self.to_make_flippable(tri, t, ind)
                if i - hf >= 0:
                    if crossings[i - hf] == (tmf[1], tmf[0]):
                        resolver[i - hf] = True
        feasible = []
        mincc = max(ccs) + 1
        for i in range(len(crossings)):
            feasible.append(flippability[i] and (sameside[i] or resolver[i]))
            if feasible[i] and ccs[i] < mincc:
                mincc = ccs[i]
                mini = i
        assert(mincc < max(ccs) + 1)
        if mincc > lim:
            return (mincc, [])
        c = crossings[mini]
        t1 = tri.find_triangle(c[0], c[1])
        i = t1.get_ind(c[1])
        prev = t1.pt(i + 1)
        t2 = tri.find_triangle(c[1], c[0])
        j = t2.get_ind(c[0])
        nex = t2.pt(j + 1)
        c1 = when[t1]
        c2 = when[t2]
        when[t1] = mincc + 1
        when[t2] = mincc + 1
        tri.flip(c)
        rec_crossings = list(crossings)
        if sameside[mini]:
            rec_crossings.pop(mini)
        else:
            if turn(self.pts[prev], self.pts[nex], self.pts[e[0]]) > 0: rec_crossings[mini] = (prev, nex)
            else: rec_crossings[mini] = (nex, prev)
        rec_score = self.get_edge_cost(tri, e, when, rec_crossings, lim)
        tri.flip((prev, nex))
        tri.flip(c)
        tri.flip((prev, nex))
        t1 = tri.find_triangle(c[0], c[1])
        t2 = tri.find_triangle(c[1], c[0])
        when[t1] = c1
        when[t2] = c2
        '''
        if (mini < len(crossings) - 1) and feasible[mini+1] and (mincc == ccs[mini+1]):
            mini = mini + 1
            c = crossings[mini]
            t1 = tri.find_triangle(c[0], c[1])
            i = t1.get_ind(c[1])
            prev = t1.pt(i + 1)
            t2 = tri.find_triangle(c[1], c[0])
            j = t2.get_ind(c[0])
            nex = t2.pt(j + 1)
            c1 = when[t1]
            c2 = when[t2]
            when[t1] = mincc + 1
            when[t2] = mincc + 1
            tri.flip(c)
            rec_crossings = list(crossings)
            if sameside[mini]:
                rec_crossings.pop(mini)
            else:
                if turn(self.pts[prev], self.pts[nex], self.pts[e[0]]) > 0: rec_crossings[mini] = (prev, nex)
                else: rec_crossings[mini] = (nex, prev)
            rec_score2 = self.get_edge_cost(tri, e, when, rec_crossings, rec_score[0])
            flag = False
            if rec_score2[0] < rec_score[0] or (rec_score2[0] == rec_score[0] and len(rec_score2[1]) < len(rec_score[1])):
                rec_score = rec_score2
                flag = True
            tri.flip((prev, nex))
            t1 = tri.find_triangle(c[0], c[1])
            t2 = tri.find_triangle(c[1], c[0])
            when[t1] = c1
            when[t2] = c2
            if not flag: c = crossings[mini - 1]
        '''
        # print(crossings)
        # print([(c, mincc)] + rec_score[1])
        # print('flip', c, 'at', mincc)
        # print('whens:', when[t1], when[t2])
        return (max(mincc, rec_score[0]), [(c, mincc)] + rec_score[1])


    
    def to_make_flippable(self, tri, t, i):
        p = self.pts[t.pt(i - 1)]
        pr = self.pts[t.pts[i]]
        pl = self.pts[t.pt(i + 1)]
        tt = t.neis[i]
        j = tt.get_ind(t.pt(i + 1))
        q = self.pts[tt.pt(j + 2)]
        num = 0
        while True:
            if turn(p, pr, q) <= 0:
                t = tt
                i = (j + 2) % 3
                pr = q
            elif turn(self.pts[t.pt(i + 2)], self.pts[t.pts[i]], q) <= 0:
                t = tt
                i = (j + 2) % 3
            elif turn(p, pl, q) >= 0:
                pl = q
                t = tt
                i = (j + 1) % 3
            elif turn(self.pts[t.pt(i + 2)], self.pts[tt.pts[j]], q) >= 0:
                t = tt
                i = (j + 1) % 3
            else:
                break
            tt = t.neis[i]
            if not tt: break
            j = tt.get_ind(t.pt(i + 1))
            q = self.pts[tt.pt(j + 2)]
            num += 1
        return num, (t.pts[i], t.pt(i + 1))



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
        num = len(mtriangulations)
        pfps =[[] for _ in range(num)]
        tl = 0
        while True:
            mscore = 0
            flips = [[] for _ in range(num)]
            for i in range(num):
                ncand = []
                nscore = 0
                tri = mtriangulations[i]
                edges = list(tri.edges)
                for e in edges:
                    if not self.flippable(tri, e): continue
                    escore = 0
                    for j in range(num):
                        if i==j: continue
                        score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
                        escore += score
                    if escore > 0:
                        ncand.append((e, escore))
                ncand.sort(key = lambda x:x[1], reverse=True)
                marked = set()
                for (p1, p2), score in ncand:
                    t1 = tri.find_triangle(p1, p2)
                    t2 = tri.find_triangle(p2, p1)
                    if t1 in marked or t2 in marked:
                        continue
                    flips[i].append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                    nscore += score
                
                if nscore > mscore:
                    mscore = nscore
                    mi = i
            if mscore == 0:
                break
            tl+=1
            print(tl)
            print(mscore)
            end = time.time()
            print('time:', f"{end - start:.5f} sec")
            pfps[mi].append(flips[mi])
            for e in flips[mi]:
                mtriangulations[mi].flip(e)
        self.pFlips = []
        for i in range(num):
            if i < num-1:
                assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
            self.pFlips.append(pfps[i])
        
        print("total length:",tl)
        return mtriangulations[0]
                    
    def findCenterGlobal2(self):
        start = time.time()
        mtriangulations = [copy.deepcopy(t) for t in self.triangulations]
        num = len(mtriangulations)
        pfps =[[] for _ in range(num)]
        scores = [dict() for _ in range(num)]
        tl = 0
        for i in range(num):
            tri = mtriangulations[i]
            edges = list(tri.edges)
            for e in edges:
                if not self.flippable(tri, e): continue
                escore = 0
                for j in range(num):
                    if i==j: continue
                    score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
                    escore += score
                scores[i][e] = escore
            print(i+1, "/", num)
            end = time.time()
            print('time:', f"{end - start:.5f} sec")
        while True:
            mscore = 0
            for i in range(num):
                tri = mtriangulations[i]
                ncand = []
                nflips = []
                nscore = 0
                for e in scores[i]:
                    if scores[i][e] > 0:
                        ncand.append((e, scores[i][e]))
                ncand.sort(key=lambda x:x[1], reverse=True)
                marked = set()
                for (p1, p2), score in ncand:
                    t1 = tri.find_triangle(p1, p2)
                    t2 = tri.find_triangle(p2, p1)
                    if t1 in marked or t2 in marked:
                        continue
                    nflips.append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                    nscore += score    
                if nscore > mscore:
                    mscore = nscore
                    mi = i
                    mflips = nflips
            if mscore == 0:
                break
            tl+=1
            print(tl)
            print(mscore)
            end = time.time()
            print('time:', f"{end - start:.5f} sec")
            pfps[mi].append(mflips)
            for e in mflips:
                mtriangulations[mi].flip(e)
            for i in range(num):
                if i == mi:
                    scores[i].clear()
                    tri = mtriangulations[i]
                    edges = list(tri.edges)
                    for e in edges:
                        if not self.flippable(tri, e): continue
                        escore = 0
                        for j in range(num):
                            if i==j: continue
                            score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
                            escore += score
                        scores[i][e] = escore
                else:
                    tri = mtriangulations[i]
                    edges = list(tri.edges)
                    for e in edges:
                        if not self.flippable(tri, e): continue
                        score, _ = self.flip_score(tri, mtriangulations[mi], e, 1)
                        scores[i][e] += score
        self.pFlips = []
        for i in range(num):
            if i < num-1:
                assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
            self.pFlips.append(pfps[i])
        
        print("total length:",tl)
        return mtriangulations[0]

    def WriteData(self):

        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_uid


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": sum([len(pFlip) for pFlip in self.pFlips])} # , "input": self.input}
        
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
            df = DataFrame(df_dict)
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

        start = time.time()

        print(self.pFlips)
        print(len(self.pFlips))
        print(len(self.triangulations))

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
