import json
import sys
import random
from Point import Point
from Triangulation import Triangle, Triangulation
import copy
import time
import os
import pandas as pd
import datetime
import numpy as np
from pathlib import Path
import pdb
from multiprocessing import Process, Pool

sys.setrecursionlimit(1000000)
SEARCH_DEPTH = 1
PAR_LEN = 2.5
PAR_CROSS = 1

class Data:
    def __init__(self, inp=''):
        if not inp: 
            self.triangulations = []
            self.log = False
        else:
            self.input = inp
            self.triangulations = []
            self.ReadData()
            self.instance_uid = (inp.split('/')[-1]).split('.')[0]
            self.log = False

    def __del__(self):
        for t in self.triangulations:
            del t

    def ReadData(self):
        if "solution" not in self.input:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
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

            self.pFlips = [None] * len(self.triangulations)
            self.center = self.triangulations[0].fast_copy()

        else:
            f = open(self.input, "r", encoding="utf-8")
            root = json.load(f)
            self.instance_name = root["instance_uid"]
            self.instance_uid = self.instance_name
            self.pFlips = root["flips"]
            self.dist = sum([len(x) for x in self.pFlips])
            org_dist = self.dist
            try:
                org_input = root["meta"]["input"]
            except:
                org_input = "data/benchmark_instances/"+self.instance_uid+".json"

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

            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.make_triangulation(root["triangulations"][min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip((flp[0], flp[1]))

            
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
        if depth == 0:
            return (ori_cross, 0)
        new_cross = self.count_cross(tri_dest, (p2, p4))
        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        if depth == 1:
            return m_score
        tri.flip(e)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                nsc = self.flip_score(tri, tri_dest, pe, depth - 1)
                m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))
        tri.flip((p2, p4))
        return m_score    
    

    def findCenterGlobal(self):
        if self.log:
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
            if self.log:
                print(i, "/", num)
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
            if self.log:
                print(tl)
                print(mscore)
                end = time.time()
                print('time:', f"{end - start:.5f} sec")
            pfps[mi].append(mflips)
            for i in range(num):
                if i == mi:
                    continue
                tri = mtriangulations[i]
                edges = list(tri.edges)
                for e in edges:
                    if not self.flippable(tri, e): continue
                    score, _ = self.flip_score(tri, mtriangulations[mi], e, 1)
                    scores[i][e] -= score
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
        
        if self.log: print("total length:",tl)
        while len(mtriangulations) > 1:
            tri = mtriangulations.pop()
            del tri

        self.dist = sum([len(pFlip) for pFlip in self.pFlips])
        return mtriangulations[0]

    def WriteData(self):

        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_name

        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": self.dist, "input": self.input, "center": self.center.return_edge()}
        
        folder = "solutions"
        with open(folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
