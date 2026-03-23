import json
import sys
import random
from Point import Point
from Triangulation import Triangle, Triangulation
import copy
import time
import os
#import pandas as pd
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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from math import hypot
try:
    import dill as _pickle
except Exception:
    import pickle as _pickle


from multiprocessing import shared_memory

def shm_create_from_numpy(arr: np.ndarray, name: str):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm

def shm_attach_array(name: str, shape, dtype):
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr

def build_triangulation_from_edges(pts_x, pts_y, edge_list):
    pts = [Point(pts_x[i], pts_y[i]) for i in range(len(pts_x))]
    tri = Triangulation()
    tri.adj = [-1] * len(pts)
    graph = [[] for _ in range(len(pts))]
    for u, v in edge_list:
        graph[u].append(v); graph[v].append(u)
        tri.adj[u] = v; tri.adj[v] = u
        tri.edges.add((min(u, v), max(u, v)))
    for i in range(len(pts)):
        for j in range(len(graph[i])):
            v1 = graph[i][j]
            if v1 < i:
                continue
            for k in range(j + 1, len(graph[i])):
                v2 = graph[i][k]
                if v2 < i:
                    continue
                if v1 in graph[v2]:
                    if turn(pts[i], pts[v1], pts[v2]) > 0:
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
                            smallflag &= turn(pts[t.pt(m)], pts[t.pt(m + 1)], pts[v3]) >= 0
                        if smallflag:
                            flag = True; break
                    if flag:
                        del t; continue
                    for l in range(3):
                        tri.dict[t.pt(l), t.pt(l + 1)] = t
                        tt = tri.find_triangle(t.pt(l + 1), t.pt(l))
                        if tt:
                            tt.neis[tt.get_ind(t.pt(l + 1))] = t
                            t.neis[l] = tt
                    tri.triangles.add(t)
    return tri

def _build_prefix_states(base_tri, seq):
    prefix = [base_tri.fast_copy()]
    cur = prefix[0]
    for i in range(len(seq)):
        nxt = cur.fast_copy()
        for e in seq[i]:
            nxt.flip(e)
        prefix.append(nxt)
        cur = nxt
    return prefix

def _tri_fingerprint(tri):
    try:
        E = tri.edges() if callable(getattr(tri, "edges", None)) else tri.edges
        norm = []
        for (a, b) in E:
            norm.append((a, b) if a <= b else (b, a))
        norm.sort()
        return hash(tuple(norm))
    except Exception:
        if hasattr(tri, "to_signature"):
            return hash(tri.to_signature())
        return id(tri)

def _rebuild_prefix_from(prefix, start_i, seq):
    cur = prefix[start_i]
    del prefix[start_i+1:]
    for i in range(start_i, len(seq)):
        nxt = cur.fast_copy()
        for e in seq[i]:
            nxt.flip(e)
        prefix.append(nxt)
        cur = nxt
def flippable(pts, tri, e):
    p1, p3 = e
    t1 = tri.find_triangle(p1, p3)
    t2 = tri.find_triangle(p3, p1)
    if (not t1) or (not t2):
        return False
    i = t1.get_ind(p3); p4 = t1.pt(i + 1)
    j = t2.get_ind(p1); p2 = t2.pt(j + 1)
    return (turn(pts[p2], pts[p3], pts[p4]) > 0) and (turn(pts[p2], pts[p1], pts[p4]) < 0)

def find_triangle_containing(pts, tri, con):
    q1, q2 = con
    if tri.find_triangle(q1, q2) or tri.find_triangle(q2, q1):
        return None
    p = tri.adj[q1]
    t = tri.find_triangle(q1, p) or tri.find_triangle(p, q1)
    r1 = pts[q1]; r4 = pts[q2]
    while True:
        i = t.get_ind(q1)
        r2 = pts[t.pt(i + 1)]
        r3 = pts[t.pt(i + 2)]
        if turn(r1, r2, r4) < 0:
            t = t.neis[i]
        elif turn(r1, r3, r4) > 0:
            t = t.nei(i+2)
        else:
            return t

def count_cross(pts, tri, con):
    t = find_triangle_containing(pts, tri, con)
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
        if turn(pts[q1], pts[q2], pts[t.pt(i + 1)]) < 0:
            tt = t.nei(i + 1)
            j = tt.get_ind(t.pt(i + 1))
        else:
            tt = t.neis[i]
            j = tt.get_ind(t.pts[i])
    return cnt

def flip_score(pts, tri, tri_dest, e, depth):
    p1, p3 = e
    t1 = tri.find_triangle(p1, p3)
    t2 = tri.find_triangle(p3, p1)
    i = t1.get_ind(p3); p4 = t1.pt(i + 1)
    j = t2.get_ind(p1); p2 = t2.pt(j + 1)
    ori_cross = count_cross(pts, tri_dest, e)
    if depth == 0:
        return (ori_cross, 0)
    new_cross = count_cross(pts, tri_dest, (p2, p4))
    n_cross = ori_cross - new_cross
    m_score = (n_cross, depth)
    if depth == 1:
        return m_score
    tri.flip(e)
    for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
        if flippable(pts, tri, pe):
            nsc = flip_score(pts, tri, tri_dest, pe, depth - 1)
            m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))
    tri.flip((p2, p4))
    return m_score

def parallel_flip_path(pts, tri1, tri2, depth=1):
    tri = tri1.fast_copy()
    pfp = []
    while True:
        cand = []
        for e in list(tri.edges):
            if flippable(pts, tri, e):
                score = flip_score(pts, tri, tri2, e, depth)
                if score[0] > 0:
                    cand.append((e, score))
        if not cand:
            break
        cand.sort(key=lambda x: x[1], reverse=True)
        flips, marked = [], set()
        for (p1, p2), _ in cand:
            t1 = tri.find_triangle(p1, p2)
            t2 = tri.find_triangle(p2, p1)
            if t1 in marked or t2 in marked:
                continue
            flips.append((p1, p2))
            marked.add(t1); marked.add(t2)
        for e in flips:
            tri.flip(e)
        pfp.append(flips)
    assert(tri.edges == tri2.edges)
    return pfp

def parallel_flip_path_reverse(pts, tri1, tri2, depth=1):
    tri = tri2.fast_copy()
    pfp = []
    while True:
        cand = []
        for e in list(tri.edges):
            if flippable(pts, tri, e):
                score = flip_score(pts, tri, tri1, e, depth)
                if score[0] > 0:
                    cand.append((e, score))
        if not cand:
            break
        cand.sort(key=lambda x: x[1], reverse=True)
        flips, marked = [], set()
        for (p1, p2), _ in cand:
            t1 = tri.find_triangle(p1, p2)
            t2 = tri.find_triangle(p2, p1)
            if t1 in marked or t2 in marked:
                continue
            flips.append((p1, p2))
            marked.add(t1); marked.add(t2)
        flips_rev = []
        for e in flips:
            flips_rev.append(tri.flip(e))
        pfp.append(flips_rev)
    assert(tri.edges == tri1.edges)
    return pfp[::-1]

def compuePFS_total_pure(pts_x, pts_y, T1, T2):
    pts = [Point(pts_x[i], pts_y[i]) for i in range(len(pts_x))]
    p1 = parallel_flip_path(pts, T1, T2)
    p1r = parallel_flip_path_reverse(pts, T1, T2)
    path_list = [p1, p1r]
    opt = min(path_list, key=len)
    out = []
    for rnd in opt:
        out.append([[a, b] for (a, b) in rnd])
    return out

def _pfd2_single_triangulation_worker(payload_bytes):
    (tri_num, base_edges, seq, shm_meta, lru_capacity) = _pickle.loads(payload_bytes)

    shm_x, X = shm_attach_array(shm_meta["x"]["name"],
                                tuple(shm_meta["x"]["shape"]),
                                np.dtype(shm_meta["x"]["dtype"]))
    shm_y, Y = shm_attach_array(shm_meta["y"]["name"],
                                tuple(shm_meta["y"]["shape"]),
                                np.dtype(shm_meta["y"]["dtype"]))
    try:
        base = build_triangulation_from_edges(X, Y, base_edges)

        return _pickle.dumps((tri_num, seq, total_delta, improved))
    finally:
        shm_x.close(); shm_y.close()


class Data:
    def __init__(self, inp):
        self.input = inp
        self.triangulations = []
        self.ReadData()
        self.instance_uid = (inp.split('/')[-1]).split('.')[0]

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

    def parallel_flip_path(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri1.fast_copy()
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

    def parallel_flip_path_reverse(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri2.fast_copy()
        pfp = []
        while True:
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri1, e, SEARCH_DEPTH)
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
            flips_reverse = []
            for e in flips:
                flips_reverse.append(tri.flip(e))
            pfp.append(flips_reverse)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]

    def parallel_flip_path_2way(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri1.fast_copy()
        trii = tri2.fast_copy()
        pfp = []
        reverse_pfp = []
        while True:
            cand = []
            total_val1 = 0
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, trii, e, SEARCH_DEPTH)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips1 = []
            marked = set()
            for (p1, p2), val in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips1.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
                total_val1+=val[0]
            cand = []
            total_val2 = 0
            edges = list(trii.edges)
            for e in edges:
                if self.flippable(tri, e):
                    score = self.flip_score(trii, tri, e, SEARCH_DEPTH)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips2 = []
            marked = set()
            for (p1, p2), val in cand:
                t1 = trii.find_triangle(p1, p2)
                t2 = trii.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips2.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
                total_val2+=val[0]
            if total_val1>total_val2:
                for e in flips1:
                    tri.flip(e)
                pfp.append(flips1)
            else:
                flips = []
                for e in flips2:
                    flip_e = trii.flip(e)
                    flips.append(flip_e)
                reverse_pfp.append(flips)
        assert(tri.edges == trii.edges)
        return pfp+reverse_pfp[::-1]

    def parallel_flip_path2(self, tri1:Triangulation, tri2:Triangulation):
        tri = tri1.fast_copy()
        pfp = []
        prev_flip = set()
        step = 0
        while True:
            step+=1
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    if e in prev_flip:
                        continue

                    score = self.flip_score(tri, tri2, e, 0)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip = []
                    continue
                else:
                    break
            cand.sort(key=lambda x: x[1],reverse=True)
            if step>100:
                print(cand)
                print(prev_flip)
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
            prev_flip = []
            for e in flips:
                e1 = tri.flip(e)
                prev_flip.append(e1)
            prev_flip = set(prev_flip)
            pfp.append(flips)

        assert(tri.edges == tri2.edges)
        return pfp

    def parallel_flip_path2_reverse(self, tri1:Triangulation, tri2:Triangulation):

        tri = tri2.fast_copy()
        pfp = []
        prev_flip = set()
        step = 0
        while True:
            step+=1
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    if e in prev_flip:
                        continue

                    score = self.flip_score(tri, tri1, e, 0)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip = []
                    continue
                else:
                    break
            cand.sort(key=lambda x: x[1],reverse=True)
            if step>100:
                print(cand)
                print(prev_flip)
            # print(len(cand))
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
            prev_flip = []
            for e in flips:
                e1 = tri.flip(e)
                prev_flip.append(e1)
            prev_flip = set(prev_flip)
            pfp.append(prev_flip)

        assert(tri.edges == tri1.edges)
        return pfp[::-1]

    def parallel_flip_path3(self, tri1:Triangulation, tri2:Triangulation):

        tri = tri1.fast_copy()
        pfp = []
        step = 0
        while True:
            step+=1
            assert step<1000, f"Too many steps in parallel_flip_path3 for {self.instance_uid}"
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):

                    score = self.flip_score2(tri, tri2, e, SEARCH_DEPTH)
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

    def parallel_flip_path3_reverse(self, tri1:Triangulation, tri2:Triangulation):

        tri = tri2.fast_copy()
        pfp = []
        step = 0
        while True:
            step+=1
            assert step<1000, f"Too many steps in parallel_flip_path3 for {self.instance_uid}"
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):

                    score = self.flip_score2(tri, tri1, e, SEARCH_DEPTH)
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
            pfp1 = []
            for e in flips:
                pfp1.append(tri.flip(e))
            pfp.append(pfp1)

        assert(tri.edges == tri1.edges)
        return pfp[::-1]

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

    def flip_score2(self, tri:Triangulation, tri_dest:Triangulation, e:tuple, depth:int):
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
        n_cross = (ori_cross - new_cross)/ori_cross if ori_cross>0 else - new_cross
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

    def internal_division(self, T1: Triangulation, w1: int, T2: Triangulation, w2: int):

        T1copy = copy.deepcopy(T1)

        # parallel version
        pfp = self.parallel_flip_path(T1, T2)
        midVal = int(len(pfp) * (w2 / (w1 + w2)))
        print('len(pfp):', len(pfp), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)

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

                if old_score>sum([len(pFlip) for pFlip in self.pFlips]):
                    os.remove(opt_folder+"/"+sol)
                    with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                        json.dump(inst, f, indent='\t')

        if not already_exist:
            with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                json.dump(inst, f, indent='\t')


    def compuePFS_total(self, T1, T2):
        prev_pFlips_i = []
        pFlips_paired1 = self.parallel_flip_path(T1, T2)
        pFlips_paired11 = self.parallel_flip_path_reverse(T1, T2)
        pFlips_paired2 = self.parallel_flip_path2(T1, T2)
        pFlips_paired21 = self.parallel_flip_path2_reverse(T1, T2)
        pFlips_paired3 = self.parallel_flip_path3(T1, T2)
        pFlips_paired31 = self.parallel_flip_path3_reverse(T1, T2)
        path_list = [pFlips_paired1,pFlips_paired2,pFlips_paired3,pFlips_paired11,pFlips_paired21,pFlips_paired31]
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]


        for round in pFlips_paired:
            round_temp = []
            for oneFlip in round:
                (p1, p2) = oneFlip
                oneFlip_temp = [p1, p2]
                round_temp.append(oneFlip_temp)

            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i

    def computeDistanceSum(self, centerT):
        tot_dist = 0
        for i in range(len(self.triangulations)):
            prev_pFlips_i = self.pFlips[i][:]
            self.pFlips[i] = []
            pFlips_paired1 = self.parallel_flip_path(self.triangulations[i], centerT)
            pFlips_paired11 = self.parallel_flip_path_reverse(self.triangulations[i], centerT)
            pFlips_paired2 = self.parallel_flip_path2(self.triangulations[i], centerT)
            pFlips_paired21 = self.parallel_flip_path2_reverse(self.triangulations[i], centerT)
            pFlips_paired3 = self.parallel_flip_path3(self.triangulations[i], centerT)
            pFlips_paired31 = self.parallel_flip_path3_reverse(self.triangulations[i], centerT)
            path_list = [pFlips_paired1,pFlips_paired2,pFlips_paired3,pFlips_paired11,pFlips_paired21,pFlips_paired31]
            opt_ind = np.argmin([len(x) for x in path_list])
            pFlips_paired = path_list[opt_ind]
            if len(prev_pFlips_i)<len(pFlips_paired):
                self.pFlips[i] = prev_pFlips_i[:]
                tot_dist+=len(prev_pFlips_i)
            else:
                for round in pFlips_paired:
                    round_temp = []
                    for oneFlip in round:
                        (p1, p2) = oneFlip
                        oneFlip_temp = [p1, p2]
                        round_temp.append(oneFlip_temp)
                    self.pFlips[i].append(round_temp)
                tot_dist+=len(pFlips_paired)
        self.dist = tot_dist


    def computePFDOnly(self, centerT):
        tot_dist = 0
        pF = []
        for i in range(len(self.triangulations)):
            pFi = []
            pFlips_paired1 = self.parallel_flip_path(self.triangulations[i], centerT)
            pFlips_paired2 = self.parallel_flip_path2(self.triangulations[i], centerT)
            if len(pFlips_paired1)<len(pFlips_paired2):
                pFlips_paired = pFlips_paired1
            else:
                pFlips_paired = pFlips_paired2
            for round in pFlips_paired:
                round_temp = []
                for oneFlip in round:
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
        end_step = 10000

        edges = list(T.edges)
        starting_edge_ind = 0
        random.shuffle(edges)

        while total_step < end_step:
            total_step += 1
            do_random = random.random() > 0.999 ** step

            if do_random or starting_edge_ind == len(edges):
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

                    flips.append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                for e in flips:
                    T.flip((e[0], e[1]))

                edges = list(T.edges)
                random.shuffle(edges)
                starting_edge_ind = 0

                pF, new_len = self.computePFDOnly(T)
                pF, new_len = self.random_compute_pfd2_only(pF)
                if total_best>new_len:
                    self.dist = new_len
                    self.pFlips = pF
                    self.WriteData()
                total_best = min(total_best, new_len)
                prev_len = new_len
                step = 0
                continue

            T1 = T.fast_copy()
            e = edges[starting_edge_ind]
            if not self.flippable(T1, e):
                starting_edge_ind += 1
                continue

            T1.flip(e)

            pF, new_len = self.computePFDOnly(T)
            pF, new_len = self.random_compute_pfd2_only(pF)
            if new_len <= prev_len:
                step = 0
                T = T1
                edges = list(T.edges)
                random.shuffle(edges)

                if new_len < prev_len:
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

    def random_compute_pfd(self):
        prev_len = self.dist
        prev_best = prev_len
        tri_num = 0

        while tri_num < len(self.triangulations):
            seq = self.pFlips[tri_num]
            if len(seq) == 0:
                tri_num += 1
                continue
            if seq == 1:
                tri_num += 1
                continue
            seq_iter = 1
            local_T:Triangulation = self.triangulations[tri_num].fast_copy()
            for e in seq[0]:
                local_T.flip(e)
            while seq_iter < len(seq):
                seq1  = self.compuePFS_total(self.triangulations[tri_num], local_T)
                seq2 = self.compuePFS_total(local_T, self.center)
                if len(seq1) + len(seq2) < len(seq):
                    self.pFlips[tri_num] = seq1 + seq2
                    print(f"[{self.instance_uid}] Updated: {prev_len} -> {prev_len+len(seq1)+len(seq2)-len(seq)}")
                    prev_len = prev_len+len(seq1)+len(seq2)-len(seq)
                    break
                else:
                    for e in seq[seq_iter]:
                        local_T.flip(e)
                    seq_iter += 1
            if seq_iter == len(seq):
                tri_num += 1
        if prev_len < prev_best:
            total_dist = sum([len(pFlip) for pFlip in self.pFlips])
            self.dist = total_dist
            self.WriteData()
            print(f"[{self.instance_uid}] Improved: {prev_best} -> {total_dist}")
        return self.center

    def random_compute_pfd2(self,debug=True, tri_num=-1):
        print(f"Random compute pfd2 for {self.instance_uid}")
        prev_len = self.dist
        prev_best = prev_len
        if tri_num==-1:
            tri_num = len(self.triangulations)-1

        from multiprocessing import Pool
        while tri_num < len(self.triangulations) and 0<=tri_num:
            seq = self.pFlips[tri_num]
            if len(seq) == 0:
                tri_num -= 1
                continue
            if len(seq) == 1:
                tri_num -= 1
                continue
            seq_iter = 1

            T1_ind = 0
            T2_ind = 1
            if debug:
                print(f"[{self.instance_uid}] Processing triangulation {tri_num}")
            while T2_ind<len(seq):

                local_T1:Triangulation = self.triangulations[tri_num].fast_copy()
                for ind1 in range(T1_ind):
                    for e in seq[ind1]:
                        local_T1.flip(e)
                local_T2:Triangulation = local_T1.fast_copy()
                for ind2 in range(T1_ind, T2_ind):
                    for e in seq[ind2]:
                        local_T2.flip(e)
                seq1  = self.compuePFS_total(local_T1, local_T2)
                if len(seq1)< T2_ind - T1_ind:
                    self.pFlips[tri_num] = self.pFlips[tri_num][0:T1_ind] + seq1 + self.pFlips[tri_num][T2_ind:]
                    print(f"[{self.instance_uid}] Updated: {prev_len} -> {prev_len+len(seq1)- (T2_ind - T1_ind)}")
                    prev_len = prev_len+len(seq1)- (T2_ind - T1_ind)
                    self.dist = sum([len(pFlip) for pFlip in self.pFlips])
                    self.WriteData()
                    T1_ind = 0
                    T2_ind = 1
                    break
                else:
                    if T1_ind + 1 < T2_ind:
                        T1_ind += 1
                    else:
                        T2_ind += 1
                        T1_ind = 0
            if T2_ind==len(seq):
                tri_num -= 1
        if prev_len < prev_best:
            total_dist = sum([len(pFlip) for pFlip in self.pFlips])
            self.dist = total_dist
            self.WriteData()
            print(f"[{self.instance_uid}] Improved: {prev_best} -> {total_dist}")
        print(f"[{self.instance_uid}] Done")
        return self.center

    def random_compute_pfd2_update_new(self,debug=False):
        # print(f"Random compute pfd2 for {self.instance_uid}")
        prev_len = self.dist
        prev_best = prev_len
        tri_num = len(self.triangulations)-1
        while tri_num < len(self.triangulations) and 0<=tri_num:
            seq = self.pFlips[tri_num]
            if len(seq) == 0:
                tri_num -= 1
                continue
            if len(seq) == 1:
                tri_num -= 1
                continue
            seq_iter = 1

            T1_ind = 0
            T2_ind = 1
            if debug:
                print(f"[{self.instance_uid}] Processing triangulation {tri_num}")
            while T2_ind<len(seq):

                # print(tri_num, T1_ind, T2_ind)
                local_T1:Triangulation = self.triangulations[tri_num].fast_copy()
                for ind1 in range(T1_ind):
                    for e in seq[ind1]:
                        local_T1.flip(e)
                local_T2:Triangulation = local_T1.fast_copy()
                for ind2 in range(T1_ind, T2_ind):
                    for e in seq[ind2]:
                        local_T2.flip(e)
                seq1  = self.compuePFS_total(local_T1, local_T2)
                if len(seq1)< T2_ind - T1_ind:
                    self.pFlips[tri_num] = self.pFlips[tri_num][0:T1_ind] + seq1 + self.pFlips[tri_num][T2_ind:]
                    print(f"[{self.instance_uid}] Updated: {prev_len} -> {prev_len+len(seq1)- (T2_ind - T1_ind)}")
                    prev_len = prev_len+len(seq1)- (T2_ind - T1_ind)
                    self.dist = sum([len(pFlip) for pFlip in self.pFlips])
                    self.WriteData()
                    T1_ind = 0
                    T2_ind = 1
                    break
                else:
                    if len(seq1)== T2_ind - T1_ind:
                        self.pFlips[tri_num] = self.pFlips[tri_num][0:T1_ind] + seq1 + self.pFlips[tri_num][T2_ind:]
                        # print(f"[{self.instance_uid}] Updated: {prev_len} -> {prev_len+len(seq1)- (T2_ind - T1_ind)}")
                        prev_len = prev_len+len(seq1)- (T2_ind - T1_ind)
                        self.dist = sum([len(pFlip) for pFlip in self.pFlips])
                        self.WriteData()
                    if T1_ind + 1 < T2_ind:
                        T1_ind += 1
                        # print(tri_num, T1_ind, T2_ind, len(seq))
                    else:
                        T2_ind += 1
                        T1_ind = 0

            if T2_ind==len(seq):
                tri_num -= 1
        if prev_len < prev_best:
            total_dist = sum([len(pFlip) for pFlip in self.pFlips])
            self.dist = total_dist
            self.WriteData()
            print(f"[{self.instance_uid}] Improved: {prev_best} -> {total_dist}")

        return self.center

    def random_compute_pfd2_parallel(self, max_workers=None, debug=False):
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        import numpy as np

        pts_x = np.asarray(self.pts_x, dtype=np.float64)
        pts_y = np.asarray(self.pts_y, dtype=np.float64)

        shm_x = shm_create_from_numpy(pts_x, name="PTS_X_SHARED")
        shm_y = shm_create_from_numpy(pts_y, name="PTS_Y_SHARED")
        shm_meta = {
            "x": {"name": shm_x.name, "shape": pts_x.shape, "dtype": str(pts_x.dtype)},
            "y": {"name": shm_y.name, "shape": pts_y.shape, "dtype": str(pts_y.dtype)},
        }

        try:
            tasks = []
            for tri_num, (tri, seq) in enumerate(zip(self.triangulations, self.pFlips)):
                if seq and len(seq) > 1:
                    base_edges = list(tri.edges)
                    payload = _pickle.dumps((tri_num, base_edges, tuple(tuple(map(tuple, seq))), shm_meta, 64))
                    tasks.append((tri_num, payload))

            if not tasks:
                return self.center

            try:
                ctx = get_context('forkserver')
            except Exception:
                ctx = get_context('spawn')

            results = {}
            with ProcessPoolExecutor(max_workers=(max_workers or 4), mp_context=ctx) as ex:
                futs = {ex.submit(_pfd2_single_triangulation_worker, payload): tri_num for tri_num, payload in tasks}
                for fut in as_completed(futs):
                    tri_num = futs[fut]
                    out = fut.result()
                    tri_num2, new_seq, delta, improved = _pickle.loads(out)
                    assert tri_num2 == tri_num
                    if improved:
                        self.pFlips[tri_num] = [list(p) for p in new_seq]
            self.dist = sum(len(p) for p in self.pFlips)
            self.WriteData()
            return self.center

        finally:
            shm_x.close(); shm_y.close()
            shm_x.unlink(); shm_y.unlink()


    def random_compute_pfd2_only(self, pFlips):
        prev_len = sum([len(pFlip) for pFlip in pFlips])
        prev_best = prev_len
        tri_num = 0

        while tri_num < len(self.triangulations):
            seq = pFlips[tri_num]
            if len(seq) == 0:
                tri_num += 1
                continue
            if len(seq) == 1:
                tri_num += 1
                continue
            seq_iter = 1

            T1_ind = 0
            T2_ind = 1
            while T2_ind<len(seq):
                local_T1:Triangulation = self.triangulations[tri_num].fast_copy()
                for ind1 in range(T1_ind):
                    for e in seq[ind1]:
                        local_T1.flip(e)
                local_T2:Triangulation = local_T1.fast_copy()
                for ind2 in range(T1_ind, T2_ind):
                    for e in seq[ind2]:
                        local_T2.flip(e)
                seq1  = self.compuePFS_total(local_T1, local_T2)
                if len(seq1)< T2_ind - T1_ind:
                    pFlips[tri_num] = pFlips[tri_num][0:T1_ind] + seq1 + pFlips[tri_num][T2_ind:]
                    prev_len = prev_len+len(seq1)- (T2_ind - T1_ind)
                    T1_ind = 0
                    T2_ind = 1
                    break
                else:
                    if T1_ind + 1 < T2_ind:
                        T1_ind += 1
                    else:
                        T2_ind += 1
                        T1_ind = 0

            if T2_ind==len(seq):
                tri_num += 1
        total_dist = sum([len(pFlip) for pFlip in pFlips])

        return pFlips, total_dist

    def _tri_fingerprint(self, tri):

        try:

            edges = getattr(tri, "edges", None)
            if callable(edges):
                E = edges()
            else:
                E = tri.edges
            norm = []
            for (a, b) in E:
                norm.append((a, b) if a <= b else (b, a))
            norm.sort()
            return hash(tuple(norm))
        except Exception:
            if hasattr(tri, "to_signature"):
                return hash(tri.to_signature())
            return id(tri)

    def _build_prefix_states(self, base_tri, seq):
        prefix = [base_tri.fast_copy()]
        cur = prefix[0]
        for i in range(len(seq)):
            nxt = cur.fast_copy()
            for e in seq[i]:
                nxt.flip(e)
            prefix.append(nxt)
            cur = nxt
        return prefix

    def _rebuild_prefix_from(self, prefix, start_i, seq):
        cur = prefix[start_i]
        del prefix[start_i+1:]
        for i in range(start_i, len(seq)):
            nxt = cur.fast_copy()
            for e in seq[i]:
                nxt.flip(e)
            prefix.append(nxt)
            cur = nxt

    def random_compute_pfd3(self):
        print(f"Random compute pfd2 for {self.instance_uid}")

        prev_total_len = self.dist
        prev_best = prev_total_len

        pfs_cache = {}

        tri_num = 0
        updates_since_write = 0

        while tri_num < len(self.triangulations):
            base_tri = self.triangulations[tri_num]
            seq = self.pFlips[tri_num]

            if len(seq) <= 1:
                tri_num += 1
                continue

            prefix = self._build_prefix_states(base_tri, seq)

            T1_ind = 0
            T2_ind = 1

            improved_here = False

            while T2_ind < len(seq):
                local_T1 = prefix[T1_ind]
                local_T2 = prefix[T2_ind]

                f1 = self._tri_fingerprint(local_T1)
                f2 = self._tri_fingerprint(local_T2)
                key = (tri_num, f1, f2)

                if key in pfs_cache:
                    seq1 = pfs_cache[key]
                else:
                    seq1 = self.compuePFS_total(local_T1.fast_copy(), local_T2.fast_copy())
                    pfs_cache[key] = seq1

                old_len = (T2_ind - T1_ind)
                new_len = len(seq1)

                if new_len < old_len:
                    seq = self.pFlips[tri_num] = seq[0:T1_ind] + seq1 + seq[T2_ind:]

                    self._rebuild_prefix_from(prefix, T1_ind, seq)

                    delta = new_len - old_len
                    prev_total_len += delta
                    updates_since_write += 1
                    improved_here = True

                    if updates_since_write >= 1:
                        print(f"[{self.instance_uid}] Updated: {prev_total_len - delta} -> {prev_total_len}")
                        self.dist = sum(len(p) for p in self.pFlips)
                        self.WriteData()
                        updates_since_write = 0

                    T1_ind = 0
                    T2_ind = 1
                    continue
                else:
                    if T1_ind + 1 < T2_ind:
                        T1_ind += 1
                    else:
                        T2_ind += 1
                        T1_ind = 0

            if improved_here:
                self.dist = sum(len(p) for p in self.pFlips)
                print(f"[{self.instance_uid}] Partial improved → dist now {self.dist}")

            tri_num += 1

        if updates_since_write > 0:
            self.dist = sum(len(p) for p in self.pFlips)
            self.WriteData()

        if self.dist < prev_best:
            print(f"[{self.instance_uid}] Improved: {prev_best} -> {self.dist}")

        return self.center

    def computeDistanceSum2(self, centerT):

        start = time.time()

        print(self.pFlips)
        print(len(self.pFlips))
        print(len(self.triangulations))

        for i in range(len(self.triangulations)):

            self.pFlips[i] = []

            pFlips_paired = self.parallel_flip_path2(self.triangulations[i], centerT)

            for round in pFlips_paired:

                round_temp = []

                for oneFlip in round:

                    (p1, p2), (p3, p4) = oneFlip

                    oneFlip_temp = [p1, p2]

                    round_temp.append(oneFlip_temp)

                self.pFlips[i].append(round_temp)



            print('parallel flip distance from the center to T', i, ':', len(self.pFlips[i]))

            end = time.time()
            print('time:', f"{end - start:.5f} sec")


def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
