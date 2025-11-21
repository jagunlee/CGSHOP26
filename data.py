from collections import defaultdict
import copy
import cv2
import datetime
from heapq import *
import json
import math
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import pdb
import random
from scipy.sparse import coo_matrix
from shapely.geometry import LineString
from shapely.strtree import STRtree
import time

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Diag:
    def __init__(self, i:int, j:int):
        self.p1 = i
        self.p2 = j
        self.nei_pts = [None, None]
    def pts(self):
        return (self.p1, self.p2)
    
class Triangulation:
    def __init__(self, pts, T):
        self.pts = pts
        self.edges: dict = dict()
        self.make_triangulation(T)
        self.times = dict()

    # def __init__(self, pts, edges:dict):
    #     self.pts = pts
    #     self.edges = edges

    def return_edge(self):
        edge_list = []
        for e in self.edges.keys():
            edge_list.append(list(e))
        return edge_list

    def make_triangulation(self, T):
        edges = dict()
        for e in T:
            edges[(min(e[0],e[1]),max(e[0],e[1]))] = Diag(min(e[0],e[1]),max(e[0],e[1]))
        nei_dict = dict()
        for i in range(len(self.pts)):
            nei_dict[i] = []
        for e in T:
            nei_dict[e[0]].append(e[1])
            nei_dict[e[1]].append(e[0])
        for i in range(len(self.pts)):
            nei_dict[i], _, found = self.sort_cw_with_half_circle(nei_dict[i], i)
            for j in range(len(nei_dict[i])-1):
                if i<nei_dict[i][j]:
                    edges[(i,nei_dict[i][j])].nei_pts[1] = nei_dict[i][j+1]
                else:
                    edges[(nei_dict[i][j],i)].nei_pts[0] = nei_dict[i][j+1]
            if not found:
                if i<nei_dict[i][len(nei_dict[i])-1]:
                    edges[(i,nei_dict[i][len(nei_dict[i])-1])].nei_pts[1] = nei_dict[i][0]
                else:
                    edges[(nei_dict[i][len(nei_dict[i])-1],i)].nei_pts[0] = nei_dict[i][0]
        self.edges = edges
        
    def sort_cw_with_half_circle(self, pts, center):
        cx, cy = self.pts[center].x, self.pts[center].y
        with_angles = [(pt, math.atan2(self.pts[pt].y - cy, self.pts[pt].x - cx)) for pt in pts]
        with_angles.sort(key=lambda pa: -pa[1])
        n = len(with_angles)
        angles = [a for _, a in with_angles]
        best_start = 0
        found = False
        doubled = angles + [a - 2*math.pi for a in angles]  # wrap-around
        for i in range(n):
            j = i + n - 1
            if doubled[i] - doubled[j] <= math.pi + 1e-12:
                best_start = i
                found = True
                break
        with_angles = with_angles+with_angles
        if found:
            reordered = with_angles[best_start:best_start+n]
            reordered_angles = [a if a <= math.pi else a - 2*math.pi for _, a in reordered]
        else:
            reordered = with_angles
            reordered_angles = angles
        reordered_pts = [pt for pt, _ in reordered]
        return reordered_pts, reordered_angles, found

        
    def intersect(self, d11, d12, d21, d22):
        def _orient(a, b, c) -> float:
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        # (d11,d12)와 (d21,d22)가 intersect하는지 확인
        EPS = 1e-9
        p1 = self.pts[d11]
        p2 = self.pts[d12]
        p3 = self.pts[d21]
        p4 = self.pts[d22]
        o1 = _orient(p1, p2, p3)
        o2 = _orient(p1, p2, p4)
        o3 = _orient(p3, p4, p1)
        o4 = _orient(p3, p4, p2)

        # 일반적인 교차 (서로 다른 편에 위치)
        if (o1 * o2 < -EPS) and (o3 * o4 < -EPS):
            return True
        return False
            # x1 = ((p2.y-p1.y)*(p4.x-p3.x)*p1.x-(p2.x-p1.x)*(p4.x-p3.x)*p1.y-(p4.y-p3.y)*(p2.x-p1.x)*p3.x+(p4.x-p3.x)*(p2.x-p1.x)*p3.y)/((p2.y-p1.y)*(p4.x-p3.x)-(p4.y-p3.y)*(p2.x-p1.x))
        # if p1.x<x1 and x1<p2.x:
        #     return True
        # return False
    
    def is_convex_quad(self, i, j):
        # (i,j) edge가 실제로 존재하고 flip 가능한지 확인
        i, j = min(i,j), max(i,j)
        try:
            edge:Diag = self.edges[(i,j)]
            d1, d2 = edge.nei_pts
            if d1==None or d2==None:
                return False
            return self.intersect(i, j, d1, d2)
            
        except:
            return False
        
    def maximal_disjoint_convex_quad(self, E:list[(int, int)], prev_use=[]):
        res = []
        second_best = None
        used = [[False]*len(self.pts) for _ in range(len(self.pts))]
        if not E: return res
        for i, e in enumerate(E):
            if self.is_convex_quad(e[0], e[1]):
                if e not in prev_use:
                    res.append(e)
                    used[e[0]][e[1]] = True
                    used[e[1]][e[0]] = True
                    used[e[0]][self.edges[e].nei_pts[0]] = True
                    used[e[0]][self.edges[e].nei_pts[1]] = True
                    used[e[1]][self.edges[e].nei_pts[0]] = True
                    used[e[1]][self.edges[e].nei_pts[1]] = True
                    used[self.edges[e].nei_pts[0]][e[0]] = True
                    used[self.edges[e].nei_pts[1]][e[0]] = True
                    used[self.edges[e].nei_pts[0]][e[1]] = True
                    used[self.edges[e].nei_pts[1]][e[1]] = True
                    break
                else:
                    second_best = e
        if not res:
            e = second_best
            if e==None: return res
            res.append(e)
            used[e[0]][e[1]] = True
            used[e[1]][e[0]] = True
            used[e[0]][self.edges[e].nei_pts[0]] = True
            used[e[0]][self.edges[e].nei_pts[1]] = True
            used[e[1]][self.edges[e].nei_pts[0]] = True
            used[e[1]][self.edges[e].nei_pts[1]] = True
            used[self.edges[e].nei_pts[0]][e[0]] = True
            used[self.edges[e].nei_pts[1]][e[0]] = True
            used[self.edges[e].nei_pts[0]][e[1]] = True
            used[self.edges[e].nei_pts[1]][e[1]] = True
            for j in range(len(E)):
                e = E[j]
                if self.is_convex_quad(e[0], e[1]):
                    if used[e[0]][e[1]]==True:
                        continue
                    used[e[0]][e[1]] = True
                    used[e[1]][e[0]] = True
                    used[e[0]][self.edges[e].nei_pts[0]] = True
                    used[e[0]][self.edges[e].nei_pts[1]] = True
                    used[e[1]][self.edges[e].nei_pts[0]] = True
                    used[e[1]][self.edges[e].nei_pts[1]] = True
                    used[self.edges[e].nei_pts[0]][e[0]] = True
                    used[self.edges[e].nei_pts[1]][e[0]] = True
                    used[self.edges[e].nei_pts[0]][e[1]] = True
                    used[self.edges[e].nei_pts[1]][e[1]] = True
                    res.append(E[j])
        else:
            for j in range(len(E)):
                e = E[j]
                if e not in prev_use:
                    if self.is_convex_quad(e[0], e[1]):
                        if used[e[0]][e[1]]==True:
                            continue
                        used[e[0]][e[1]] = True
                        used[e[1]][e[0]] = True
                        used[e[0]][self.edges[e].nei_pts[0]] = True
                        used[e[0]][self.edges[e].nei_pts[1]] = True
                        used[e[1]][self.edges[e].nei_pts[0]] = True
                        used[e[1]][self.edges[e].nei_pts[1]] = True
                        used[self.edges[e].nei_pts[0]][e[0]] = True
                        used[self.edges[e].nei_pts[1]][e[0]] = True
                        used[self.edges[e].nei_pts[0]][e[1]] = True
                        used[self.edges[e].nei_pts[1]][e[1]] = True
                        res.append(E[j])
            for j in range(len(E)):
                e = E[j]
                if self.is_convex_quad(e[0], e[1]):
                    if used[e[0]][e[1]]==True:
                        continue
                    used[e[0]][e[1]] = True
                    used[e[1]][e[0]] = True
                    used[e[0]][self.edges[e].nei_pts[0]] = True
                    used[e[0]][self.edges[e].nei_pts[1]] = True
                    used[e[1]][self.edges[e].nei_pts[0]] = True
                    used[e[1]][self.edges[e].nei_pts[1]] = True
                    used[self.edges[e].nei_pts[0]][e[0]] = True
                    used[self.edges[e].nei_pts[1]][e[0]] = True
                    used[self.edges[e].nei_pts[0]][e[1]] = True
                    used[self.edges[e].nei_pts[1]][e[1]] = True
                    res.append(E[j])
                
        return res
        
    def flip(self, i, j):
        # (i,j) edge가 flip 가능하면 flip하고 새로 생긴 edge return, 아니면 (-1,-1) return
        i, j = min(i,j), max(i,j)
        if not self.is_convex_quad(i,j):
            return -1, -1
        edge:Diag = self.edges[(min(i,j),max(i,j))]
        d1, d2 = edge.nei_pts
        self.edges[min(i,d1), max(i,d1)].nei_pts[i<d1] = d2
        self.edges[min(d1,j), max(d1,j)].nei_pts[d1<j] = d2
        self.edges[min(j,d2), max(j,d2)].nei_pts[j<d2] = d1
        self.edges[min(d2,i), max(d2,i)].nei_pts[d2<i] = d1
        
        if d1<d2:
            self.edges[d1, d2] = Diag(d1, d2)
            self.edges[d1, d2].nei_pts = [j,i]
        else:
            self.edges[d2, d1] = Diag(d2, d1)
            self.edges[d2, d1].nei_pts = [i,j]
        del self.edges[(min(i,j),max(i,j))]
        return min(d1, d2),max(d1, d2)
        
    def find_difference(self, T):
        e1 = set(self.edges.keys())
        e2 = set(T.edges.keys())
        def intersect_num(e, l):
            n = 0
            for e1 in l:
                if self.intersect(e[0],e[1],e1[0],e1[1]): n+=1
            return n
        l1,l2 = list(e1-e2), list(e2-e1)
        _l1, _l2 = l1[:], l2[:]
        for i in range(len(l1)):
            l1[i] = (intersect_num(l1[i],_l2), l1[i])
        for i in range(len(l2)):
            l2[i] = (intersect_num(l2[i],_l1), l2[i])
        l1.sort(reverse=True)
        l2.sort(reverse=True)
        l1 = [x[1] for x in l1]
        l2 = [x[1] for x in l2]
        return (l1, l2)
        
    def check_1pfd(self, T):
        E1, E2 = self.find_difference(T)
        E1_in = self.maximal_disjoint_convex_quad(E1)
        if len(E1_in)==len(E1): 
            return True
        return False
        # T와 1 parallel flip distance인지 확인
        
    
class Data:
    def __init__(self, input):
        try:
            base = Path(__file__).resolve()
        except NameError:
            base = Path(os.getcwd()).resolve()
        self.application_path = str(base.parents[0])
        self.input = input
        self.df = None
        # self.compute_intersect()
        self.ReadData()
        

    def ReadData(self):
        print("--------------------ReadData--------------------")
        if "solution" not in self.input:
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                print(f"instance: {self.instance_uid}")
                pts_x = root["points_x"]
                pts_y = root["points_y"]
                self.pts = []
                for i in range(len(pts_y)):
                    self.pts.append(Point(pts_x[i], pts_y[i]))
                self.triangulations = []
                Ts = root["triangulations"]
                for T in Ts:
                    self.triangulations.append(Triangulation(self.pts, T))
                print(f"num of pts: {len(self.pts)}")
                print(f"num of triangulations: {len(self.triangulations)}")
            # for i in range(len(self.triangulations)):
            #     self.DrawTriangulation(self.triangulations[i], name = f"{i}")
            max_pfd = 0
            inp = []
            for i in range((len(self.triangulations))-1):
                for j in range(i+1, len(self.triangulations)):
                    inp.append((i,j))
            # print(inp)
            _multi = False
            _ini_sol = False
            initial_sol = [0]*len(self.triangulations)
            self.center = self.triangulations[0]
            self.dist = float("INF")
            self.flip = [[] for _ in range(len(self.triangulations))]
            if _ini_sol:
                if _multi:
                    with Pool() as pool:
                        res = pool.starmap(self.compute_pfd, inp)
                    for res1 in res:
                        initial_sol[res1[2]]+=res1[0]
                        initial_sol[res1[3]]+=res1[0]
                    max_pfd = max(res, key=lambda x:x[0])
                    print(f"Maximum Parallel flip distance: {max_pfd[0]}")
                else:
                    for i in range((len(self.triangulations))-1):
                        for j in range(i+1, len(self.triangulations)):
                            pfd, *_ = self.compute_pfd(i,j)
                            max_pfd = max(max_pfd, pfd)
                            initial_sol[i]+=pfd
                            initial_sol[j]+=pfd
                    print(f"Maximum Parallel flip distance: {max_pfd}")
                print(f"Initial Center: {np.argmin(initial_sol)} (total dist: {min(initial_sol)})")
                self.center = self.triangulations[np.argmin(initial_sol)]
                _, self.flip = self.compute_center_dist(self.center)
                self.WriteData()
        else:
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                print(f"instance: {self.instance_uid}")
                self.flip = root["flips"]
                self.dist = sum([len(x) for x in self.flip])
                org_input = root["meta"]["input"]
            self.input = org_input
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                pts_x = root["points_x"]
                pts_y = root["points_y"]
                self.pts = []
                for i in range(len(pts_y)):
                    self.pts.append(Point(pts_x[i], pts_y[i]))
                self.triangulations = []
                Ts = root["triangulations"]
                for T in Ts:
                    self.triangulations.append(Triangulation(self.pts, T))
                print(f"num of pts: {len(self.pts)}")
                print(f"num of triangulations: {len(self.triangulations)}")
            min_flip_ind = np.argmin([len(x) for x in self.flip])
            self.center =  copy.deepcopy(self.triangulations[min_flip_ind])
            for flip_seq in self.flip[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])


        
        

    def intersect(self, d11, d12, d21, d22):
        def _orient(a, b, c) -> float:
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        # (d11,d12)와 (d21,d22)가 intersect하는지 확인
        EPS = 1e-9
        p1 = self.pts[d11]
        p2 = self.pts[d12]
        p3 = self.pts[d21]
        p4 = self.pts[d22]
        o1 = _orient(p1, p2, p3)
        o2 = _orient(p1, p2, p4)
        o3 = _orient(p3, p4, p1)
        o4 = _orient(p3, p4, p2)

        # 일반적인 교차 (서로 다른 편에 위치)
        if (o1 * o2 < -EPS) and (o3 * o4 < -EPS):
            return True
        return False
    
    def find_center_np(self, debug=False):
        step = 0
        T = [copy.deepcopy(t) for t in self.triangulations]
        res_e_lists = [[] for _ in range(len(self.triangulations))]
        edge_to_idx = {}
        idx_to_edge = []
        tri_edges = []
        for t in T:
            idxs = []
            for e in t.edges:
                u, v = e[0], e[1]
                key = (u, v) if u < v else (v, u)
                if key not in edge_to_idx:
                    edge_to_idx[key] = len(edge_to_idx)
                    idx_to_edge.append(key)
                idxs.append(edge_to_idx[key])
            tri_edges.append(np.array(idxs, dtype=np.int32))
        M = len(edge_to_idx)
        usage = np.zeros(M, dtype=np.int32)      # exs_num
        inter = np.zeros(M, dtype=np.int32)      # inter_num
        x1 = np.empty(M, dtype=np.float64)
        y1 = np.empty(M, dtype=np.float64)
        x2 = np.empty(M, dtype=np.float64)
        y2 = np.empty(M, dtype=np.float64)
        for (u, v), idx in edge_to_idx.items():
            x1[idx], y1[idx] = self.pts[u].x, self.pts[u].y
            x2[idx], y2[idx] = self.pts[v].x, self.pts[v].y
        segments = [LineString([(x1[i], y1[i]), (x2[i], y2[i])]) for i in range(M)]
        tree = STRtree(segments)
        for edges in tri_edges:
            usage[edges] += 1
        neighbors = [set() for _ in range(M)]
        for i,seg in enumerate(segments):
            cand = tree.query(seg)
            for j in cand:
                # if i>=j:
                #     continue
                if idx_to_edge[i][0]==idx_to_edge[j][0] or idx_to_edge[i][1]==idx_to_edge[j][0] or idx_to_edge[i][0]==idx_to_edge[j][1] or idx_to_edge[i][1]==idx_to_edge[j][1]:
                    continue
                if seg.intersects(segments[j]):
                    neighbors[i].add(int(j))
                    neighbors[j].add(int(i))
                    inter[i]+=usage[j]
                    inter[j]+=usage[i]
        # rows = []
        # cols = []
        # for i in range(M):
        #     for j in neighbors[i]:
        #         rows.append(i)
        #         cols.append(j)
        #         # rows.append(j)
        #         # cols.append(i)

        # data = np.ones(len(rows), dtype=np.int32)
        # A = coo_matrix((data, (rows, cols)), shape=(M, M)).tocsr()

        
        # inter = A.dot(usage)   # inter_num
        weight = (inter/2 - usage) / usage
        prev_weight = sum(weight)

        T_val = np.zeros(len(T), dtype=np.float64)
        for i, edges in enumerate(tri_edges):
            T_val[i] = weight[edges].sum()
        
        while True:
            if debug: print(T_val)
            # print(res_e_lists)
            if debug: 
                if step>60: break
            done = True
            set_list = [set(t.edges.keys()) for t in T]
            base_set = set_list[0]
            for i in range(1, len(T)):
                if base_set!=set_list[i]:
                    done = False
                    break
            if done:
                dist, flip = self.compute_center_dist(T[0])
                print(f"Total distance from center: {self.dist} -> {dist}")
                self.center = T[0]
                self.dist = dist
                self.flip = flip
                return T[0], dist
            # pdb.set_trace()
            step+=1
            # update_t_ind = -1
            # update_t_val = -float("INF")
            # flip_list = []
            # for t_ind, t in enumerate(T):
            #     t_val = 0
            #     t_edges = sorted(tri_edges[t_ind], key=lambda x:-weight[x])
            #     for i in range(len(t_edges)):
            #         if weight[t_edges[i]]<0:
            #             break
            #     update_e = [idx_to_edge[j] for j in t_edges[:i]]
            #     _flip_list = t.maximal_disjoint_convex_quad(update_e, res_e_lists[t_ind])
            #     for e in _flip_list:
            #         t_val+=weight[edge_to_idx[e]]
            #     if t_val>update_t_val:
            #         update_t_ind = t_ind
            #         update_t_val = t_val
            #         flip_list = _flip_list
                
            update_t_ind = np.argmax(T_val)
            t:Triangulation = T[update_t_ind]
            t_edges = sorted(tri_edges[update_t_ind], key=lambda x:-weight[x])
            for i in range(len(t_edges)):
                if weight[t_edges[i]]<0:
                    break
            update_e = [idx_to_edge[j] for j in t_edges[:i]]
            if not update_e: pdb.set_trace()
            flip_list = t.maximal_disjoint_convex_quad(update_e, res_e_lists[update_t_ind])
            local_res_list = []
            for e in flip_list:
                local_res_list.append(t.flip(e[0],e[1]))
            res_e_lists[update_t_ind] = local_res_list
            if debug: print(flip_list, local_res_list)
            # print(flip_list, local_res_list)
            removed_idx = []
            for e in flip_list:
                removed_idx.append(edge_to_idx[e])
            added_idx = []
            for e in local_res_list:
                if e in edge_to_idx:
                    added_idx.append(edge_to_idx[e])
                else:
                    new_idx = len(idx_to_edge)
                    edge_to_idx[e] = new_idx
                    idx_to_edge.append(e)
                    added_idx.append(new_idx)

                    usage = np.append(usage, 0)
                    inter = np.append(inter, 0)
                    neighbors.append(set())
                    for other_idx, other_key in enumerate(idx_to_edge[:-1]):
                        # if e[0]==other_key[0] or e[1]==other_key[0] or e[0]==other_key[1] or e[1]==other_key[1]:
                        #     continue
                        if self.intersect(e[0], e[1], other_key[0], other_key[1]):
                            neighbors[new_idx].add(int(other_idx))
                            neighbors[other_idx].add(int(new_idx))
                            inter[new_idx] += usage[other_idx]*2
            delta_usage = defaultdict(int)
            for idx in removed_idx:
                delta_usage[idx]-=1
            for idx in added_idx:
                delta_usage[idx]+=1

            for e_idx, d in delta_usage.items():
                if d==0:
                    continue
                usage[e_idx]+=d
                for f_idx in neighbors[e_idx]:
                    inter[f_idx]+=d*2
            tri_set = set(tri_edges[update_t_ind])
            for idx in removed_idx:
                if idx in tri_set:
                    tri_set.remove(idx)
            for idx in added_idx:
                tri_set.add(idx)
            tri_edges[update_t_ind] = list(tri_set)
            if debug:
                for edge in t.edges:
                    if edge_to_idx[edge] not in tri_edges[update_t_ind]:
                        pdb.set_trace()
                for e_ind in tri_edges[update_t_ind]:
                    if idx_to_edge[e_ind] not in t.edges:
                        pdb.set_trace()


            weight = np.full_like(usage, fill_value=0, dtype=np.float64)
            mask = (usage > 0)
            weight[mask] = (inter[mask]/2 - usage[mask]) / usage[mask]
            T_val = np.zeros(len(T), dtype=np.float64)
            for i, edges in enumerate(tri_edges):
                T_val[i] = weight[edges].sum()
            if debug:
                e_list = dict()
                for t in T:
                    for e in t.edges:
                        if e in e_list.keys():
                            e_list[e][0]+=1
                        else:
                            e_list[e] = [1,0]
                for e1 in e_list.keys():
                    for e2 in e_list.keys():
                        if e1==e2: continue
                        if self.intersect(e1[0], e1[1], e2[0], e2[1]):
                            e_list[e1][1]+=e_list[e2][0]
                            e_list[e2][1]+=e_list[e1][0]
                for e in e_list.keys():
                    e_ind = edge_to_idx[e]
                    if e_list[e][0]!=usage[e_ind]: pdb.set_trace()
                    if e_list[e][1]!=inter[e_ind]: pdb.set_trace()
            # _T_val = [0]*len(T)
            # for i, t in enumerate(T):
            #     for e in t.edges:
            #         exs_num, inter_num = e_list[e]
            #         _T_val[i]+=(inter_num-exs_num)/exs_num
            print(f"[{self.instance_uid}, {step} step] Triangulation {update_t_ind} flipped, {len(local_res_list)} edges")
        # dist, flip = self.compute_center_dist(T[0])
        # print(f"Total distance from center: {self.dist} -> {dist}")
        # self.center = T[0]
        # self.dist = dist
        # self.flip = flip
        # return T[0], dist


    def find_center(self):
        T = [copy.deepcopy(t) for t in self.triangulations]
        res_e_lists = [[] for _ in range(len(self.triangulations))]
        step = 0
        while True:
            done = True
            set_list = [set(t.edges.keys()) for t in T]
            base_set = set_list[0]
            for i in range(1, len(T)):
                if base_set!=set_list[i]:
                    done = False
                    break
            if done:
                dist, flip = self.compute_center_dist(T[0])
                print(f"Total distance from center: {self.dist} -> {dist}")
                self.center = T[0]
                self.dist = dist
                self.flip = flip
                return T[0], dist
            step+=1
            e_list = dict()
            for t in T:
                for e in t.edges:
                    if e in e_list.keys():
                        e_list[e][0]+=1
                    else:
                        e_list[e] = [1,0]
            for e1 in e_list.keys():
                for e2 in e_list.keys():
                    if e1==e2: continue
                    if self.intersect(e1[0], e1[1], e2[0], e2[1]):
                        e_list[e1][1]+=e_list[e2][0]
                        e_list[e2][1]+=e_list[e1][0]
            T_val = [0]*len(T)
            for i, t in enumerate(T):
                for e in t.edges:
                    exs_num, inter_num = e_list[e]
                    T_val[i]+=(inter_num-exs_num)/exs_num
            print(T_val)
            update_t_ind = np.argmax(T_val)
            t:Triangulation = T[update_t_ind]
            update_e = list(t.edges.keys())
            update_e.sort(key = lambda x: (e_list[x][1]-e_list[x][0])/e_list[x][0], reverse=True)
            for i in range(len(update_e)):
                x = update_e[i]
                if (e_list[x][1]-e_list[x][0])/e_list[x][0]<0:
                    break
            # print(update_e, len(update_e), i)
            update_e = update_e[:i]
            flip_list = t.maximal_disjoint_convex_quad(update_e, res_e_lists[update_t_ind])
            
            local_res_list = []
            for e in flip_list:
                local_res_list.append(t.flip(e[0],e[1]))
            res_e_lists[update_t_ind] = local_res_list
            print(flip_list, local_res_list)
            print(f"[{self.instance_uid}, {step} step] Triangulation {update_t_ind} flipped, {len(local_res_list)} edges")
            

    def random_move(self):
        prev_len, old_flip = self.compute_center_dist(self.center)
        total_best = prev_len
        T:Triangulation = copy.deepcopy(self.center)
        print(f"Start with {prev_len}")
        step = 0
        total_step = 0
        end_step = 10*len(self.triangulations)*len(self.pts)
        edges = list(T.edges.keys())
        # print(edges)
        # pdb.set_trace()
        starting_edge_ind = 0
        random.shuffle(edges)
        while total_step<end_step:
            total_step+=1
            random_move = random.random()>0.999**step
            if random_move or starting_edge_ind==len(edges): 
                random.shuffle(edges)
                _e_list = T.maximal_disjoint_convex_quad(edges)
                random_choice = [random.random() for _ in range(len(_e_list))]
                e_list = []
                for i,e in enumerate(_e_list):
                    if random_choice[i]>0.5:
                        e_list.append(_e_list[i])

                for e in e_list:
                    T.flip(e[0], e[1])
                edges = list(T.edges.keys())
                random.shuffle(edges)
                starting_edge_ind = 0
                new_len, _ = self.compute_center_dist(T)
                total_best = min(total_best, new_len)
                # print(f"[{self.instance_uid} {total_step}/{end_step}] Random move! {prev_len}->{new_len} (total best: {total_best})")
                prev_len = new_len
                
                step = 0
            else:
                T1 = copy.deepcopy(T)
                e = edges[starting_edge_ind]
                if not T1.is_convex_quad(e[0], e[1]):
                    starting_edge_ind +=1
                    continue
                T1.flip(e[0], e[1])
                new_len, flip = self.compute_center_dist(T1)
                if new_len<=prev_len:
                    step = 0
                    T = copy.deepcopy(T1)
                    edges = list(T1.edges.keys())
                    random.shuffle(edges)
                    if new_len<prev_len:
                        self.center = copy.deepcopy(T)
                        self.dist = new_len
                        self.flip = flip
                        if new_len<total_best:
                            print(f"[{self.instance_uid} {total_step}/{end_step}] {total_best}->{new_len}")
                            total_best = min(new_len, total_best)
                            # print(f"[{self.instance_uid} {total_step}/{end_step}] {prev_len}->{new_len} (total best: {total_best})")
                            self.WriteData()
                        prev_len = new_len
                        
                    starting_edge_ind = 0
                else:
                    step+=1
                    starting_edge_ind+=1
        return self.center
    
        
    def compute_pfd(self, i, j):
        T, T1 = copy.deepcopy(self.triangulations[i]), copy.deepcopy(self.triangulations[j])
        step = 0
        res_e_list = []
        flip_list = []
        while True:
            E1, E2 = T.find_difference(T1)
            if not E1:
                break
            step+=1
            e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
            if not e_list:
                pdb.set_trace()
            res_e_list = []
            flip_iter = []
            for e in e_list:
                flip_iter.append([e[0],e[1]])
                res_e_list.append(T.flip(e[0],e[1]))
            flip_list.append(flip_iter)
            # self.DrawTriangulation(T, colored_edges=res_e_list,name=f"step {step}")
            # self.DrawTriangulation(T, colored_edges=res_e_list,name=f"check")
        print(f"{i} -> {j} can be done in {step} step!")
        return step, flip_list, i, j

    def compute_center_dist(self, T1:Triangulation):
        if not T1:
            return float("INF")
        total_length = 0
        flip = []
        for i,_T in enumerate(self.triangulations):
            T = copy.deepcopy(_T)
            step = 0
            res_e_list = []
            flip_list = []
            while True:
                E1, E2 = T.find_difference(T1)
                # print(E1)
                if not E1:
                    break
                step+=1
                e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
                if not e_list:
                    pdb.set_trace()
                res_e_list = []
                f_iter = []
                for e in e_list:
                    f_iter.append([e[0],e[1]])
                    res_e_list.append(T.flip(e[0],e[1]))
                flip_list.append(f_iter)
                # self.DrawTriangulation(T, colored_edges=res_e_list,name=f"step {step}")
                # self.DrawTriangulation(T, colored_edges=res_e_list,name=f"check")
            total_length+=step
            flip.append(flip_list)
            # print(f"{i} -> center can be done in {step} step!")
        return total_length, flip
    
    def WriteData(self):
        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_uid
        inst["flips"] = self.flip
        inst["meta"] = {"dist":self.dist, "input": self.input, "center": self.center.return_edge()}
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
                if old_score>self.dist:
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

            df.loc[len(df.index)-1, self.instance_uid] = min(df.loc[len(df.index)-1, self.instance_uid], self.dist)
            df.to_csv("result.csv")
            pass

    def DrawTriangulation(self, T, colored_edges = [], name = "", folder = ""):
        if name:
            name = "_" + name
        minx = min(list(p.x for p in self.pts))
        miny = min(list(p.y for p in self.pts))
        maxx = max(list(p.x for p in self.pts))
        maxy = max(list(p.y for p in self.pts))
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        lineType = cv2.LINE_AA
        width = int(maxx-minx)
        height = int(maxy-miny)
        rad = 1000/width
        add_size = 80
        width = int(width*rad)+add_size
        height = int(height*rad)+add_size
        minw = add_size//2-int(minx*rad)
        minh = height + int(miny*rad)-add_size//2
        img = np.zeros((height, width, 3),dtype="uint8")+255
        for e in T.edges.keys():
            if e in colored_edges:
                cv2.line(img, (minw+int(rad*self.pts[e[0]].x),minh-int(rad*self.pts[e[0]].y)), (minw+int(rad*self.pts[e[1]].x),minh-int(rad*self.pts[e[1]].y)), (0,0,255), 2)
            else:
                cv2.line(img, (minw+int(rad*self.pts[e[0]].x),minh-int(rad*self.pts[e[0]].y)), (minw+int(rad*self.pts[e[1]].x),minh-int(rad*self.pts[e[1]].y)), (0,0,0), 2)
        for i,p in enumerate(self.pts):
            cv2.circle(img, (minw+int(rad*p.x),minh-int(rad*p.y)), 5,(255,0,0),-1)
            cv2.putText(img, str(i), (minw+int(rad*p.x)+10,minh-int(rad*p.y)+10), fontFace, fontScale, (0,0,0), thickness, lineType)
        if folder:
            folder = os.path.join(self.application_path, "solutions")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(folder+"/"+self.instance_uid + ".triangulation" + name + ".png", img)
        else:
            loc = os.path.join(self.application_path, "solutions")
            # print("loc: "+loc)
            os.makedirs(loc, exist_ok=True)
            cv2.imwrite(loc+"/"+self.instance_uid + ".triangulation" + name + ".png", img)
        


    
    

def turn(p1:Point, p2:Point, p3:Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

def cw_key(center):

    cx, cy = center.x, center.y
    def key(pt):
        # 각도 계산
        angle = math.atan2(pt.y - cy, pt.x - cx)
        # atan2는 CCW 기준이므로 시계방향으로 바꾸려면 -angle 반환
        return -angle
    return key