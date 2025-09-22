import cv2
from heapq import *
import json
import math
import numpy as np
import os
from pathlib import Path
import pdb
import random

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Diag:
    def __init__(self, i:int, j:int):
        self.p1 = i
        self.p2 = j
        self.nei_pts = [None, None]
    
class Triangulation:
    def __init__(self, pts, T):
        self.pts = pts
        self.edges: dict = dict()
        self.make_triangulation(T)

    # def __init__(self, pts, edges:dict):
    #     self.pts = pts
    #     self.edges = edges

        
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
        used = [False]*len(self.pts)
        for i, e in enumerate(E):
            if self.is_convex_quad(e[0], e[1]):
                if e not in prev_use:
                    res.append(e)
                    used[e[0]] = True
                    used[e[1]] = True
                    used[self.edges[e].nei_pts[0]] = True
                    used[self.edges[e].nei_pts[1]] = True
                    break
                else:
                    second_best = e
        if not res:
            res.append(e)
            used[e[0]] = True
            used[e[1]] = True
            used[self.edges[e].nei_pts[0]] = True
            used[self.edges[e].nei_pts[1]] = True
        for j in range(len(E)):
            if self.is_convex_quad(e[0], e[1]):
                check_ind = [E[j][0],E[j][1],self.edges[E[j]].nei_pts[0],self.edges[E[j]].nei_pts[1]]
                check_use = [used[k] for k in check_ind]
                if True in check_use:
                    continue
                for ind in check_ind: used[ind] = True
                res.append(E[j])
        return res
        
    def flip(self, i, j):
        # (i,j) edge가 flip 가능하면 flip하고 새로 생긴 edge return, 아니면 (-1,-1) return
        i, j = min(i,j), max(i,j)
        if not self.is_convex_quad(i,j):
            return -1, -1
        edge:Diag = self.edges[(min(i,j),max(i,j))]
        d1, d2 = edge.nei_pts
        if self.edges[min(i,d1), max(i,d1)].nei_pts[i<d1]!=j: 
            print(f"Something Wrong!! {i} {d1}")
            pdb.set_trace()
        self.edges[min(i,d1), max(i,d1)].nei_pts[i<d1] = d2
        if self.edges[min(d1,j), max(d1,j)].nei_pts[d1<j]!=i: 
            print(f"Something Wrong!! {d1} {j}")
            pdb.set_trace()
        self.edges[min(d1,j), max(d1,j)].nei_pts[d1<j] = d2
        if self.edges[min(j,d2), max(j,d2)].nei_pts[j<d2]!=i: 
            print(f"Something Wrong!! {j} {d2}")
            pdb.set_trace()
        self.edges[min(j,d2), max(j,d2)].nei_pts[j<d2] = d1
        if self.edges[min(d2,i), max(d2,i)].nei_pts[d2<i]!=j: 
            print(f"Something Wrong!! {d2} {i}")
            pdb.set_trace()
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
        return (list(e1-e2), list(e2-e1))
        
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
        self.ReadData()

    def ReadData(self):
        print("--------------------ReadData--------------------")
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
        for i in range(len(self.triangulations)):
            self.DrawTriangulation(self.triangulations[i], name = f"{i}")
        T:Triangulation = self.triangulations[0]
        step = 0
        res_e_list = []
        while True:
            step+=1
            E1, E2 = T.find_difference(self.triangulations[1])
            if not E1:
                break
            e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
            if not e_list:
                pdb.set_trace()
            res_e_list = []
            for e in e_list:
                res_e_list.append(T.flip(e[0],e[1]))
            self.DrawTriangulation(T, colored_edges=res_e_list,name=f"step {step}")
            self.DrawTriangulation(T, colored_edges=res_e_list,name=f"check")
            # pdb.set_trace()
        # for e in self.triangulations[0].edges.values():
        #     print(f"edge {e.p1} {e.p2}: {e.nei_pts[0]}, {e.nei_pts[1]}")
        
    def WriteData(self):
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