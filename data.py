import json
import numpy as np
import cv2
import math
import os
from pathlib import Path

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Diag:
    def __init__(self, i:int, j:int):
        self.p1 = i
        self.p2 = j
        self.nei_pts = [None, None]
    
    
    
class Data:
    def __init__(self, input):
        try:
            base = Path(__file__).resolve()
        except NameError:
            base = Path(os.getcwd()).resolve()
        self.application_path = str(base.parents[1])
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
                self.triangulations.append(self.make_triangulation(T))
        
    def WriteData(self):
        pass

    def DrawTriangulation(self, T, colored_edges = [], name = "", folder = ""):
        if name:
            name = "_" + name
        minx = min(list(p.x for p in self.pts))
        miny = min(list(p.y for p in self.pts))
        maxx = max(list(p.x for p in self.pts))
        maxy = max(list(p.y for p in self.pts))
        width = int(maxx-minx)
        height = int(maxy-miny)
        rad = 1000/width
        width = int(width*rad)+40
        height = int(height*rad)+40
        minw = 20-int(minx*rad)
        minh = height + int(miny*rad)-20
        img = np.zeros((height, width, 3),dtype="uint8")+255
        for e in T:
            if (e[0]) in colored_edges:
                cv2.line(img, (minw+int(rad*self.pts[e[0]].x),minh-int(rad*self.pts[e[0]].y)), (minw+int(rad*self.pts[e[1]].x),minh-int(rad*self.pts[e[1]].y)), (0,0,255), 2)
            else:
                cv2.line(img, (minw+int(rad*self.pts[e[0]].x),minh-int(rad*self.pts[e[0]].y)), (minw+int(rad*self.pts[e[1]].x),minh-int(rad*self.pts[e[1]].y)), (0,0,0), 2)
        for p in self.pts:
            cv2.circle(img, (minw+int(rad*p.x),minh-int(rad*p.y)), 5,(255,0,0),-1)
        if folder:
            folder = os.path.join(self.application_path, "solutions")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(folder+"/"+self.instance_name + ".triangulation" + name + ".png", img)
        else:
            os.makedirs(os.path.join(self.application_path, "solutions"), exist_ok=True)
            cv2.imwrite("solutions/"+self.instance_name + ".triangulation" + name + ".png", img)


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
            nei_dict[i], _, found = self.sort_cw_with_half_circle(i, nei_dict[i])
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
        return list(edges.values())




        


    def intersect(self, d1:Diag, d2:Diag):
        p1 = self.pts[d1.p1]
        p2 = self.pts[d1.p2]
        p3 = self.pts[d2.p1]
        p4 = self.pts[d2.p2]
        x1 = ((p2.y-p1.y)*(p4.x-p3.x)*p1.x-(p2.x-p1.x)*(p4.x-p3.x)*p1.y-(p4.y-p3.y)*(p2.x-p1.x)*p3.x+(p4.x-p3.x)*(p2.x-p1.x)*p3.y)/((p2.y-p1.y)*(p4.x-p3.x)-(p4.y-p3.y)*(p2.x-p1.x))
        if p1.x<x1 and x1<p2.x:
            return True
        return False
    def sort_cw_with_half_circle(self, pts, center):

        cx, cy = self.pts[center].x, self.pts[center].y

        # 각도 계산 (atan2, 라디안)
        with_angles = [(pt, math.atan2(self.pts[pt].y - cy, self.pts[pt].x - cx)) for pt in pts]

        # 시계 방향 정렬 (angle 큰 것 -> 작은 것 순)
        with_angles.sort(key=lambda pa: -pa[1])

        n = len(with_angles)
        angles = [a for _, a in with_angles]

        # cyclic shift 탐색: 연속 n개 중 각도 범위 ≤ π 찾기
        best_start = 0
        found = False
        doubled = angles + [a + 2*math.pi for a in angles]  # wrap-around
        for i in range(n):
            j = i + n - 1
            if doubled[j] - doubled[i] <= math.pi + 1e-12:
                best_start = i
                found = True
                break

        if found:
            reordered = with_angles[best_start:best_start+n]
            reordered_angles = [a if a <= math.pi else a - 2*math.pi for _, a in reordered]
        else:
            # 불가능하다면 그냥 기본 CW 정렬 반환
            reordered = with_angles
            reordered_angles = angles

        reordered_pts = [pt for pt, _ in reordered]
        return reordered_pts, reordered_angles, found


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

p1 = Point(1,1)
p2 = Point(2,2)
p3 = Point(1,2)
p4 = Point(2,1)

d1 = Diag(p1, p2)
d2 = Diag(p3, p4)
d3 = Diag(p1, p3)
print(d1.intersect(d2))
print(d1.intersect(d3))