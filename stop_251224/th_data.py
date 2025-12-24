import json
import sys
import random
from th_Point import Point
from th_Triangulation import Triangle, Triangulation
from copy import deepcopy
# import cv2
import time
import os
import pandas as pd
import datetime
# from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
# from cgshop2026_pyutils.geometry import FlippableTriangulation
# from cgshop2026_pyutils.verify import check_for_errors
# import cgshop2026_pyutils

sys.setrecursionlimit(1000000)

class Edge:
    def __init__(self, ):
        self.endpoints = []

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
        # print("--------------------ReadData--------------------")
        f = open(self.input, "r", encoding="utf-8")
        root = json.load(f)
        # print(root)
        self.instance_name = root["instance_uid"]
        self.pts_x = root["points_x"]
        self.pts_y = root["points_y"]
        self.pts = []
        for i in range(len(self.pts_y)):
            self.pts.append(Point(self.pts_x[i], self.pts_y[i])) #hy: !!! This is pts in Data!!!! not in Triangulation
        #print("---ReadData: Triangulations")
        print(len(root["triangulations"]), ": ", end = " ")
        for t in root["triangulations"]:
            self.triangulations.append(self.make_triangulation(t))
            print(len(self.triangulations), end=" ")
        print()


        #print(len(self.triangulations))
        # self.distance = [] * len(self.triangulations)
        self.pFlips = [None] * len(self.triangulations)
        # print(len(self.pFlips))

    def make_triangulation(self, t: Triangulation):
        tri = Triangulation()
        graph = [[] for _ in range(len(self.pts))]
        for u, v in t:
            graph[u].append(v)
            graph[v].append(u)
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

    def resolve_cross_random(self):
        pass

    ##hy
    #def resolve_cross_fast(self, tri:Triangulation, con:tuple):
    #    q1 = con[0]
    #    q2 = con[1]
    #    # Find edges having q1 = (q1, a)
    #    # Find edges having q2 = (q2, b)
    #    list_a = [(set(key)-{q1}).pop() for key in tri.dict.keys() if q1 in key]
    #    list_b = [(set(key)-{q2}).pop() for key in tri.dict.keys() if q2 in key]
    #    # If (a,b) in tri, test isFlippable()
    #    for a in list_a:
    #        for b in list_b:
    #            if check_Flippable(tri, (a,b)):


    # con: edge in T2 that may be not in T1
    def resolve_cross(self, tri: Triangulation, con: tuple, t=None):
        if not t:
            q1 = con[0]
            q2 = con[1]
            for t in tri.triangles:
                i = t.get_ind(q1) # hy: i = 0~2 in a triangle t
                if i != -1: # hy: q1 node in a triangle
                    r1 = self.pts[q1] # hy: pts from Data not from Triangulation!!! r1 is (x,y) coordinate of q1
                    r2 = self.pts[t.pt(i + 1)] #hy: t.pt gives real node index
                    r3 = self.pts[t.pt(i + 2)]
                    r4 = self.pts[q2]
                    if (turn(r1, r2, r4) < 0 or turn(r1, r3, r4) > 0):
                        continue
                    if r2 == r4:
                        return []
                    elif r3 == r4:
                        return []
                    else:
                        return self.resolve_cross(tri, con, t)
        else:
            q1 = con[0]
            q2 = con[1]
            i = t.get_ind(q1)
            f = self.flip(tri, t, (i + 1) % 3) #hy: f = [(flip edge), (diagonal edge)]
            r = t.pt(i + 2)
            if (r == q2):
                return f
            elif (turn(self.pts[q2], self.pts[q1], self.pts[r]) < 0):
                return f + self.resolve_cross(tri, con, t)
            else:
                return f + self.resolve_cross(tri, con, t.nei(i + 2))

    #hy:
    def flip_from_seq(self, tri:Triangulation, t:Triangle, i:int):

        tt = t.neis[i]
        j = tt.get_ind(t.pt(i + 1)) #hy: t's i+1'th neis's node's index in tt's pts ㅋㅋㅋㅋㅋ

        #hy: coordinates
        p = self.pts[t.pt(i + 2)]
        pr = self.pts[t.pts[i]]
        pl = self.pts[t.pt(i + 1)]
        q = self.pts[tt.pt(j + 2)]
        #print("         ****** t.neis *******")
        #for i in range(3):
        #    if t.neis[i] == None:
        #        continue
        #    print("         t.neis[",i,"]: ")
        #    self.print_triangle(t.neis[i])
        #    print()
        #print("\n*************flipping in", i)
        #print("tt is t.neis[", i, "], and it is:")
        #self.print_triangle(tt)
        #print("         ****** tt.neis *******")
        #for i in range(3):
        #    if tt.neis[i] == None: continue
        #    print("         tt.neis[",i,"]: ")
        #    self.print_triangle(tt.neis[i])
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
                # t_pts = sorted([t.pts[0], t.pts[1], t.pts[2]])
                break
            print("hy: t.neis[i] = ", t.neis)
            tt = t.neis[i]
            j = tt.get_ind(t.pt(i + 1))
            q = self.pts[tt.pt(j + 2)]

        # compute the flip
        ti = t.nei(i + 1)
        tj = tt.nei(j + 1)
        pi = t.pt(i + 2)
        pj = tt.pt(j + 2)
        q1, q2 = min(t.pts[i], tt.pts[j]), max(t.pts[i], tt.pts[j])
        #f = [((q1, q2), (min(pi, pj), max(pi, pj)))]
        Fe_De = [((q1, q2), (min(pi, pj), max(pi, pj)))] # hy: Flip edge and Diagonal edge
        # f = [(q1, q2)]
        del(tri.dict[(q1, q2)])
        del(tri.dict[(q2, q1)])

        # edge update (for the triangulation)
        tri.edges.remove((q1, q2))
        tri.edges.add(((min(pi, pj), max(pi, pj))))

        # incidence update for t
        t.pts[(i + 1) % 3] = pj
        t.neis[i] = tj
        if tj:
            tj.neis[tj.get_ind(pj)] = t
        t.neis[(i + 1) % 3] = tt
        tri.dict[(t.pt(i), t.pt(i+1))] = t
        tri.dict[(t.pt(i+1), t.pt(i+2))] = t
        ##hy
        #if (t.pt(i), t.pt(i+1)) not in tri.dict:
        #    tri.dict[(t.pt(i), t.pt(i+1))] = []
        #tri.dict[(t.pt(i), t.pt(i+1))].append(t)
        #if (t.pt(i+1), t.pt(i+2)) not in tri.dict:
        #    tri.dict[(t.pt(i+1), t.pt(i+2))] = []
        #tri.dict[(t.pt(i+1), t.pt(i+2))].append(t)



        # incidence update for tt
        tt.pts[(j + 1) % 3] = pi
        tt.neis[j] = ti
        if ti:
            ti.neis[ti.get_ind(pi)] = tt
        tt.neis[(j + 1) % 3] = t
        tri.dict[(tt.pt(j), tt.pt(j+1))] = tt
        tri.dict[(tt.pt(j+1), tt.pt(j+2))] = tt
        ##hy
        #if (tt.pt(i), tt.pt(i+1)) not in tri.dict:
        #    tri.dict[(tt.pt(i), tt.pt(i+1))] = []
        #tri.dict[(tt.pt(i), tt.pt(i+1))].append(t)
        #if (tt.pt(i+1), tt.pt(i+2)) not in tri.dict:
        #    tri.dict[(tt.pt(i+1), tt.pt(i+2))] = []
        #tri.dict[(tt.pt(i+1), tt.pt(i+2))].append(t)

        #return f
        return Fe_De


    # triangulation tri에서, triangle t의 i번째 edge를 flip
    def flip(self, tri: Triangulation, t: Triangle, i: int):

        tt = t.neis[i]
        j = tt.get_ind(t.pt(i + 1)) #hy: t's i+1'th neis's node's index in tt's pts ㅋㅋㅋㅋㅋ

        p = self.pts[t.pt(i + 2)]
        pr = self.pts[t.pts[i]]
        pl = self.pts[t.pt(i + 1)]
        q = self.pts[tt.pt(j + 2)]
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
                # t_pts = sorted([t.pts[0], t.pts[1], t.pts[2]])
                break

            tt = t.neis[i]
            j = tt.get_ind(t.pt(i + 1))
            q = self.pts[tt.pt(j + 2)]

        # compute the flip
        ti = t.nei(i + 1)
        tj = tt.nei(j + 1)
        pi = t.pt(i + 2)
        pj = tt.pt(j + 2)
        q1, q2 = min(t.pts[i], tt.pts[j]), max(t.pts[i], tt.pts[j])
        #f = [((q1, q2), (min(pi, pj), max(pi, pj)))]
        Fe_De = [((q1, q2), (min(pi, pj), max(pi, pj)))] # hy: Flip edge and Diagonal edge
        # f = [(q1, q2)]
        del(tri.dict[(q1, q2)])
        del(tri.dict[(q2, q1)])

        # edge update (for the triangulation)
        tri.edges.remove((q1, q2))
        tri.edges.add(((min(pi, pj), max(pi, pj))))

        # incidence update for t
        t.pts[(i + 1) % 3] = pj
        t.neis[i] = tj
        if tj:
            tj.neis[tj.get_ind(pj)] = t
        t.neis[(i + 1) % 3] = tt
        tri.dict[(t.pt(i), t.pt(i+1))] = t
        tri.dict[(t.pt(i+1), t.pt(i+2))] = t

        # incidence update for tt
        tt.pts[(j + 1) % 3] = pi
        tt.neis[j] = ti
        if ti:
            ti.neis[ti.get_ind(pi)] = tt
        tt.neis[(j + 1) % 3] = t
        tri.dict[(tt.pt(j), tt.pt(j+1))] = tt
        tri.dict[(tt.pt(j+1), tt.pt(j+2))] = tt
        #return f
        return Fe_De
        # L = [f[0], t_pts]
        # return L

    # flip F: ((a, b), (c, d)) 꼴
    # point-in-polygon 여러 번. point-in-polygon은 halfplane으로.
    # 삼각형 정렬.

    def flipDiagonal(self, tri: Triangulation, F):
        #print("~~~~~~~~ flipDiagonal ~~~~~~")
        change = False

        #for t in tri.triangles:
        #    a, b = F[0] #hy: F[0] is edge in tri. After flip F[0], then F[1] will appear.

        #    if sorted([t.pts[0], t.pts[1]]) == sorted([a, b]):
        #        self.flip(tri, t, 0)
        #        change = True
        #        break
        #    elif sorted([t.pts[1], t.pts[2]]) == sorted([a, b]):
        #        self.flip(tri, t, 1)
        #        change = True
        #        break
        #    elif sorted([t.pts[2], t.pts[0]]) == sorted([a, b]):
        #        self.flip(tri, t, 2)
        #        change = True
        #        break

        #hy: faster
        v1, v2 = F[0]
        e = (v1,v2)
        Face = tri.dict[e]
        if sorted([Face.pts[0], Face.pts[1]])== sorted(e):
            self.flip_from_seq(tri, Face, 0)
            change = True
        elif sorted([Face.pts[1], Face.pts[2]])== sorted(e):
            self.flip_from_seq(tri, Face, 1)
            change = True
        elif sorted([Face.pts[2], Face.pts[0]])== sorted(e):
            self.flip_from_seq(tri, Face, 2)
            change = True

        assert(change)

    ##hy: Only check flippable or not.
    #def check_Flippable(self, tri:Triangulation, e:tuple):
    #    if e not in tri.edges:
    #        return False

    #    v1, v2 = e
    #    e_r = (v2,v1)
    #    #hy: Find two faces if exist
    #    if e in tri.dict:
    #        face1 = tri.dict[e]
    #        if e_r in tri.dict:
    #            face2 = tri.dict[e_r]
    #        else:#hy: e is convex hull
    #            return False
    #    else: #hy: e_r is convex hull
    #        return False


    #    #hy: set order (v1, v2, v3, v4)
    #    #hy: (v1, v3) is e
    #    if v1 == face2.pts[0] and v2 == face2.pts[2]:
    #        v1, v2, v3 = face2.pts
    #        Face = face2
    #        v4 = int(list(set(face1.pts)-set(face2.pts))[0])
    #    elif v1 == face2.pts[2] and v2 == face2.pts[1]:
    #        v2, v3, v1 = face2.pts
    #        Face = face2
    #        v4 = int(list(set(face1.pts)-set(face2.pts))[0])
    #    elif v1 == face2.pts[1] and v2 == face2.pts[0]:
    #        v3, v1, v2 = face2.pts
    #        Face = face2
    #        v4 = int(list(set(face1.pts)-set(face2.pts))[0])

    #    v1 = self.pts[v1]
    #    v2 = self.pts[v2]
    #    v3 = self.pts[v3]
    #    v4 = self.pts[v4]
    #    if turn(v1, v2, v3) <0 or turn(v2, v3, v4)<0 or turn(v3, v4, v1)<0 or turn(v4, v1, v2)<0:
    #        return False
    #    else:
    #        return True
    #        #if sorted([Face.pts[0], Face.pts[1]])== sorted(e):
    #        #    self.flip_from_seq(tri, Face, 0)
    #        #    return True
    #        #elif sorted([Face.pts[1], Face.pts[2]])== sorted(e):
    #        #    self.flip_from_seq(tri, Face, 1)
    #        #    return True
    #        #elif sorted([Face.pts[2], Face.pts[0]])== sorted(e):
    #        #    self.flip_from_seq(tri, Face, 2)
    #        #    return True


    #hy: Flip e if it is flippable.
    def isFlippable(self, tri:Triangulation, e:tuple):
        if e not in tri.edges:
            return False

        v1, v2 = e
        e_r = (v2,v1)
        #hy: Find two faces if exist
        if e in tri.dict:
            face1 = tri.dict[e]
            if e_r in tri.dict:
                face2 = tri.dict[e_r]
            else:#hy: e is convex hull
                return False
        else: #hy: e_r is convex hull
            return False


        #hy: set order (v1, v2, v3, v4)
        #hy: (v1, v3) is e
        if v1 == face2.pts[0] and v2 == face2.pts[2]:
            v1, v2, v3 = face2.pts
            Face = face2
            v4 = int(list(set(face1.pts)-set(face2.pts))[0])
        elif v1 == face2.pts[2] and v2 == face2.pts[1]:
            v2, v3, v1 = face2.pts
            Face = face2
            v4 = int(list(set(face1.pts)-set(face2.pts))[0])
        elif v1 == face2.pts[1] and v2 == face2.pts[0]:
            v3, v1, v2 = face2.pts
            Face = face2
            v4 = int(list(set(face1.pts)-set(face2.pts))[0])

        v1 = self.pts[v1]
        v2 = self.pts[v2]
        v3 = self.pts[v3]
        v4 = self.pts[v4]
        if turn(v1, v2, v3) <0 or turn(v2, v3, v4)<0 or turn(v3, v4, v1)<0 or turn(v4, v1, v2)<0:
            return False
        else:
            if sorted([Face.pts[0], Face.pts[1]])== sorted(e):
                self.flip_from_seq(tri, Face, 0)
                return True
            elif sorted([Face.pts[1], Face.pts[2]])== sorted(e):
                self.flip_from_seq(tri, Face, 1)
                return True
            elif sorted([Face.pts[2], Face.pts[0]])== sorted(e):
                self.flip_from_seq(tri, Face, 2)
                return True

    def flip_sequence(self, tri1: Triangulation, tri2: Triangulation, numTrials: int = 1):
        fs = []
        tri = deepcopy(tri1)
        edges = list(tri2.edges)
        random.shuffle(edges)
        cnt = 0
        for e in edges:
            #print("flip_sequence: e = ", e)
            fs += self.resolve_cross(tri, e) #hy: fs = [((need-to-flip edge in T1), (to-get "e" in T2)), ((), ()), ....]
            #print("fs = ", fs)
            #print("-----------------")
            #cnt += 1
            #if cnt % 100 == 0:
            #    print(cnt, "/", len(edges), " in flip_sequence")
        del tri
        return fs


    def resolve_cross_new(self, tri: Triangulation, con: tuple, t=None):
        if not t:
            q1 = con[0]
            q2 = con[1]
            for t in tri.triangles:
                i = t.get_ind(q1)
                if i != -1:
                    r1 = self.pts[q1]
                    r2 = self.pts[t.pt(i + 1)]
                    r3 = self.pts[t.pt(i + 2)]
                    r4 = self.pts[q2]
                    if (turn(r1, r2, r4) < 0 or turn(r1, r3, r4) > 0):
                        continue
                    if r2 == r4:
                        return []
                    elif r3 == r4:
                        return []
                    else:
                        return self.resolve_cross(tri, con, t)
        else:
            q1 = con[0]
            q2 = con[1]
            # print("resolving", q1, q2)
            # print("----- Starting triangle -----")
            # self.print_triangle(t)
            # print("-----------------------------")

            i = t.get_ind(q1)
            ts = [(t, (i + 1) % 3)]
            while True:
                tt, j = ts[-1]
                ttt = tt.neis[j]
                k = (ttt.get_ind(tt.pts[j]) + 1) % 3
                if turn(self.pts[q1], self.pts[q2], self.pts[ttt.pts[k]]) <= 0:
                    ts.append((ttt, k))
                else:
                    ts.append((ttt, (k-1)%3))
                if ttt.pts[k] == q2:
                    break
            ind = 0
            f = []
            while ind < len(ts) - 1:
                tt, j = ts[ind]
                ttt = tt.neis[j]
                k = (ttt.get_ind(tt.pts[j]) + 1) % 3
                r1 = self.pts[tt.pt(j - 1)]
                r2 = self.pts[tt.pt(j)]
                r3 = self.pts[tt.pt(j + 1)]
                r4 = self.pts[ttt.pt(k)]
                if (turn(r1, r2, r4) <= 0 or turn(r1, r3, r4) >= 0):
                    ind += 1
                    continue
                if ind > 0:
                    tb, ib = ts[ind - 1]
                    r1 = self.pts[tb.pt(ib - 1)]
                    r2 = self.pts[tb.pt(ib)]
                    r3 = self.pts[tb.pt(ib + 1)]
                    if (turn(r1, r2, r4) <= 0 or turn(r1, r3, r4) >= 0):
                        ind += 1
                        continue
                f += self.flip(tri, tt, j)
                ind += 2
            r = t.pt(i + 2)
            if (r == q2):
                return f
            elif (turn(self.pts[q2], self.pts[q1], self.pts[r]) < 0):
                return f + self.resolve_cross(tri, con, t)
            else:
                return f + self.resolve_cross(tri, con, t.nei(i + 2))
    '''
    def flip_sequence_new(self, tri1: Triangulation, tri2: Triangulation, numTrials: int = 1):
        # numTrials = 10
        bestDist = 10000
        bestSeq = None

        for i in range(numTrials):

            fs = []
            tri = tri1.copy()
            edges = list(tri2.edges)
            random.shuffle(edges)

            # list of (triangulation, len(fs) so far)
            L = []

            for e in edges:

                before = len(fs)

                ADDED = self.resolve_cross(tri, e)
                # print('added length:', len(ADDED))
                fs += ADDED
                # fs += self.resolve_cross(tri, e)

                # if something added
                if len(fs) != before:
                    L.append([deepcopy(tri), len(fs)])

            del tri

            if len(fs) < bestDist:
                bestDist = len(fs)
                bestSeq = fs

        return bestSeq, L
    '''
    def new_pfp(self, tri1:Triangulation, tri2:Triangulation):
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


    # tri1에서 tri2로의 pfp
    # tri2에서 tri1로 가는 경우 리스트를 반대로 뒤집어야 함
    def parallel_flip_path(self, tri1: Triangulation, tri2: Triangulation):
        #print("parallel_flip_path")
        best = []
        bestscore = len(tri1.edges) * 2
        trial = 0
        NUM_TRIAL = len(self.pts) // 10
        #NUM_TRIAL = 1
        while trial < NUM_TRIAL or bestscore == len(tri1.edges)*2:

            trial += 1

            # flip sequence 자체에서 들어감
            fs = self.flip_sequence(tri1, tri2)
            done = [False] * len(fs)
            tri = deepcopy(tri1)
            pfp = []
            donenum = 0
            while not all(done):
                fps = []
                usedtri = set()
                for i in range(len(fs)):
                    if done[i]: continue
                    (p1, p2), (p3, p4) = fs[i]
                    t1 = tri.find_triangle(p1, p2)
                    if not t1 or t1 in usedtri:
                        continue
                    t2 = tri.find_triangle(p2, p1)
                    assert (t2)
                    if t2 in usedtri:
                        continue
                    p5, p6 = t1.pt(t1.get_ind(p2) + 1), t2.pt(t2.get_ind(p1) + 1)
                    if (min(p5, p6), max(p5, p6)) != (p3, p4):
                        continue
                    done[i] = True
                    usedtri.add(t1)
                    usedtri.add(t2)
                    fps.append(fs[i])
                donenum += len(fps)
                #print("*", donenum, "/", len(done))
                for _, con in fps:
                    self.resolve_cross(tri, con)
                pfp.append(fps)
            if len(pfp) < bestscore:
                best = pfp
                bestscore = len(pfp)
                trial = 0
        #print("len(best) = ", len(best))
        return best

    #hy: same with generate_pfp() but different return shape
    def parallel_flip_path_fast(self, tri1: Triangulation, tri2: Triangulation):
        #print("parallel_flip_path_fast")
        # flip sequence 자체에서 들어감
        fs = self.flip_sequence(tri1, tri2)
        done = [False] * len(fs)
        tri = deepcopy(tri1)
        pfp = []
        donenum = 0
        while not all(done):
            fps = []
            usedtri = set()
            for i in range(len(fs)):
                if done[i]: continue
                (p1, p2), (p3, p4) = fs[i]
                t1 = tri.find_triangle(p1,p2)
                if not t1 or t1 in usedtri: continue
                t2 = tri.find_triangle(p2,p1)
                assert (t2)
                if t2 in usedtri: continue
                p5, p6 = t1.pt(t1.get_ind(p2) + 1), t2.pt(t2.get_ind(p1) + 1)
                if (min(p5, p6), max(p5, p6)) != (p3, p4): continue
                done[i] = True
                usedtri.add(t1)
                usedtri.add(t2)
                fps.append(fs[i])
            donenum += len(fps)
            #print("*", donenum, "/", len(done))
            for _, con in fps:
                self.resolve_cross(tri, con)
            pfp.append(fps)
        best = pfp#hy
        #if len(pfp) < bestscore:
        #    best = pfp
        #    bestscore = len(pfp)
        #    trial = 0
        #print("len(best) = ", len(best))
        return best


    #hy: just generate pfp, no need to best
    def generate_pfp(self, tri1: Triangulation, tri2: Triangulation):
        # flip sequence 자체에서 들어감
        fs = self.flip_sequence(tri1, tri2)
        done = [False] * len(fs)
        tri = deepcopy(tri1)
        pfp = []
        donenum = 0
        while not all(done):
            fps = []
            usedtri = set()
            for i in range(len(fs)):
                if done[i]: continue
                (p1, p2), (p3, p4) = fs[i]
                t1 = tri.find_triangle(p1, p2)
                if not t1 or t1 in usedtri:
                    continue
                t2 = tri.find_triangle(p2, p1)
                assert (t2)
                if t2 in usedtri:
                    continue
                p5, p6 = t1.pt(t1.get_ind(p2) + 1), t2.pt(t2.get_ind(p1) + 1)
                if (min(p5, p6), max(p5, p6)) != (p3, p4):
                    continue
                done[i] = True
                usedtri.add(t1)
                usedtri.add(t2)
                fps.append(fs[i])
            donenum += len(fps)
            #print("+", donenum, "/", len(done))
            flippable_edge=[]
            for f, con in fps:
                self.resolve_cross(tri, con)
                flippable_edge.append(f)
            #pfp.append(fps)
            #print("fps: ", fps)
            #print("flippable_edge: ", flippable_edge)
            pfp.append(flippable_edge)
        return pfp


    def print_triangle(self, t: Triangle):
        #print("Triangle :", end="")
        print("node ", t.pts[0],t.pts[1],t.pts[2])
        #print("print_triangle :")
        #print("node ", t.pts[0], ":(x,y)= ", self.pts[t.pts[0]])
        #print("node ", t.pts[1], ":(x,y)= ", self.pts[t.pts[1]])
        #print("node ", t.pts[2], ":(x,y)= ", self.pts[t.pts[2]])
        if turn(self.pts[t.pts[0]], self.pts[t.pts[1]], self.pts[t.pts[2]])<0:
            print("negative!")
            print("node ", t.pts[0], ":(x,y)= ", self.pts[t.pts[0]])
            print("node ", t.pts[1], ":(x,y)= ", self.pts[t.pts[1]])
            print("node ", t.pts[2], ":(x,y)= ", self.pts[t.pts[2]])
        #print(t.neis[0])
        #print(t.neis[1])
        #print(t.neis[2])

    # w1:w2 내분점에서 가장 가까운 중간 triangulation을 반환
    def internal_division(self, T1: Triangulation, w1: int, T2: Triangulation, w2: int):
        #print("________ internal_division _________")
        T1copy = deepcopy(T1)

        # parallel version
        #start = time.time()
        pfp = self.parallel_flip_path(T1, T2)
        #pfp = self.parallel_flip_path_fast(T1, T2)
        #end = time.time() - start
        #print("parallel_flip_path_fast time = ", end)
        midVal = int(len(pfp) * (w2 / (w1 + w2)))
        #print('len(pfp):', len(pfp), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)

        # use_pfp: 사용되는 parallel flip들 모음
        use_pfp = pfp[:midVal]

        #start = time.time()
        for flips in use_pfp:
            for flip in flips:

                self.flipDiagonal(T1copy, flip)
        #end = time.time() - start
        #print("flipDiagonal time = ", end)

        return T1copy

        # sequential version
        '''
        fs = self.flip_sequence(T1, T2)

        midVal = int(len(fs) * (w1 / (w1 + w2)))
        print('len(fs):', len(fs), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)

        # use_fs: 사용되는 sequential flip들 모음
        use_fs = fs[:midVal]

        for F in use_fs:
            self.flipDiagonal(T1copy, F)

        return T1copy
        '''

    # reconstruction이 필요
    # 우선, 한 번의 diagonal flip이 이루어진 triangulation 계산?

    def findCenter(self):

        #start = time.time()
        #print("in findCenter")
        # random.shuffle(self.triangulations)
        centerT = self.triangulations[0]
        weight = 1

        for i in range(1, len(self.triangulations)):
            #print("... ", i,"th triangle")
            nextT = self.triangulations[i]
            # 내분을 통해 새로운 central triangulation 계산
            centerT = self.internal_division(centerT, weight, nextT, 1)
            weight += 1

            #end = time.time()
            #print('time:', f"{end - start:.5f} sec")

        #with open("centers/" + self.instance_uid + ".json", "w", encoding="utf-8") as f:
        #    json.dump(list(centerT.edges), f, indent='\t')

        return centerT
        # local search to move to a certain direction

        # A -> B로 가는 것과 A -> C로 가는 게 비슷하면 좋겠지.

    def WriteData(self):

        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_uid


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": sum([len(pFlip) for pFlip in self.pFlips])} # , "input": self.input}

        folder = "solutions"
        with open(folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        opt_folder = "pb_opt"
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

        #start = time.time()

        #print("self.pFlips = ", self.pFlips)
        #print("len() = ", len(self.pFlips))
        #print("len(self.triangulations) = ",len(self.triangulations))

        Dsum=0
        for i in range(len(self.triangulations)):

            # list[list[list[int, int]]]
            self.pFlips[i] = []

            # list[list[list[tuple(int, int), tuple(int, int)]]]
            #pFlips_paired = self.parallel_flip_path_fast(self.triangulations[i], centerT)
            pFlips_paired = self.parallel_flip_path(self.triangulations[i], centerT)
            #pFlips_paired = self.generate_pfp(self.triangulations[i], centerT)#hy for short time

            for round_ in pFlips_paired:

                round_temp = []

                for oneFlip in round_:

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
            Dsum += len(self.pFlips[i])
            #print('parallel flip distance from the center to T', i, ':', len(self.pFlips[i]))
            #print('pfd from the center to T_', i, ':', len(self.pFlips[i]))

            #end = time.time()
            #print('time:', f"{end - start:.5f} sec")
        return Dsum #hy

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
