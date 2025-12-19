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
            print(cand)
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
            print(len(flips))
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
