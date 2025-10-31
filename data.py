import json
import sys
import random
from copy import deepcopy

sys.setrecursionlimit(1000000)

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __ne__(self, p):
        return self.x != p.x or self.y != p.y

    def __lt__(self, p):
        return (self.x, self.y) < (p.x, p.y)

    def __le__(self, p):
        return (self.x, self.y) <= (p.x, p.y)

    def __gt__(self, p):
        return (self.x, self.y) > (p.x, p.y)

    def __ge__(self, p):
        return (self.x, self.y) >= (p.x, p.y)

class Edge:
    def __init__(self, ):
        
        self.endpoints = []

class Triangle:
    # p, q, r: point index
    def __init__(self, p: int, q: int, r: int):
        self.pts = [p, q, r]
        self.neis = [None, None, None]

    def get_ind(self, p: int):
        for i in range(3):
            if self.pts[i] == p:
                return i
        return -1

    def pt(self, i: int):
        return self.pts[i % 3]

    def nei(self, i: int):
        return self.neis[i % 3]


class Triangulation:
    def __init__(self):
        self.triangles = set()
        self.edges = set()

    def __del__(self):
        for t in self.triangles:
            del t

    def find_triangle(self, q1: int, q2: int):
        for t in self.triangles:
            if t.pts[0] == q1 and t.pts[1] == q2:
                return t
            if t.pts[1] == q1 and t.pts[2] == q2:
                return t
            if t.pts[2] == q1 and t.pts[0] == q2:
                return t
        return None

    def copy(self):
        newtri = Triangulation()
        newtri.edges = set(self.edges)
        tridict = dict()
        for t in self.triangles:
            tt = Triangle(t.pts[0], t.pts[1], t.pts[2])
            tridict[t] = tt
            for i in range(3):
                if t.neis[i] in tridict:
                    ttt = tridict[t.neis[i]]
                    tt.neis[i] = ttt
                    ttt.neis[ttt.get_ind(tt.pt(i + 1))] = tt
            newtri.triangles.add(tt)
        return newtri


class Data:
    def __init__(self, inp):
        self.input = inp
        self.triangulations = []
        self.ReadData()

    def ReadData(self):
        # print("--------------------ReadData--------------------")
        f = open(self.input, "r", encoding="utf-8")
        root = json.load(f)
        # print(root)
        self.instance_name = root["instance_uid"]
        pts_x = root["points_x"]
        pts_y = root["points_y"]
        self.pts = []
        for i in range(len(pts_y)):
            self.pts.append(Point(pts_x[i], pts_y[i]))
        for t in root["triangulations"]:
            self.triangulations.append(self.make_triangulation(t))

    def make_triangulation(self, t):
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
                            tt = tri.find_triangle(t.pt(l + 1), t.pt(l))
                            if tt:
                                tt.neis[tt.get_ind(t.pt(l + 1))] = t
                                t.neis[l] = tt
                        tri.triangles.add(t)
        return tri

    def resolve_cross(self, tri: Triangulation, con: tuple, t=None):
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
            f = self.flip(tri, t, (i + 1) % 3)
            r = t.pt(i + 2)
            if (r == q2):
                return f
            elif (turn(self.pts[q2], self.pts[q1], self.pts[r]) < 0):
                return f + self.resolve_cross(tri, con, t)
            else:
                return f + self.resolve_cross(tri, con, t.nei(i + 2))

    # triangulation tri에서, triangle t의 i번째 edge를 flip
    def flip(self, tri: Triangulation, t: Triangle, i: int):
        tt = t.neis[i]
        j = tt.get_ind(t.pt(i + 1))
        p = self.pts[t.pt(i + 2)]
        pr = self.pts[t.pts[i]]
        pl = self.pts[t.pt(i + 1)]
        q = self.pts[tt.pt(j + 2)]
        # self.print_triangle(t)
        # print("flipping in", i)
        # self.print_triangle(tt)
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
            j = tt.get_ind(t.pt(i + 1))
            q = self.pts[tt.pt(j + 2)]
        ti = t.nei(i + 1)
        tj = tt.nei(j + 1)
        pi = t.pt(i + 2)
        pj = tt.pt(j + 2)
        q1, q2 = min(t.pts[i], tt.pts[j]), max(t.pts[i], tt.pts[j])
        f = [((q1, q2), (min(pi, pj), max(pi, pj)))]
        tri.edges.remove((q1, q2))
        tri.edges.add(((min(pi, pj), max(pi, pj))))
        t.pts[(i + 1) % 3] = pj
        t.neis[i] = tj
        if tj:
            tj.neis[tj.get_ind(pj)] = t
        t.neis[(i + 1) % 3] = tt
        tt.pts[(j + 1) % 3] = pi
        tt.neis[j] = ti
        if ti:
            ti.neis[ti.get_ind(pi)] = tt
        tt.neis[(j + 1) % 3] = t
        return f

    # flip F: ((a, b), (c, d)) 꼴
    # point-in-polygon 여러 번. point-in-polygon은 halfplane으로.
    # 삼각형 정렬.
    def isFlippable(F):
        pass
    
    # flipped diagonal F: ((a, b), (c, d)) 꼴
    def flipDiagonal(self, tri: Triangulation, F):
        if not isFlippable(F):
            return

        T = tri.find_triangle(F)
        
            
    def flip_sequence(self, tri1: Triangulation, tri2: Triangulation, numTrials: int = 1):
        fs = []
        tri = tri1.copy()
        edges = list(tri2.edges)
        random.shuffle(edges)
        for e in edges:
            fs += self.resolve_cross(tri, e)
        del tri
        return fs

    
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

    def parallel_flip_path(self, tri1: Triangulation, tri2: Triangulation):
        best = []
        bestscore = len(tri1.edges) * 2
        trial = 0
        NUM_TRIAL = len(self.pts) // 10
        while trial < NUM_TRIAL or bestscore == len(tri1.edges) * 2:
            trial += 1

            # flip sequence 자체에서 들어감
            fs = self.flip_sequence(tri1, tri2)
            done = [False] * len(fs)
            tri = tri1.copy()
            pfp = []
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
                for _, con in fps:
                    self.resolve_cross(tri, con)
                pfp.append(fps)
            if len(pfp) < bestscore:
                best = pfp
                bestscore = len(pfp)
                trial = 0
                
        return best

    def print_triangle(self, t: Triangle):
        print("Triangle :", end="")
        print(t)
        print(t.pts[0], ":", self.pts[t.pts[0]])
        print(t.pts[1], ":", self.pts[t.pts[1]])
        print(t.pts[2], ":", self.pts[t.pts[2]])
        print(t.neis[0])
        print(t.neis[1])
        print(t.neis[2])
        
    # parallel 버전으로 바꿔야 함.
    # w1:w2 내분점에서 가장 가까운 중간 triangulation을 반환
    def internal_division(self, T1: Triangulation, w1: int, T2: Triangulation, w2: int):
        # print('internal division start. w1 =', w1, ', w2 =', w2)

        # 이 함수 자체는 sequential로 하는 게 나으려나?

        T1copy = deepcopy(T1)
        
        midVal = len(S) * (w1 / (w1 + w2))
        print('len(S):', len(S), 'w1:', w1, 'w2:', w2, 'midVal:', midVal)
        
        # sequential version
        fs = flip_sequence(T1, T2)
        use_fs = fs[midVal]
        for F in use_fs:
            flip(T1copy, F)

        return T1copy

        '''
        # parallel version
        
        # S, L = self.flip_sequence_new(T1, T2) # S stands for Sequence, List (of triangulations)

        bestT = None # best triangulation
        bestDiff = 100000000 # 내분점과의 최소거리
        for pr in L:
            curDiff = abs(pr[1] - midVal)
            if curDiff < bestDiff:
                print('curDiff:', curDiff, 'pr[1]:', pr[1])
                bestDiff = curDiff
                bestT = pr[0]
            
        return bestT
        '''
        
    # reconstruction이 필요
    # 우선, 한 번의 diagonal flip이 이루어진 triangulation 계산?
    
    def parallel_flip_path_all(self):

        # random.shuffle(self.triangulations)
        centerT = self.triangulations[0]
        weight = 1
        
        for i in range(1, len(self.triangulations)):
            nextT = self.triangulations[i]
            # 내분을 통해 새로운 central triangulation 계산
            centerT = self.internal_division(centerT, weight, nextT, 1) 
            weight += 1

        return centerT
        # local search to move to a certain direction

        # A -> B로 가는 것과 A -> C로 가는 게 비슷하면 좋겠지.

def angle(p1: Point, p2: Point, p3: Point):
    if p1 == p2:
        raise Exception("Cannot Calculate Angle")
    if p2 == p3:
        raise Exception("Cannot Calculate Angle")
    p12x = p2.x - p1.x
    p12y = p2.y - p1.y
    p23x = p2.x - p3.x
    p23y = p2.y - p3.y
    ab = p12x * p23x + p12y * p23y
    a = p12x * p12x + p12y * p12y
    b = p23x * p23x + p23y * p23y
    if (ab >= MyNum(0)):
        return - ab * ab / a / b
    else:
        return ab * ab / a / b

# CCW라면 true 반환, CW라면 false반환
def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)