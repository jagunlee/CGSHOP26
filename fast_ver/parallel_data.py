import json, os
from fast_Triangulation import *
import numba
import time
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
from itertools import repeat



# 1. 전역 변수나 별도 공간에 배열을 배치하여 fork 시 복사 비용 최소화
#def init_worker(pts_coor, tri_f_pts, tri_f_nei, tri_e2f, tri_des_f_pts, tri_des_f_nei, tri_des_e2f, tri_des_adj):
    #global gb_pts_coor, gb_f_pts, gb_e2f, gb_des_f_pts, gb_des_f_nei, gb_des_e2f, gb_des_adj
def init_worker(pts_coor, tri_des_f_pts, tri_des_f_nei, tri_des_e2f, tri_des_adj):
    global gb_pts_coor, gb_des_f_pts, gb_des_f_nei, gb_des_e2f, gb_des_adj

    gb_pts_coor = pts_coor

    #gb_f_pts = tri_f_pts
    #gb_e2f = tri_e2f

    gb_des_f_pts = tri_des_f_pts
    gb_des_f_nei = tri_des_f_nei
    gb_des_e2f = tri_des_e2f
    gb_des_adj = tri_des_adj

#def check_edge_score_numpy(e, t1, t2):
def check_edge_score_numpy(e, t1, t2, cur_f_pts):
    if flippable_fast(e, t1, t2, cur_f_pts):
        score = flip_score_fast(e, t1, t2, cur_f_pts)
        if score[0] > 0:
            return (e, score)
    return None

#def flippable_fast(e:tuple, t1, t2):
@numba.njit
def flippable_fast(e, t1, t2, cur_f_pts):
    q1, q3 =e

    row1 = cur_f_pts[t1]
    if row1[0] == q3: i=0
    elif row1[1] == q3: i=1
    else: i=2
    p4 = row1[(i+1)%3]

    row2 = cur_f_pts[t2]
    if row2[0] == q1: j=0
    elif row2[1] == q1: j=1
    else: j=2
    p2 = row2[(j+1)%3]

    p1, p3 = q1, q3
    q1, q2, q3, q4 = gb_pts_coor[p1], gb_pts_coor[p2], gb_pts_coor[p3], gb_pts_coor[p4]
    turn_val1= (q3[0]-q2[0])*(q4[1]-q2[1]) - (q3[1]-q2[1])*(q4[0]-q2[0])
    if turn_val1<=0: return False
    turn_val2= (q1[0]-q2[0])*(q4[1]-q2[1]) - (q1[1]-q2[1])*(q4[0]-q2[0])
    return turn_val2 < 0




def find_t_c_fast(q1, q2):
    p1, p2 = np.int64(q1), np.int64(q2)
    if((p1<<32)|p2) in gb_des_e2f or ((p2<<32)|p1) in gb_des_e2f: return None

    p = gb_des_adj[p1]
    p = np.int64(p)

    t = gb_des_e2f.get((p1<<32)|p)
    if t is None:
        t = gb_des_e2f.get((p<<32)| p1)
    assert(t!=None)

    r1 = gb_pts_coor[q1]
    r4 = gb_pts_coor[q2]

    return _find_t_c_fast(q1, q2, t, r1, r4)

@numba.njit
def _find_t_c_fast(q1, q2, t, r1, r4):
    while True:
        row = gb_des_f_pts[t]
        if row[0] == q1: i=0
        elif row[1] == q1: i=1
        else: i=2
        r2 = gb_pts_coor[row[(i+1)%3]]
        r3 = gb_pts_coor[row[(i+2)%3]]

        turn_val1= (r2[0]-r1[0])*(r4[1]-r1[1]) - (r2[1]-r1[1])*(r4[0]-r1[0])
        turn_val2= (r3[0]-r1[0])*(r4[1]-r1[1]) - (r3[1]-r1[1])*(r4[0]-r1[0])
        if turn_val1 < 0:
            t = gb_des_f_nei[t, i]
        elif turn_val2 >0:
            t = gb_des_f_nei[t, (i+2)%3]
        else:
            return t


@numba.njit
def numba_count_cross_fast(q1, q2, t):
    row = gb_des_f_pts[t]
    if row[0] == q1: i=0
    elif row[1] == q1: i=1
    else: i=2

    idx_next = (i+1)%3
    tt = gb_des_f_nei[t, idx_next]
    tmp = gb_des_f_pts[t, idx_next]
    row_tt = gb_des_f_pts[tt]
    if row_tt[0] == tmp: j=0
    elif row_tt[1] == tmp: j=1
    else: j=2

    p_q1 = gb_pts_coor[q1]
    p_q2 = gb_pts_coor[q2]

    cnt = 1
    while True:
        if row_tt[(j+1)%3] == q2: break
        cnt +=1
        t, i = tt, j
        idx_next = (i+1)%3

        tmp = gb_des_f_pts[t, idx_next]
        p_tmp = gb_pts_coor[tmp]
        turn_val= (p_q2[0]-p_q1[0])*(p_tmp[1]-p_q1[1]) - (p_q2[1]-p_q1[1])*(p_tmp[0]-p_q1[0])
        if turn_val <0:
            tt = gb_des_f_nei[t, idx_next]
            tmp = gb_des_f_pts[t, idx_next]
            row_tt = gb_des_f_pts[tt]
            if row_tt[0] == tmp: j=0
            elif row_tt[1] == tmp: j=1
            else: j=2
        else:
            tt = gb_des_f_nei[t, i%3]
            tmp = gb_des_f_pts[t, i%3]
            row_tt = gb_des_f_pts[tt]
            if row_tt[0] == tmp: j=0
            elif row_tt[1] == tmp: j=1
            else: j=2
    return cnt


def flip_score_fast(e:tuple, t1, t2, cur_f_pts):
    p1, p3 = e

    row1 = cur_f_pts[t1]
    if row1[0] == p3: i = 0
    elif row1[1] == p3: i = 1
    else: i = 2
    p4 = row1[(i + 1) % 3]

    row2 = cur_f_pts[t2]
    if row2[0] == p1: j = 0
    elif row2[1] == p1: j = 1
    else: j = 2
    p2 = row2[(j + 1) % 3]


    t1 = find_t_c_fast(p1, p3)
    t2 = find_t_c_fast(p2, p4)

    ori_cross=0
    new_cross=0

    if t1 is None:
        ori_cross=0
    else:
        ori_cross = numba_count_cross_fast(p1,p3,t1)
    if t2 is None:
        new_cross=0
    else:
        new_cross = numba_count_cross_fast(p2,p4,t2)
    n_cross = ori_cross - new_cross
    depth=1
    m_score = (n_cross, depth)
    return m_score

class FastData:
    def __init__(self, inp):
        self.input = inp
        self.pts = None # np.array([N, 2])
        self.ReadData()


    def ReadData(self):
        if "solution" not in self.input:
            pass
        else:
            with open(self.input, "r", encoding="utf-8") as f:
                root=json.load(f)

            self.pFlips = root["flips"]
            self.dist = sum([len(x) for x in self.pFlips])
            self.instance_name = root["instance_uid"]


            org_input = '/Users/hyeyun/Experiment/PFD/hyeyun_git/data/benchmark_instances/'+self.instance_name+'.json'
            with open(org_input, "r", encoding="utf-8") as f:
                root=json.load(f)
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = np.array(list(zip(root["points_x"], root["points_y"])), dtype=np.float64) # can be replaced by pts_x,y?

            self.num_pts = len(root["points_x"])
            self.num_edges = len(root["triangulations"][0])
            self.num_faces = self.num_edges - self.num_pts + 1 #Euler Characteristic, F = E-V+1

            self.num_tris = len(root["triangulations"])
            self.triangulations = [None] * (self.num_tris+1)
            print("make triangulation ...:")
            for i, t_data in enumerate(root["triangulations"]):
                self.triangulations[i] = self.make_triangulation(t_data)
                print(f"T{i}", end=' ', flush=True)
            print(end='\n')

            # restore center
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.triangulations[min_flip_ind].fast_copy()
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])
            self.triangulations[-1] = self.center.fast_copy()
            new_dist = 0
            new_pfp = [None]*(self.num_tris)
            for i in range(self.num_tris):
                start = time.time()
                new_pfp1 = self.parallel_flip_path2(i, -1)
                print(f"T{i}: {time.time()-start:.2f}s", end=' ', flush=True)
                start = time.time()
                new_pfp2 = self.parallel_flip_path_rev2(-1, i)
                print(f"rev: {time.time()-start:.2f}s", end='\n')
                if len(new_pfp1) < len(new_pfp2):
                    new_pfp[i] = new_pfp1
                else:
                    new_pfp[i] = new_pfp2
                new_dist+=len(new_pfp[i])
            print(f"Original dist: {self.dist}, New dist: {new_dist}")
            if new_dist<self.dist:
                self.dist = new_dist
                self.pFlips = new_pfp
                self.WriteData()
                print("dist and pFlips are updated!\n")


    def make_triangulation(self, t_data):
        num_pts = self.num_pts
        num_faces = self.num_faces
        tri = FastTriangulation(num_pts, num_faces)
        pts_coor = self.pts
        f_pts = tri.face_pts
        f_nei = tri.face_nei
        e2f = tri.edge_to_face

        graph = [[] for _ in range(num_pts)]
        for u, v in t_data:
            graph[u].append(v)
            graph[v].append(u)
            tri.adj[u] = v
            tri.adj[v] = u
            tri.edges.add((min(u, v), max(u, v)))


        face_idx = 0
        for i in range(num_pts):
            for j in range(len(graph[i])):
                v1 = graph[i][j]
                if v1 < i : continue
                for k in range(j+1, len(graph[i])):
                    v2 = graph[i][k]
                    if v2 < i : continue
                    if v1 in graph[v2]:
                        p1, p2, p3 = pts_coor[i], pts_coor[v1], pts_coor[v2]
                        if turn(p1, p2, p3)>0:
                            pts = [i, v1, v2]
                        else:
                            pts = [i, v2, v1]
                        flag = False
                        for l in range(len(graph[i])):
                            if l ==j  or l ==k: continue
                            v3 = graph[i][l]
                            smallflag = True
                            for m in range(3):
                                q1 = pts[m%3]
                                q2 = pts[(m+1)%3]
                                smallflag &= turn(pts_coor[q1], pts_coor[q2], pts_coor[v3]) >=0
                            if smallflag:
                                flag = True
                                break
                        if flag:
                            continue

                        # save face
                        f_pts[face_idx] = pts
                        face_pts = pts
                        for l in range(3):
                            p1 = face_pts[l]
                            p2 = face_pts[(l+1)%3]
                            key12 = (np.int64(p1)<<32)|np.int64(p2)
                            # save edge_to_face
                            e2f[key12] = face_idx

                            key21 = (np.int64(p2)<<32)|np.int64(p1)
                            share_face_idx = e2f.get(key21)
                            if share_face_idx is not None:
                                tpt = f_pts[face_idx][(l+1)%3]
                                row_tt = f_pts[share_face_idx]
                                if row_tt[0] == tpt: kk=0
                                elif row_tt[1] == tpt: kk=1
                                else: kk=2
                                f_nei[share_face_idx][kk] = face_idx
                                f_nei[face_idx][l] = share_face_idx
                        face_idx+=1
        return tri

    def flippable(self, tri:FastTriangulation, e:tuple):
        q1, q3 = e
        e2f = tri.edge_to_face
        f_pts = tri.face_pts
        f_nei = tri.face_nei
        pts_coor = self.pts

        p1, p3 = np.int64(q1), np.int64(q3)
        key13 = (p1<<32)|p3
        key31 = (p3<<32)|p1

        t1 = e2f.get(key13)
        t2 = e2f.get(key31)

        if t1 is None or t2 is None: return False

        row1 = f_pts[t1]
        if row1[0] == q3: i=0
        elif row1[1] == q3: i=1
        else: i=2
        p4 = row1[(i+1)%3]

        row2 = f_pts[t2]
        if row2[0] == q1: j=0
        elif row2[1] == q1: j=1
        else: j=2
        p2 = row2[(j+1)%3]


        q1, q2, q3, q4 = pts_coor[p1], pts_coor[p2], pts_coor[p3], pts_coor[p4]
        turn_val1= (q3[0]-q2[0])*(q4[1]-q2[1]) - (q3[1]-q2[1])*(q4[0]-q2[0])
        if turn_val1<=0: return False
        turn_val2= (q1[0]-q2[0])*(q4[1]-q2[1]) - (q1[1]-q2[1])*(q4[0]-q2[0])
        return turn_val2 < 0

    def find_triangle_containing(self, tri, con:tuple):
        q1, q2 = con
        #tri = self.triangulations[tri_idx]
        e2f = tri.edge_to_face
        f_pts = tri.face_pts
        f_nei = tri.face_nei
        pts_coor = self.pts

        p1, p2 = np.int64(q1), np.int64(q2)
        if((p1<<32)|p2) in e2f or ((p2<<32)|p1) in e2f: return None

        p = tri.adj[q1]
        p = np.int64(p)
        #assert((min(p, q1), max(p, q1)) in tri.edges)

        t = e2f.get((p1<<32)|p)
        if t is None:
            t = e2f.get((p<<32)| p1)
        assert(t!=None)

        return _find_t_c(f_pts, f_nei, pts_coor, q1, q2, t)

    def flip_score(self, tri:FastTriangulation, tri_target:FastTriangulation, e:tuple, depth:int):
        p1, p3 = e

        e2f = tri.edge_to_face
        key13 = (np.int64(p1) << 32) | np.int64(p3)
        key31 = (np.int64(p3) << 32) | np.int64(p1)

        t1 = e2f.get(key13)
        t2 = e2f.get(key31)

        #if t1 is None or t2 is None: return (-999, depth)

        f_pts = tri.face_pts
        row1 = f_pts[t1]
        if row1[0] == p3: i = 0
        elif row1[1] == p3: i = 1
        else: i = 2
        p4 = row1[(i + 1) % 3]

        row2 = f_pts[t2]
        if row2[0] == p1: j = 0
        elif row2[1] == p1: j = 1
        else: j = 2
        p2 = row2[(j + 1) % 3]

        t1 = self.find_triangle_containing(tri_target, (p1, p3))
        t2 = self.find_triangle_containing(tri_target, (p2, p4))
        ori_cross=0
        new_cross=0
        f_pts = tri_target.face_pts
        f_nei = tri_target.face_nei
        pts_coor = self.pts
        if t1 is None:
            ori_cross=0
        else:
            ori_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p1,p3,t1)
        if t2 is None:
            new_cross=0
        else:
            new_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p2,p4,t2)
        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        if depth == 1:
            return m_score
        tri.flip(p1, p3)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                #nsc = self.flip_score(tri, target_idx, pe, 1)
                nsc = self.flip_score(tri, tri_target, pe, 1)
                m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))
        tri.flip(p2, p4)
        return m_score



    def parallel_flip_path2(self, start_idx, target_idx):
        tri = self.triangulations[start_idx].fast_copy()
        tri_target = self.triangulations[target_idx].fast_copy()
        pfp = []
        count=1
        prev_flip =set()
        e2f = tri.edge_to_face
        with Pool(processes=4,
                  initializer=init_worker,
                  initargs=(self.pts,
                            tri_target.face_pts, tri_target.face_nei, tri_target.edge_to_face, tri_target.adj)) as pool:
            while True:
                cand = []
                edges = list(tri.edges)
                #start= time.time()
                target_edges = [e for e in edges if e not in prev_flip]

                te_t1=[]
                te_t2=[]
                new_target_edges=[]
                for te in target_edges:
                    q1, q3 = te
                    key13 = (np.int64(q1)<<32)|np.int64(q3)
                    key31 = (np.int64(q3)<<32)|np.int64(q1)
                    t1 = e2f.get(key13)
                    t2 = e2f.get(key31)
                    if t1 is not None and t2 is not None:
                        new_target_edges.append(te)
                        te_t1.append(t1)
                        te_t2.append(t2)
                del target_edges

                if new_target_edges==[]: break
                c_size = max(1, min(200, int(len(new_target_edges)*0.01)))
                args_to_process = zip(new_target_edges, te_t1, te_t2, repeat(tri.face_pts))

                #args_to_process = zip(new_target_edges, te_t1, te_t2)
                results = pool.starmap(check_edge_score_numpy, args_to_process, chunksize=c_size)
                cand = [r for r in results if r is not None and r[1][0] >0]
                if not cand:
                    if prev_flip:
                        prev_flip=set()
                        continue
                    else: break
                cand.sort(key=lambda x: x[1], reverse=True)
                flips = []
                marked = set()
                #print(f"{count}: score takes:{time.time()-start:.2f}s", end=' ', flush=True)
                #start= time.time()
                for (p1, p2), _ in cand:
                    t1 = tri.find_face(p1, p2)
                    t2 = tri.find_face(p2, p1)
                    if t1 in marked or t2 in marked: continue
                    flips.append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                for e in flips:
                    p1, p2 = e
                    e1 = tri.flip(p1, p2)
                    prev_flip.add(e1)
                pfp.append(flips)
                #print(f"flip takes:{time.time()-start:.2f}s", end='\n')
                count+=1
        tri2 = self.triangulations[target_idx]
        assert(tri.edges == tri2.edges)
        return pfp


    def parallel_flip_path_rev2(self, start_idx, target_idx):
        tri = self.triangulations[start_idx].fast_copy()
        tri_target = self.triangulations[target_idx].fast_copy()
        rev_pfp=[]
        prev_flip =set()
        while True:
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if e in prev_flip:
                    continue
                if self.flippable(tri, e):
                    #score = self.flip_score(tri, target_idx, e, 1)#hy 0-> 1
                    score = self.flip_score(tri, tri_target, e, 1)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=set()
                    continue
                else: break
            cand.sort(key=lambda x: x[1],reverse=True)
            flips = []
            marked = set()
            for (p1, p3), _ in cand:
                t1 = tri.find_face(p1, p3)
                t2 = tri.find_face(p3, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p3))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                p1, p3 = e
                e1 = tri.flip(p1, p3)
                prev_flip.add(e1)
            rev_pfp.append(prev_flip)
        tri2 = self.triangulations[target_idx]
        assert(tri.edges == tri2.edges)
        rev_pfp.reverse()
        return rev_pfp

    def pfd_distribution(self, pfd):
        pfd_set = sorted(set(pfd))
        print("______ pfd distribution ________")
        for i, dist in enumerate(pfd_set):
            print(f"pfd {dist:03d} ({pfd.count(dist):03d}): ", end=' ')
            for _ in range(pfd.count(dist)):
                print("*", end='')
            all_indexs =[j for j in range(len(pfd)) if dist==pfd[j]]
            print(" (T:", end='')
            for idx in all_indexs:
                  print(idx, end=',')
            print(end=')\n')
        print()

    def inst_info(self):
        print(f"_________ {self.instance_name} info ___________")
        print(f"# of points: {len(self.pts_x)}")
        print(f"# of triangulations: {self.num_tris}")
        print(f"# of edges in T: {self.num_edges}")
        print(f"# of faces in T: {self.num_faces}")
        pfd = [len(x) for x in self.pFlips]
        max_pfd = max(pfd)
        max_pfd_tid = pfd.index(max_pfd)
        min_pfd = min(pfd)
        min_pfd_tid = pfd.index(min_pfd)
        print(f"max pfd T_{max_pfd_tid}: {max_pfd}")
        print(f"min pfd T_{min_pfd_tid}: {min_pfd}")




    # Do randomly flip edges with (flip_value) weight value
    def perturb_center3(self, flip_value, C_edges, best_dist):
        initial_center = self.triangulations[-1].fast_copy()
        flipped=0
        edge_list=[]
        for e in C_edges.keys():
            if C_edges[e]==flip_value:
                edge_list.append(e)
            if C_edges[e]>flip_value: break
        random.shuffle(edge_list)

        for e in edge_list:
            if self.flippable(initial_center, e):
                p1, p2 = e
                initial_center.flip(p1, p2)
                flipped+=1

        if flipped==0: return flipped, [], []
        # compute pfd sum
        all_pFlips=[]*(self.num_tris)
        for i in range(self.num_tris):
            pFlips=[]
            pFlips_paired1 = self.parallel_flip_path2(i, -1)
            pFlips_paired2 = self.parallel_flip_path_rev2(-1, i)
            pFlips_paired=[]
            if len(pFlips_paired1) < len(pFlips_paired2):
                pFlips_paired=pFlips_paired1
            else:
                pFlips_paired=pFlips_paired2
            for round_ in pFlips_paired:
                round_temp = [np.array(oneFlip).tolist() for oneFlip in round_]
                pFlips.append(round_temp)
            all_pFlips.append(pFlips)
        pfd = [len(f) for f in all_pFlips]
        if sum(pfd) <= best_dist:
            return flipped, pfd, all_pFlips
        return flipped, pfd, []

    def WriteData(self):
        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        self.instance_uid = self.instance_name
        inst["instance_uid"] = self.instance_uid


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": sum([len(pFlip) for pFlip in self.pFlips])} # , "input": self.input}

        path = '/Users/hyeyun/Experiment/PFD/hyeyun_git/'
        folder = "hy_solutions"
        with open(path+folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        #verify
        org_input = '/Users/hyeyun/Experiment/PFD/hyeyun_git/data/benchmark_instances/'+self.instance_name+'.json'
        with open(org_input, "r", encoding="utf-8") as f:
            root=json.load(f)

        instance = CGSHOP2026Instance(
            instance_uid=self.instance_uid,
            points_x=self.pts_x,
            points_y=self.pts_y,
            triangulations=root["triangulations"],
            )
        solution = CGSHOP2026Solution(
                instance_uid=self.instance_uid,
                flips=self.pFlips,
                )

        errors = check_for_errors(instance, solution)

        if errors != []:
            print(errors)
            exit(0)
        else: print("No errors")
        opt_folder = "hy_opt"
        opt_list = os.listdir(path+opt_folder)
        already_exist = False

        for sol in opt_list:
            if self.instance_uid+".solution.json" in sol:
                already_exist = True

                with open(path+opt_folder+"/"+sol, "r", encoding="utf-8") as ff:
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
            with open(path+opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                json.dump(inst, f, indent='\t')


def turn(p1, p2, p3):
    # negative: (p1, p2, p3) CW
    # positive: (p1, p2, p3) CCW
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])


@numba.njit
def _numba_count_cross(f_pts, f_nei, pts_coor, q1, q2, t):
    row = f_pts[t]
    if row[0] == q1: i=0
    elif row[1] == q1: i=1
    else: i=2

    idx_next = (i+1)%3
    tt = f_nei[t, idx_next]
    tmp = f_pts[t, idx_next]
    row_tt = f_pts[tt]
    if row_tt[0] == tmp: j=0
    elif row_tt[1] == tmp: j=1
    else: j=2

    p_q1 = pts_coor[q1]
    p_q2 = pts_coor[q2]

    cnt = 1
    while True:
        if row_tt[(j+1)%3] == q2: break
        cnt +=1
        t, i = tt, j
        idx_next = (i+1)%3

        tmp = f_pts[t, idx_next]
        p_tmp = pts_coor[tmp]
        turn_val= (p_q2[0]-p_q1[0])*(p_tmp[1]-p_q1[1]) - (p_q2[1]-p_q1[1])*(p_tmp[0]-p_q1[0])
        if turn_val <0:
            tt = f_nei[t, idx_next]
            tmp = f_pts[t, idx_next]
            row_tt = f_pts[tt]
            if row_tt[0] == tmp: j=0
            elif row_tt[1] == tmp: j=1
            else: j=2
        else:
            tt = f_nei[t, i%3]
            tmp = f_pts[t, i%3]
            row_tt = f_pts[tt]
            if row_tt[0] == tmp: j=0
            elif row_tt[1] == tmp: j=1
            else: j=2
    return cnt


@numba.njit
def _find_t_c(f_pts, f_nei, pts_coor, q1, q2, t):
    r1 = pts_coor[q1]
    r4 = pts_coor[q2]
    while True:
        row = f_pts[t]
        if row[0] == q1: i=0
        elif row[1] == q1: i=1
        else: i=2
        r2 = pts_coor[row[(i+1)%3]]
        r3 = pts_coor[row[(i+2)%3]]

        turn_val1= (r2[0]-r1[0])*(r4[1]-r1[1]) - (r2[1]-r1[1])*(r4[0]-r1[0])
        turn_val2= (r3[0]-r1[0])*(r4[1]-r1[1]) - (r3[1]-r1[1])*(r4[0]-r1[0])
        if turn_val1 < 0:
            t = f_nei[t, i]
        elif turn_val2 >0:
            t = f_nei[t, (i+2)%3]
        else:
            return t


