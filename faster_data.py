import json, os
from fast_Triangulation import *
import numba
import time
import random
import math
# from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
# from cgshop2026_pyutils.verify import check_for_errors
import copy
class FastData:
    def __init__(self, inp=''):
        if not inp:
            self.triangulations=[]
        else:
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
            self.instance_uid = root["instance_uid"]


            org_input = 'data/benchmark_instances/'+self.instance_uid+'.json'
            with open(org_input, "r", encoding="utf-8") as f:
                root=json.load(f)
            self.pts_x = root["points_x"]
            self.pts_y = root["points_y"]
            self.pts = np.array(list(zip(root["points_x"], root["points_y"])), dtype=np.float64) # can be replaced by pts_x,y?

            self.num_pts = len(root["points_x"])
            self.num_edges = len(root["triangulations"][0])
            self.num_faces = self.num_edges - self.num_pts + 1 #Euler Characteristic, F = E-V+1

            self.num_tris = len(root["triangulations"])
            self.triangulations = [None] * (self.num_tris)
            for i, t_data in enumerate(root["triangulations"]):
                self.triangulations[i] = self.make_triangulation(t_data)

            # self.inst_info()

            # restore center
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            #self.center = self.triangulations[min_flip_ind].fast_copy()
            self.center = self.make_triangulation(root["triangulations"][min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])

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

    def flip_score(self, tri:FastTriangulation, tri_target:FastTriangulation, e:tuple, depth:int, ver):
        p1, p3 = e

        e2f = tri.edge_to_face
        key13 = (np.int64(p1) << 32) | np.int64(p3)
        key31 = (np.int64(p3) << 32) | np.int64(p1)

        t1 = e2f.get(key13)
        t2 = e2f.get(key31)

        #if t1 is None or t2 is None: return (-999, depth)

        f_pts = tri.face_pts
        row1 = f_pts[t1]
        i=0
        if row1[0] == p3: i = 0
        elif row1[1] == p3: i = 1
        else: i = 2
        p4 = row1[(i + 1) % 3]

        row2 = f_pts[t2]
        j=0
        if row2[0] == p1: j = 0
        elif row2[1] == p1: j = 1
        else: j = 2
        p2 = row2[(j + 1) % 3]

        ori_cross=0
        new_cross=0
        f_pts = tri_target.face_pts
        f_nei = tri_target.face_nei
        pts_coor = self.pts

        ftc = self.find_triangle_containing(tri_target, (p1, p3))
        if ftc is None:
            ori_cross=0
        else:
            ori_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p1,p3,ftc)

        if depth == 0:
            return (ori_cross, 0)

        ftc = self.find_triangle_containing(tri_target, (p2, p4))
        if ftc is None:
            new_cross = 0
        else:
            new_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p2,p4,ftc)
        n_cross=0
        if ver==1:
            n_cross = ori_cross - new_cross
        elif ver==2:
            n_cross = (ori_cross - new_cross)/ori_cross if ori_cross>0 else -new_cross

        m_score = (n_cross, depth)
        if depth == 1:
            return m_score

        tri.flip(p1, p3)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                nsc = self.flip_score(tri, tri_target, pe, depth - 1, ver)
                m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))
        tri.flip(p2, p4)
        return m_score


    def parallel_flip_path(self, tri1, tri2):
        tri = tri1.fast_copy()
        tri_target = tri2.fast_copy()
        pfp = []
        while True:
            cand = []
            #edges = list(tri.edges)
            prev_flip = set()
            for e in tri.edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri_target, e, 1, 1)
                    if score[0] >0:
                        cand.append((e, score))
            if not cand:
                break
            cand.sort(key=lambda x: x[1], reverse=True)
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_face(p1, p2)
                t2 = tri.find_face(p2, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            for e in flips:
                p1, p2 = e
                tri.flip(p1, p2)
            pfp.append(flips)
        assert(tri.edges == tri_target.edges)
        return pfp

    def parallel_flip_path_reverse(self, tri1, tri2):
        tri = tri2.fast_copy()
        pfp=[]
        while True:
            cand = []
            prev_flip2=[]
            for e in tri.edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri1, e, 1, 1)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand: break
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
            flips_reverse = []
            for e in flips:
                p1, p3 = e
                e1 = tri.flip(p1,p3)
                flips_reverse.append((int(e1[0]), int(e1[1])))
            pfp.append(flips_reverse)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]

    def parallel_flip_path2(self, tri1, tri2):
        tri = tri1.fast_copy()
        tri_target = tri2.fast_copy()
        pfp = []
        prev_flip = set()
        step=0
        while True:
            step+=1
            cand = []
            #edges = list(tri.edges)
            for e in tri.edges:
                if e in prev_flip: continue
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri_target, e, 0, 1)
                    if score[0] >0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=set()
                    continue
                else: break
            cand.sort(key=lambda x: x[1], reverse=True)
            # if step>100:
                # print(cand)
                # print(prev_flip)
            flips = []
            marked = set()
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
        assert(tri.edges == tri_target.edges)
        return pfp


    def parallel_flip_path2_reverse(self, tri1, tri2):
        #tri = tri1.fast_copy()
        #tri_target = tri2.fast_copy()
        tri = tri2.fast_copy()
        pfp=[]
        step=0
        prev_flip = set()
        while True:
            step+=1
            cand = []
            #edges = list(tri.edges)
            for e in tri.edges:
                if e in prev_flip: continue
                if self.flippable(tri, e):
                    #score = self.flip_score(tri, tri_target, e, 0)
                    score = self.flip_score(tri, tri1, e, 0, 1)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=[]
                    continue
                else: break
            cand.sort(key=lambda x: x[1],reverse=True)
            # if step>100:
            #     print("in rev2:", cand)
            #     print(prev_flip)

            flips = []
            marked = set()
            for (p1, p3), _ in cand:
                t1 = tri.find_face(p1, p3)
                t2 = tri.find_face(p3, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p3))
                marked.add(t1)
                marked.add(t2)
            prev_flip=[]
            for e in flips:
                p1, p3 = e
                e1 = tri.flip(p1, p3)
                prev_flip.append((int(e1[0]), int(e1[1])))
            pfp.append(prev_flip)
            prev_flip = set(prev_flip)
        #assert(tri.edges == tri_target.edges)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]


    def parallel_flip_path3(self, tri1, tri2):
        tri = tri1.fast_copy()
        tri_target = tri2.fast_copy()
        pfp = []
        step=0
        while True:
            step+=1
            assert step<1000, f"Too many steps in parallel_flip_path3 for {self.instance_uid}"
            cand = []
            #edges = list(tri.edges)
            for e in tri.edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri_target, e, 1, 2)
                    if score[0] >0:
                        cand.append((e, score))
            if not cand:break
            cand.sort(key=lambda x: x[1], reverse=True)
            flips = []
            marked = set()
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
            pfp.append(flips)
        assert(tri.edges == tri_target.edges)
        return pfp

    def parallel_flip_path3_reverse(self, tri1, tri2):
        #tri = tri1.fast_copy()
        #tri_target = tri2.fast_copy()
        tri = tri2.fast_copy()
        pfp = []
        step=0
        while True:
            step+=1
            assert step<1000, f"Too many steps in parallel_flip_path3 for {self.instance_uid}"
            cand = []
            #edges = list(tri.edges)
            for e in tri.edges:
                if self.flippable(tri, e):
                    #score = self.flip_score(tri, tri_target, e, 1)
                    score = self.flip_score2(tri, tri1, e, 1, 2)
                    if score[0] >0:
                        cand.append((e, score))
            if not cand:break
            cand.sort(key=lambda x: x[1], reverse=True)
            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_face(p1, p2)
                t2 = tri.find_face(p2, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            pfp1=[]
            for e in flips:
                p1, p2 = e
                e1 = tri.flip(p1, p2)
                pfp1.append((int(e1[0]), int(e1[1])))
            pfp.append(pfp1)
        #assert(tri.edges == tri_target.edges)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]


    def computePFS_total(self, tri1, tri2, tri_num):
        T1 = tri1.fast_copy()
        T2 = tri2.fast_copy()

        start=time.time()
        pFlips_paired1 = self.parallel_flip_path(T1, T2)
        #print(f"1 takes {time.time()-start:.2f}s")
        start=time.time()
        pFlips_paired11 = self.parallel_flip_path_reverse(T1, T2)
        #print(f"2 takes {time.time()-start:.2f}s")
        start=time.time()

        # pFlips_paired2 = self.parallel_flip_path2(T1, T2)
        # #print(f"3 takes {time.time()-start:.2f}s")
        # start=time.time()
        # pFlips_paired21 = self.parallel_flip_path2_reverse(T1, T2)
        # #print(f"4 takes {time.time()-start:.2f}s")
        # start=time.time()

        # pFlips_paired3 = self.parallel_flip_path3(T1, T2)
        # #print(f"5 takes {time.time()-start:.2f}s")
        # start=time.time()
        # pFlips_paired31 = self.parallel_flip_path3_reverse(T1, T2)
        # #print(f"6 takes {time.time()-start:.2f}s")
        # start=time.time()


        path_list = [pFlips_paired1,pFlips_paired11]
        random.shuffle(path_list)
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]


        prev_pFlips_i=[]
        for round_ in pFlips_paired:
            round_temp=[]
            for oneFlip in round_:
                round_temp.append(list(oneFlip))
            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i


    def random_compute_fpd_replace(self):
        prev_len = self.dist
        prev_best = prev_len
        tri_num=0

        while tri_num < self.num_tris:
            # print("tri_num = ", tri_num)
            seq = self.pFlips[tri_num]
            if len(seq) == 0 or len(seq) == 1:
                tri_num +=1
                continue
            seq_iter =1
            local_T = self.triangulations[tri_num].fast_copy()
            for e in seq[0]:
                local_T.flip(e[0], e[1])
            minlen = len(seq)
            cnt = 1
            while seq_iter < len(seq):
                # print(seq_iter, end = ' ', flush =True)
                start=time.time()
                seq1 = self.computePFS_total(self.triangulations[tri_num], local_T, tri_num)
                seq2 = self.computePFS_total(local_T, self.center, tri_num)
                # print(f"{time.time()-start:.2f}s", end=' ', flush=True)
                if (len(seq1) + len(seq2)) == minlen:
                    if random.choice([False] * cnt + [True]):
                        self.pFlips[tri_num] = seq1 + seq2
                    cnt += 1
                elif (len(seq1) + len(seq2)) < minlen:
                    #if tri_num==7:
                    #    print("pFlips[", tri_num, "] updated!: ", len(self.pFlips[tri_num]), "->", len(seq1)+len(seq2))
                    #    print(self.pFlips[tri_num])
                    #    print()
                    #    print(seq1+seq2)
                    #    print("_____________________")
                    self.pFlips[tri_num] = seq1 + seq2
                    minlen = len(seq1) + len(seq2)
                    cnt = 1
                    break
                for e in seq[seq_iter]:
                    local_T.flip(e[0], e[1])
                seq_iter +=1
            # print()
            if seq_iter == len(seq):
                tri_num +=1
        #return self.center


    def findCenterGlobal(self):
        mtriangulations = [t.fast_copy() for t in self.triangulations]
        num = len(self.triangulations)
        pfps = [ [] for _ in range(num)]

        F_E =[]
        for i in range(num):
            tri = mtriangulations[i]
            fe = []
            for e in tri.edges:
                if self.flippable(tri, e):
                    fe.append(e)
            F_E.append(fe)

        mscore = 0
        flips =[]
        NCAND=[]
        mi=0
        for i in range(num):
            ncand =[]
            nscore = 0
            tri = mtriangulations[i]
            for e in F_E[i]:
                escore=0
                for j in range(num):
                    if i==j: continue
                    score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                    escore +=score
                ncand.append((e, escore))
            ncand.sort(key = lambda x:x[1], reverse=True)
            # print("T", i,"done.")
            NCAND.append(ncand)
            marked = set()
            flp =[]
            for (p1, p2), escore in ncand:
                if escore <= 0: break
                t1 = tri.find_face(p1, p2)
                t2 = tri.find_face(p2, p1)
                if t1 in marked or t2 in marked: continue
                flp.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
                nscore += escore
            flips.append(flp)
            if nscore > mscore:
                mscore = nscore
                mi = i
        if mscore ==0:
            self.pFlips=[]
            for i in range(num):
                if i< num-1:
                    assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
                self.pFlips.append(pfps[i])
            return mtriangulations[0]

        prev_mtri = mtriangulations[mi].fast_copy()

        pfps[mi].append(flips[mi])
        for e in flips[mi]:
            mtriangulations[mi].flip(e[0], e[1])
        fe =[]
        tri = mtriangulations[mi]
        for ee in tri.edges:
            if self.flippable(tri, ee):
                fe.append(ee)
        F_E[mi] = fe
        # print("count, mscore = ", 0, mscore)
        # print()

        count=1
        while True:
            mscore=0
            flips =[]
            current_mi = mi
            for i in range(num):
                ncand=[]
                nscore = 0
                tri = mtriangulations[i]
                if i!= current_mi:
                    # print("T", i, ": prev_ncand = ", prev_ncand)
                    for e, escore in NCAND[i]:
                        prev_e_score, _ = self.flip_score(tri, prev_mtri, e, 1, 1)
                        current_e_score, _ = self.flip_score(tri, mtriangulations[current_mi], e, 1, 1)
                        escore += current_e_score - prev_e_score
                        ncand.append((e, escore))
                else:#i==current_mi
                    ncand=[]
                    for e in F_E[i]:
                        escore=0
                        for j in range(num):
                            if i==j: continue
                            score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                            escore +=score
                        if escore >0:
                            ncand.append((e, escore))
                ncand.sort(key = lambda x:x[1], reverse=True)
                # if count<=2:
                #     print("T", i,", current_mi = ", current_mi, ", ncand = ",  ncand)
                # else:
                #     exit(0)
                NCAND[i] = ncand

                marked = set()
                flp =[]
                for (p1, p2), escore in ncand:
                    if escore <= 0: break
                    t1 = tri.find_face(p1, p2)
                    t2 = tri.find_face(p2, p1)
                    if t1 in marked or t2 in marked: continue
                    flp.append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                    nscore += escore
                flips.append(flp)
                if nscore > mscore:
                    mscore = nscore
                    mi = i
            # print("---- after for loop")
            if mscore ==0: break

            pfps[mi].append(flips[mi])
            prev_mtri = mtriangulations[mi].fast_copy()
            for e in flips[mi]:
                mtriangulations[mi].flip(e[0], e[1])
            fe =[]
            tri = mtriangulations[mi]
            for ee in tri.edges:
                if self.flippable(tri, ee):
                    fe.append(ee)
            F_E[mi] = fe
            # print("count, mscore = ", count, mscore)
            # print()
            count+=1
        self.pFlips=[]
        for i in range(num):
            if i< num-1:
                assert mtriangulations[i].edges == mtriangulations[i+1].edges, "findCenterGlobal ended with different centers"
            self.pFlips.append(pfps[i])
        return mtriangulations[0]

    def old_findCenterGlobal(self):
        mtriangulations = [t.fast_copy() for t in self.triangulations]
        num = len(self.triangulations)
        pfps = [ [] for _ in range(num)]

        F_E =[]
        for i in range(num):
            tri = mtriangulations[i]
            fe = []
            for e in tri.edges:
                if self.flippable(tri, e):
                    fe.append(e)
            F_E.append(fe)

        while True:
            mscore = 0
            #flips = [ [] for _ in range(num)]
            flips =[]
            for i in range(num):
                ncand =[]
                nscore = 0
                tri = mtriangulations[i]
                for e in F_E[i]:
                    escore=0
                    for j in range(num):
                        if i==j: continue
                        score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                        escore +=score
                    if escore >0:
                        ncand.append((e, escore))
                ncand.sort(key = lambda x:x[1], reverse=True)
                marked = set()
                flp =[]
                for (p1, p2), escore in ncand:
                    t1 = tri.find_face(p1, p2)
                    t2 = tri.find_face(p2, p1)
                    if t1 in marked or t2 in marked: continue
                    flp.append((p1, p2))
                    #flips[i].append((p1, p2))
                    marked.add(t1)
                    marked.add(t2)
                    nscore += escore
                flips.append(flp)
                if nscore > mscore:
                    mscore = nscore
                    mi = i
            if mscore ==0: break
            pfps[mi].append(flips[mi])
            for e in flips[mi]:
                mtriangulations[mi].flip(e[0], e[1])
            fe =[]
            tri = mtriangulations[mi]
            for ee in tri.edges:
                if self.flippable(tri, ee):
                    fe.append(ee)
            F_E[mi] = fe

        self.pFlips=[]
        for i in range(num):
            if i< num-1:
                assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
            self.pFlips.append(pfps[i])
        return mtriangulations[0]

    def findCenterGlobal2(self):
        # if self.log:
        #     start = time.time()
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
                    score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                    escore += score
                scores[i][e] = escore
            # if self.log:
            #     print(i, "/", num)
            #     end = time.time()
            #     print('time:', f"{end - start:.5f} sec")
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
                    t1 = tri.find_face(p1, p2)
                    t2 = tri.find_face(p2, p1)
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
            # if self.log:
            #     print(tl)
            #     print(mscore)
            #     end = time.time()
            #     print('time:', f"{end - start:.5f} sec")
            pfps[mi].append(mflips)
            for i in range(num):
                if i == mi:
                    continue
                tri = mtriangulations[i]
                edges = list(tri.edges)
                for e in edges:
                    if not self.flippable(tri, e): continue
                    score, _ = self.flip_score(tri, mtriangulations[mi], e, 1, 1)
                    scores[i][e] -= score
            for e in mflips:
                mtriangulations[mi].flip(*e)
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
                            score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                            escore += score
                        scores[i][e] = escore
                else:
                    tri = mtriangulations[i]
                    edges = list(tri.edges)
                    for e in edges:
                        if not self.flippable(tri, e): continue
                        score, _ = self.flip_score(tri, mtriangulations[mi], e, 1, 1)
                        scores[i][e] += score
        self.pFlips = []
        for i in range(num):
            if i < num-1:
                assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
            self.pFlips.append(pfps[i])
        
        # if self.log: print("total length:",tl)
        while len(mtriangulations) > 1:
            tri = mtriangulations.pop()
            del tri
        return mtriangulations[0]

    def flip_rev(self, tri, e):
        p, q = e
        t = self.find_triangle_containing(tri, e)
        row = f_pts[t]
        if row[0] == q1: i=0
        elif row[1] == q1: i=1
        else: i=2
        p1 = row[(i+1)%3]
        p2 = row[(i+2)%3]
        tri.flip(p1, p2)


    def random_new_center(self):
        param = 1
        len_flips = [len(pFlip) for pFlip in self.pFlips]
        max_dist = max(len_flips)
        total_dist = sum(len_flips)
        while param < max_dist *2:
            # print("-----param ", param)
            revnum = [random.randint(min(param, len(self.pFlips[i]), 1) , min(param, len(self.pFlips[i]))) for i in range(len(self.triangulations))]
            newD = FastData()
            newD.pts = self.pts
            for i in range(self.num_tris):
                # print("newT", i)
                newT = self.center.fast_copy()
                for j in range(revnum[i]):
                    for e in self.pFlips[i][-j-1]:
                        p, q = e
                        t = self.find_triangle_containing(newT, e)
                        row = newT.face_pts[t]
                        pi=0
                        if row[0] == p: pi=0
                        elif row[1] == p: pi=1
                        else: pi=2
                        p1 = row[(pi+1)%3]
                        p2 = row[(pi+2)%3]
                        newT.flip(p1, p2)
                newD.triangulations.append(newT)
            start=time.time()
            # print("findCenterGlobal() takes ... ", end=' ', flush=True)
            self.center = newD.findCenterGlobal2()
            # print(f"{time.time()-start:.2f}s")
            for i in range(self.num_tris):
                self.pFlips[i] = self.pFlips[i][:-revnum[i]] + newD.pFlips[i]
            start=time.time()
            # print("random_compute_pfd_replace()... ")
            self.random_compute_fpd_replace() # pFlip update
            # print(f"it takes {time.time()-start:.2f}s", end='\n')
            new_dist = sum([len(pFlip) for pFlip in self.pFlips])
            if total_dist != new_dist:
                self.dist = new_dist
                break
            param *=2


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
        print(f"_________ {self.instance_uid} info ___________")
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
        print(f"____________________\n")





    def WriteData(self):
        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_uid


        inst["flips"] = self.pFlips
        inst["meta"] = {"dist": sum([len(pFlip) for pFlip in self.pFlips])} # , "input": self.input}

        path = ''
        folder = "solutions"
        with open(path+folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        #verify
        org_input = 'data/benchmark_instances/'+self.instance_uid+'.json'
        with open(org_input, "r", encoding="utf-8") as f:
            root=json.load(f)

        # instance = CGSHOP2026Instance(
        #     instance_uid=self.instance_uid,
        #     points_x=self.pts_x,
        #     points_y=self.pts_y,
        #     triangulations=root["triangulations"],
        #     )
        # solution = CGSHOP2026Solution(
        #         instance_uid=self.instance_uid,
        #         flips=self.pFlips,
        #         )

        # errors = check_for_errors(instance, solution)

        # if errors != []:
        #     print(errors)
        #     exit(0)
        # else: print("No errors")
        opt_folder = "opt"
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
                    os.remove(path+opt_folder+"/"+sol)
                    with open(path+opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
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
    i=0
    if row[0] == q1: i=0
    elif row[1] == q1: i=1
    else: i=2

    idx_next = (i+1)%3
    tt = f_nei[t, idx_next]
    tmp = f_pts[t, idx_next] # == t.pt(i+1)
    row_tt = f_pts[tt]
    j=0
    if row_tt[0] == tmp: j=0
    elif row_tt[1] == tmp: j=1
    else: j=2

    p_q1 = pts_coor[q1]
    p_q2 = pts_coor[q2]

    cnt = 1
    A = p_q2[0]-p_q1[0]
    B = p_q2[1]-p_q1[1]
    while True:
        if row_tt[(j+1)%3] == q2:break
        cnt +=1
        t, i = tt, j
        idx_next = (i+1)%3

        tmp = f_pts[t, idx_next]
        p_tmp = pts_coor[tmp]
        turn_val= A*(p_tmp[1]-p_q1[1]) - B*(p_tmp[0]-p_q1[0])
        if turn_val <0:
            tt = f_nei[t, idx_next]
            row_tt = f_pts[tt]
            j=0
            if row_tt[0] == tmp: j=0
            elif row_tt[1] == tmp: j=1
            else: j=2
        else:
            tt = f_nei[t, i%3]
            tmp = f_pts[t, i]
            row_tt = f_pts[tt]
            j=0
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
            return t #face_idx


