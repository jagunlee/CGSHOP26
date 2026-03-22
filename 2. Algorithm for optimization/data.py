import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from triangulation import *
import numba
from numba.typed import Dict, List
from numba import types as typ
import time
import random
import math
# from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
# from cgshop2026_pyutils.verify import check_for_errors

def init_worker(tri_data, center_data, pfp, pts):
    global gb_tris, gb_center, gb_pFlips, gb_pts_coor, gb_tris
    gb_tris = tri_data
    gb_center = center_data
    gb_pFlips = pfp
    gb_pts_coor = pts


def process_(tri_num):
    tri_obj = gb_tris[tri_num]
    local_T = gb_tris[tri_num].fast_copy()
    center = gb_center

    seq = gb_pFlips[tri_num]
    for e in seq[0]:
        local_T.flip(e[0], e[1])
    updated_pFlips=seq
    start=time.time()
    for seq_iter in range(1, len(seq)):
        seq1, win1 = _computePFS_total(tri_obj, local_T)
        seq2, win2 = _computePFS_total(local_T, center)
        if (len(seq1) + len(seq2)) <= len(updated_pFlips):
            updated_pFlips = seq1 + seq2
        for e in seq[seq_iter]:
            local_T.flip(e[0], e[1])
    return tri_num, updated_pFlips, time.time()-start

def _computePFS_total(T1, T2):
    pts = gb_pts_coor

    t1 = T1.fast_copy()
    t2 = T2.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=False
    pFlips_paired1 = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)
    t1 = T2.fast_copy()
    t2 = T1.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=True
    tmp = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)
    pFlips_paired11 = tmp[::-1]


    path_list = [pFlips_paired1, pFlips_paired11]

    opt_ind = np.argmin([len(x) for x in path_list])
    pFlips_paired = path_list[opt_ind]
    prev_pFlips_i=[]
    for round_ in pFlips_paired:
        round_temp=[]
        for oneFlip in round_:
            round_temp.append(list(oneFlip))
        prev_pFlips_i.append(round_temp)
    return prev_pFlips_i, opt_ind

def process_typed_dict(data):
    result_dict = Dict.empty(typ.int64, typ.int64)
    for key, value in data.items():
        result_dict[key] = value
    return result_dict

@numba.njit
def _flip(p1, p2, f_pts, f_nei, e2f, adj):
    # flip (p1,p2) -> new edge (p3, p4)
    key12 = (p1 << 32) | (p2)
    key21 = (p2 << 32) | (p1)
    t1 = e2f[key12]
    t2 = e2f[key21]

    row1 = f_pts[t1]
    i=0
    if row1[0] == p2: i=0
    elif row1[1] == p2: i=1
    else: i=2
    p3 = int(row1[(i+1)%3])

    row2 = f_pts[t2]
    j=0
    if row2[0] == p1: j=0
    elif row2[1] == p1: j=1
    else: j=2
    p4 = int(row2[(j+1)%3])

    n_p2p3 = f_nei[t1, i] 
    n_p3p1 = f_nei[t1, (i+1)%3]

    m_p1p4 = f_nei[t2, j] 
    m_p4p2 = f_nei[t2, (j+1)%3]

    f_pts[t1,i] = np.int32(p4)
    f_pts[t2,j] = np.int32(p3)

    f_nei[t1, i] = np.int32(t2)
    f_nei[t1, (i+2)%3] = m_p1p4
    f_nei[t2, j] = np.int32(t1)
    f_nei[t2, (j+2)%3] = n_p2p3


    if m_p1p4 != -1:
        row = f_pts[m_p1p4]
        ii=0
        if row[0]==p4: ii=0
        elif row[1]==p4: ii=1
        else: ii=2
        f_nei[m_p1p4, ii] = np.int32(t1)

    if n_p2p3 != -1:
        jj=0
        row = f_pts[n_p2p3]
        if row[0]==p3: jj=0
        elif row[1]==p3: jj=1
        else: jj=2
        f_nei[n_p2p3, jj] = np.int32(t2)

    before = len(e2f)
    del e2f[key12]
    del e2f[key21]

    e2f[(p1 << 32) | p4] = t1
    e2f[(p4 << 32) | p3] = t1
    e2f[(p2 << 32) | p3] = t2
    e2f[(p3 << 32) | p4] = t2

    e1 = (min(p3, p4), max(p3, p4))
    adj[p1] = np.int32(p3)
    adj[p2] = np.int32(p3)

    if before != len(e2f):
        print("assert!!, before != len(e2f)")
        return (-1,-1)
    return e1


@numba.njit
def _flippable_fast(e, p2, p4, pts_coor):
    p1, p3 = e

    q1, q2, q3, q4 = pts_coor[p1], pts_coor[p2], pts_coor[p3], pts_coor[p4]
    turn_val1= (q3[0]-q2[0])*(q4[1]-q2[1]) - (q3[1]-q2[1])*(q4[0]-q2[0])
    if turn_val1<=0: return False
    turn_val2= (q1[0]-q2[0])*(q4[1]-q2[1]) - (q1[1]-q2[1])*(q4[0]-q2[0])
    return turn_val2 < 0



@numba.njit
def _numba_count_cross_fast(f_pts, f_nei, q1, q2, t, pts_coor):
    row = f_pts[t]
    i=0
    if row[0] == q1: i=0
    elif row[1] == q1: i=1
    else: i=2

    idx_next = (i+1)%3
    tt = f_nei[t, idx_next]
    tmp = f_pts[t, idx_next] 
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
def _find_t_c_fast(f_pts, f_nei, t, con, pts_coor):
    q1, q2 = con

    r1 = pts_coor[q1]
    r4 = pts_coor[q2]
    A = (r4[1]-r1[1])
    B = (r4[0]-r1[0])
    while True:
        row = f_pts[t]
        i=0
        if row[0] == q1: i=0
        elif row[1] == q1: i=1
        else: i=2
        r2 = pts_coor[row[(i+1)%3]]
        r3 = pts_coor[row[(i+2)%3]]

        turn_val1= (r2[0]-r1[0])*A - (r2[1]-r1[1])*B
        turn_val2= (r3[0]-r1[0])*A - (r3[1]-r1[1])*B
        if turn_val1 < 0:
            t = f_nei[t, i]
        elif turn_val2 >0:
            t = f_nei[t, (i+2)%3]
        else:
            return t #face_idx

@numba.njit
def _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor, ver):
    p1, p3 = e

    ### find_triangle_containing() ###
    e2f = tg_e2f
    f_pts = tg_f_pts
    f_nei = tg_f_nei
    adj = tg_adj

    key13 = (p1 << 32) | p3
    key31 = (p3 << 32) | p1

    if key13 in e2f or key31 in e2f:
        t1=-1
    else:
        p = int(adj[p1])
        key_a = (p1 << 32) | p
        key_b = (p << 32) | p1

        if key_a in e2f:
            t = e2f[key_a]
            t1 = _find_t_c_fast(f_pts, f_nei, t, (p1, p3), pts_coor)
        elif key_b in e2f:
            t = e2f[key_b]
            t1 = _find_t_c_fast(f_pts, f_nei, t, (p1, p3), pts_coor)
        else:
            t1 = -1
    #################################

    ori_cross=0
    new_cross=0

    if t1 == -1:
        ori_cross=0
    else:
        ori_cross = _numba_count_cross_fast(f_pts, f_nei, p1,p3,t1, pts_coor)
    if depth==0:
        return (ori_cross, 0)


    key24 = (p2 << 32) | p4
    key42 = (p4 << 32) | p2
    if key24 in e2f or key42 in e2f:
        t2=-1
    else:
        p = int(adj[p2])
        key_a = (p2 << 32) | p
        key_b = (p << 32) | p2

        if key_a in e2f:
            t = e2f[key_a]
            t2 = _find_t_c_fast(f_pts, f_nei, t, (p2, p4), pts_coor)
        elif key_b in e2f:
            t = e2f[key_b]
            t2 = _find_t_c_fast(f_pts, f_nei, t, (p2, p4), pts_coor)
        else:
            t2 = -1

    if t2 == -1:
        new_cross=0
    else:
        new_cross = _numba_count_cross_fast(f_pts, f_nei, p2,p4,t2, pts_coor)
    n_cross=0
    if ver==1:
        n_cross = ori_cross - new_cross
    elif ver==2:
        n_cross = (ori_cross - new_cross)/ori_cross if ori_cross>0 else -new_cross
    m_score = (n_cross, depth)
    if depth==1:
        return m_score


@numba.njit
def _parallel_flip_path(f_pts, f_nei, e2f, adj, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, pts_coor, depth, reverse):
    pfp = List()
    pfp.append(List([(1,2)]))
    pfp.pop(0)
    while True:
        cand = List()
        cand.append((1.0,(1,2)))
        cand.pop(0)
        for E_key, _ in e2f.items():
            q1, q3 = E_key>>32, E_key & 0xFFFFFFFF
            e = (q1, q3)
            key13 = (q1<<32)|q3
            key31 = (q3<<32)|q1
            if key13 not in e2f or key31 not in e2f: continue
            t1 = e2f[key13]
            t2 = e2f[key31]

            row1 = f_pts[t1]
            if row1[0] == q3: i=0
            elif row1[1] == q3: i=1
            else: i=2
            row2 = f_pts[t2]
            if row2[0] == q1: j=0
            elif row2[1] == q1: j=1
            else: j=2
            p4 = int(row1[(i+1)%3])
            p2 = int(row2[(j+1)%3])
            if _flippable_fast(e, p2, p4, pts_coor):
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor, 1)
                if score >0:
                    cand.append((score, e))
        if len(cand)==0:
            break
        cand.sort()
        flips = List()
        flips.append((1,2))
        flips.pop(0)
        marked = set()
        for ci in range(len(cand)-1, -1, -1):
            _, E = cand[ci]
            p1, p2 = E
            #find_face
            key12 = (p1<<32)|p2
            key21 = (p2<<32)|p1
            t1 = e2f[key12]
            t2 = e2f[key21]
            if t1 in marked or t2 in marked: continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)
        flips_reverse = List()
        flips_reverse.append((1,2))
        flips_reverse.pop(0)
        for e in flips:
            p1, p2 = e
            e1 = _flip(p1, p2, f_pts, f_nei, e2f, adj)
            if reverse:
                flips_reverse.append(e1)
        if reverse:
            pfp.append(flips_reverse)
        else:
            pfp.append(flips)
    return pfp


@numba.njit
def _parallel_flip_path2(f_pts, f_nei, e2f, adj, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, pts_coor, depth, reverse):
    pfp = List()
    pfp.append(List([(1,2)]))
    pfp.pop(0)

    prev_flip = List()
    prev_flip.append((1,2))
    prev_flip.pop(0)
    test=0
    count=0
    while True:
        cand = List()
        cand.append((1.0,(1,2)))
        cand.pop(0)
        for E_key, _ in e2f.items():
            q1, q3 = E_key>>32, E_key & 0xFFFFFFFF

            e = (q1, q3)
            if e in prev_flip: continue

            key13 = (q1<<32)|q3
            key31 = (q3<<32)|q1
            if key13 not in e2f or key31 not in e2f: continue
            t1 = e2f[key13]
            t2 = e2f[key31]

            row1 = f_pts[t1]
            if row1[0] == q3: i=0
            elif row1[1] == q3: i=1
            else: i=2
            row2 = f_pts[t2]
            if row2[0] == q1: j=0
            elif row2[1] == q1: j=1
            else: j=2
            p4 = int(row1[(i+1)%3])
            p2 = int(row2[(j+1)%3])

            if _flippable_fast(e, p2, p4, pts_coor):
                test+=1
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor, 1)
                if score >0:
                    cand.append((score, e))
        if len(cand)==0:
            if len(prev_flip)!=0:
                prev_flip=List()
                prev_flip.append((1,2))
                prev_flip.pop(0)
                continue
            else:
                break
        cand.sort()
        flips = List()
        flips.append((1,2))
        flips.pop(0)
        marked = set()
        for ci in range(len(cand)-1, -1, -1):
            _, E = cand[ci]
            p1, p2 = E
            #find_face
            key12 = (p1<<32)|p2
            key21 = (p2<<32)|p1
            t1 = e2f[key12]
            t2 = e2f[key21]
            if t1 in marked or t2 in marked: continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)
        if reverse:
            flips_reverse = List()
            flips_reverse.append((1,2))
            flips_reverse.pop(0)

        for e in flips:
            p1, p2 = e
            e1 = _flip(p1, p2, f_pts, f_nei, e2f, adj)

            prev_flip.append(e1)
            if reverse:
                flips_reverse.append(e1)
        if reverse:
            pfp.append(flips_reverse)
        else:
            pfp.append(flips)
    return pfp


@numba.njit
def _parallel_flip_path3(f_pts, f_nei, e2f, adj, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, pts_coor, depth, reverse):
    pfp = List()
    pfp.append(List([(1,2)]))
    pfp.pop(0)
    while True:
        cand = List()
        cand.append((1.0,(1,2)))
        cand.pop(0)
        for E_key, _ in e2f.items():
            q1, q3 = E_key>>32, E_key & 0xFFFFFFFF
            e = (q1, q3)
            key13 = (q1<<32)|q3
            key31 = (q3<<32)|q1
            if key13 not in e2f or key31 not in e2f: continue
            t1 = e2f[key13]
            t2 = e2f[key31]

            row1 = f_pts[t1]
            if row1[0] == q3: i=0
            elif row1[1] == q3: i=1
            else: i=2
            row2 = f_pts[t2]
            if row2[0] == q1: j=0
            elif row2[1] == q1: j=1
            else: j=2
            p4 = int(row1[(i+1)%3])
            p2 = int(row2[(j+1)%3])
            if _flippable_fast(e, p2, p4, pts_coor):
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor, 2)
                if score >0:
                    cand.append((score, e))
        if len(cand)==0:
            break
        cand.sort()
        flips = List()
        flips.append((1,2))
        flips.pop(0)
        marked = set()
        for ci in range(len(cand)-1, -1, -1):
            _, E = cand[ci]
            p1, p2 = E
            #find_face
            key12 = (p1<<32)|p2
            key21 = (p2<<32)|p1
            t1 = e2f[key12]
            t2 = e2f[key21]
            if t1 in marked or t2 in marked: continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)
        flips_reverse = List()
        flips_reverse.append((1,2))
        flips_reverse.pop(0)
        for e in flips:
            p1, p2 = e
            e1 = _flip(p1, p2, f_pts, f_nei, e2f, adj)
            if reverse:
                flips_reverse.append(e1)
        if reverse:
            pfp.append(flips_reverse)
        else:
            pfp.append(flips)
    return pfp



class FastData:
    def __init__(self, inp=''):
        if not inp:
            self.triangulations=[]
        else:
            self.input = inp
            self.pts = None 
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


            # restore center
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.make_triangulation(root["triangulations"][min_flip_ind])
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])
            self.inst_info()


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
                            p1 = int(face_pts[l])
                            p2 = int(face_pts[(l+1)%3])
                            key12 = ((p1)<<32)|(p2)
                            # save edge_to_face
                            e2f[key12] = face_idx

                            key21 = ((p2)<<32)|(p1)
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
        e2f = tri.edge_to_face
        f_pts = tri.face_pts
        f_nei = tri.face_nei
        pts_coor = self.pts

        p1, p2 = np.int64(q1), np.int64(q2)
        if((p1<<32)|p2) in e2f or ((p2<<32)|p1) in e2f: return None

        p = tri.adj[q1]
        p = np.int64(p)

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

    def total_flip_score(self, tri:FastTriangulation, tri_target_list, e:tuple):
        start=time.time()
        p1, p3 = e

        e2f = tri.edge_to_face
        key13 = (np.int64(p1) << 32) | np.int64(p3)
        key31 = (np.int64(p3) << 32) | np.int64(p1)

        t1 = e2f.get(key13)
        t2 = e2f.get(key31)

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

        total_score=0
        pts_coor = self.pts
        for target_idx in tri_target_list:
            tri_target = self.triangulations[target_idx]
            ori_cross=0
            new_cross=0

            f_pts = tri_target.face_pts
            f_nei = tri_target.face_nei

            ftc = self.find_triangle_containing(tri_target, (p1, p3))
            if ftc is None:
                ori_cross=0
            else:
                ori_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p1,p3,ftc)

            ftc = self.find_triangle_containing(tri_target, (p2, p4))
            if ftc is None:
                new_cross = 0
            else:
                new_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p2,p4,ftc)

            n_cross = ori_cross - new_cross
            total_score += n_cross
        return total_score, time.time()-start

    def my_total_flip_score(self, current_mi, mtriangulations,tri_target_list, e_job_list, chunk_num):
        start=time.time()
        e_job_total_score=[0 for _ in range(len(e_job_list))]
        tri = mtriangulations[current_mi]
        f_pts = tri.face_pts
        pts_coor = self.pts
        for e_idx, e in enumerate(e_job_list):
            p1, p3 = e

            e2f = tri.edge_to_face
            key13 = (np.int64(p1) << 32) | np.int64(p3)
            key31 = (np.int64(p3) << 32) | np.int64(p1)

            t1 = e2f.get(key13)
            t2 = e2f.get(key31)

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

            total_score=0
            for target_idx in tri_target_list:
                tri_target = mtriangulations[target_idx]
                ori_cross=0
                new_cross=0

                target_f_pts = tri_target.face_pts
                target_f_nei = tri_target.face_nei

                ftc = self.find_triangle_containing(tri_target, (p1, p3))
                if ftc is None:
                    ori_cross=0
                else:
                    ori_cross = _numba_count_cross(target_f_pts,target_f_nei,pts_coor,p1,p3,ftc)

                ftc = self.find_triangle_containing(tri_target, (p2, p4))
                if ftc is None:
                    new_cross = 0
                else:
                    new_cross = _numba_count_cross(target_f_pts,target_f_nei,pts_coor,p2,p4,ftc)

                n_cross = ori_cross - new_cross
                total_score += n_cross

            e_job_total_score[e_idx] = total_score
        return chunk_num, e_job_total_score, time.time()-start


    def parallel_flip_path(self, tri1, tri2):
        tri = tri1.fast_copy()
        tri_target = tri2.fast_copy()
        pfp = []

        while True:
            cand = []
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
            if not cand:
                break
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

    def random_compute_fpd(self,debug=True, tri_num=-1):
        print(f"Random compute fpd for {self.instance_uid}")
        prev_len = self.dist
        prev_best = prev_len   
        if tri_num==-1:
            tri_num = len(self.triangulations)-1
            
        from multiprocessing import Pool
        # pool = Pool(max(50,len(self.triangulations)))
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
                local_T1 = self.triangulations[tri_num].fast_copy()
                for ind1 in range(T1_ind):
                    for e in seq[ind1]:
                        local_T1.flip(e)
                local_T2 = local_T1.fast_copy()
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
                        # print(tri_num, T1_ind, T2_ind, len(seq))
                    else:
                        T2_ind += 1
                        T1_ind = 0
                        # print(tri_num, T1_ind, T2_ind, len(seq))
            if T2_ind==len(seq):
                tri_num -= 1
        if prev_len < prev_best:
            total_dist = sum([len(pFlip) for pFlip in self.pFlips])
            self.dist = total_dist
            self.WriteData()
            print(f"[{self.instance_uid}] Improved: {prev_best} -> {total_dist}")
        print(f"[{self.instance_uid}] Done")
        return self.center

    def parallel_flip_path2(self, tri1, tri2):
        tri = tri1.fast_copy()
        tri_target = tri2.fast_copy()
        pfp = []
        prev_flip = set()
        step=0
        while True:
            step+=1
            cand = []
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
                else:
                    break
            cand.sort(key=lambda x: x[1], reverse=True)
            if step>100:
                print(cand)
                print(prev_flip)
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
        tri = tri2.fast_copy()
        pfp=[]
        step=0
        prev_flip = set()
        while True:
            step+=1
            cand = []
            for e in tri.edges:
                if e in prev_flip: continue
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri1, e, 0, 1)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=[]
                    continue
                else:
                    break
            cand.sort(key=lambda x: x[1],reverse=True)
            if step>100:
                print("in rev2:", cand)
                print(prev_flip)

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
            for e in tri.edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri_target, e, 1, 2)
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
                e1 = tri.flip(p1, p2)
            pfp.append(flips)
        assert(tri.edges == tri_target.edges)
        return pfp

    def parallel_flip_path3_reverse(self, tri1, tri2):
        tri = tri2.fast_copy()
        pfp = []
        step=0
        while True:
            step+=1
            assert step<1000, f"Too many steps in parallel_flip_path3 for {self.instance_uid}"
            cand = []
            for e in tri.edges:
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri1, e, 1, 2)
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
            pfp1=[]
            for e in flips:
                p1, p2 = e
                e1 = tri.flip(p1, p2)
                pfp1.append((int(e1[0]), int(e1[1])))
            pfp.append(pfp1)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]

    def computePFS_total(self, tri1, tri2):
        T1 = tri1.fast_copy()
        T2 = tri2.fast_copy()

        pFlips_paired1 = self.parallel_flip_path(T1, T2)
        pFlips_paired11 = self.parallel_flip_path_reverse(T1, T2)

        pFlips_paired2 = self.parallel_flip_path2(T1, T2)
        pFlips_paired21 = self.parallel_flip_path2_reverse(T1, T2)

        pFlips_paired3 = self.parallel_flip_path3(T1, T2)
        pFlips_paired31 = self.parallel_flip_path3_reverse(T1, T2)

        path_list = [pFlips_paired1,pFlips_paired2,pFlips_paired3,pFlips_paired11,pFlips_paired21,pFlips_paired31]
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]

        prev_pFlips_i=[]
        for round_ in pFlips_paired:
            round_temp=[]
            for oneFlip in round_:
                round_temp.append(list(oneFlip))
            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i

    def computePFS_2(self, tri1, tri2):
        T1 = tri1.fast_copy()
        T2 = tri2.fast_copy()
        pFlips_paired2 = self.parallel_flip_path2(T1, T2)
        pFlips_paired21 = self.parallel_flip_path2_reverse(T1, T2)
        path_list = [pFlips_paired2,pFlips_paired21]
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]

        prev_pFlips_i=[]
        for round_ in pFlips_paired:
            round_temp=[]
            for oneFlip in round_:
                round_temp.append(list(oneFlip))
            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i, opt_ind

    def old_random_compute_fpd_replace(self):
        prev_len = self.dist
        prev_best = prev_len
        tri_num=0
        print()
        while tri_num < self.num_tris:
            start=time.time()
            seq = self.pFlips[tri_num]
            if len(seq) == 0 or len(seq) == 1:
                tri_num +=1
                continue
            seq_iter =1
            local_T = self.triangulations[tri_num].fast_copy()
            for e in seq[0]:
                local_T.flip(e[0], e[1])
            while seq_iter < len(seq):
                print(f"{seq_iter}/{len(seq)}", end = ' ', flush =True)
                start=time.time()
                seq1 = self.computePFS_total(self.triangulations[tri_num], local_T)
                seq2 = self.computePFS_total(local_T, self.center)
                print(f"{time.time()-start:.2f}s", end=' ', flush=True)
                if (len(seq1) + len(seq2)) <= len(seq):
                    self.pFlips[tri_num] = seq1 + seq2
                for e in seq[seq_iter]:
                    local_T.flip(e[0], e[1])
                seq_iter +=1
            print()
            print(f"T{tri_num}, len(seq)={len(seq)} takes {time.time()-start:.2f}s", end='\n')
            if seq_iter == len(seq):
                tri_num +=1


    def random_compute_fpd_replace(self, cpus):
        TN = [tri_num for tri_num in range(self.num_tris) if len(self.pFlips[tri_num])>1]

        with ProcessPoolExecutor(
                initializer=init_worker,
                initargs=(self.triangulations, self.center, self.pFlips, self.pts,),
                max_workers=cpus
                ) as exe:
            futures=[]

            for tn in TN:
                f = exe.submit(process_, tn)
                futures.append(f)
            for f in as_completed(futures):
                try:
                    tri_num, updated_seq, ttt= f.result()
                    self.pFlips[tri_num] = updated_seq
                except Exception as e:
                    print("error", e)
            print()

    def findCenterGlobal(self, cpus,chunk_size, parallel1=False, parallel2=False):
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
        NCAND=[[] for _ in range(num)]
        mi=0
        print()
        ####### first findCenterGlobal ######
        if parallel1==False:
            start=time.time()
            for i in range(num):
                ncand =[]
                nscore = 0
                tri = mtriangulations[i]
                start=time.time()
                for e in F_E[i]:
                    escore=0
                    for j in range(num):
                        if i==j: continue
                        score, _ = self.flip_score(tri, mtriangulations[j], e, 1, 1)
                        escore +=score
                    if escore >0:
                        ncand.append((e, escore))
                ncand.sort(key = lambda x:x[1], reverse=True)
                NCAND[i]=ncand
            print(f"\tfirst serial takes {time.time()-start:.2f}s")
        else:
            start=time.time()
            TN = [tri_num for tri_num in range(num)]
            with ProcessPoolExecutor(initializer=init_worker_fcg3, initargs=(self.pts,mtriangulations,),max_workers=cpus) as exe:
                    #### parallel ####

                    g_futures=[]
                    for tn in TN:
                        tri = mtriangulations[tn]
                        F_E_array = prepare_edge_array(F_E[tn], tri.edge_to_face, tri.face_pts)
                        g = exe.submit(fcg3, tn, F_E_array)
                        g_futures.append(g)
                    for g in as_completed(g_futures):
                        tri_num, ncand = g.result()
                        NCAND[tri_num] = ncand
            print(f"\tfirst parallel takes {time.time()-start:.2f}s")


        TN = [tri_num for tri_num in range(num)]
        nscores = [0 for tri_num in range(num)]
        flips = [[] for tri_num in range(num)]
        with ProcessPoolExecutor(initializer=init_worker_fcg4, initargs=(self.pts,mtriangulations,NCAND,),max_workers=cpus) as exe:
            g_futures=[]
            for tn in TN:
                g = exe.submit(fcg4, tn)
                g_futures.append(g)
            for g in as_completed(g_futures):
                tri_num, nscore, flp = g.result()
                nscores[tri_num] = nscore
                flips[tri_num] = flp
        mscore = max(nscores)
        mi = nscores.index(mscore)
        if mscore ==0:
            self.pFlips=[]
            for i in range(num):
                if i< num-1:
                    assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
                self.pFlips.append(pfps[i])
            return mtriangulations[0]

        prev_mtri = mtriangulations[mi].fast_copy()
        pfps[mi].append(flips[mi])
        start2=time.time()
        for e in flips[mi]:
            mtriangulations[mi].flip(e[0], e[1])
        print(f"\tflips takes {time.time()-start2:.2f}s")
        fe =[]
        tri = mtriangulations[mi]
        start3=time.time()
        for ee in tri.edges:
            if self.flippable(tri, ee):
                fe.append(ee)
        F_E[mi] = fe
        print(f"\tflippable takes {time.time()-start3:.2f}s")

        ####### second findCenterGlobal ######
        if parallel2==False:
            start=time.time()
            count=1
            while True:
                mscore=0
                flips =[]
                current_mi = mi
                count_start = time.time()
                for i in range(num):
                    ncand=[]
                    nscore = 0
                    tri = mtriangulations[i]
                    if i!= current_mi:
                        prev_ncand = NCAND[i]
                        for e in F_E[i]:
                            escore = 0
                            prev_e_score, _ = self.flip_score(tri, prev_mtri, e, 1,1)
                            current_e_score, _ = self.flip_score(tri, mtriangulations[current_mi], e, 1,1)
                            for cc in prev_ncand:
                                if cc[0] == e:
                                    new_cc = cc[1] - prev_e_score + current_e_score
                                    escore += new_cc
                                    break
                            if escore >0:
                                ncand.append((e, escore))
                    else:
                        ncand=[]
                        for e in F_E[i]:
                            escore=0
                            for j in range(num):
                                if i==j: continue
                                score, _ = self.flip_score(tri, mtriangulations[j], e, 1,1)
                                escore +=score
                            if escore >0:
                                ncand.append((e, escore))
                    ncand.sort(key = lambda x:x[1], reverse=True)
                    NCAND[i] = ncand

                    marked = set()
                    flp =[]
                    for (p1, p2), escore in ncand:
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
                if mscore ==0: break
                print(f"{count}: {time.time()-count_start:.2f}s", end=' ', flush=True)
                count+=1
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
            print(end='\n')
            print(f"\tsecond serial takes {time.time()-start:.2f}s")
        else:
            start=time.time()
            count=1
            with ProcessPoolExecutor(initializer=init_worker_fcg, initargs=(self.pts,),max_workers=cpus) as exe:
                while True:
                    count_start=time.time()
                    mscore=0
                    current_mi = mi
                    #### step 1: fi != current_mi ####
                    futures=[]
                    for fi in range(num):
                        if fi==current_mi:continue
                        f = exe.submit(fcg, fi, NCAND[fi], F_E[fi], mtriangulations[fi], prev_mtri, mtriangulations[mi])
                        futures.append(f)
                    for f in as_completed(futures):
                        try:
                            tri_num, nscore, flp, ncand, t_start = f.result()
                            print(f"T{tri_num}: {t_start:.2f}s", end=' ', flush=True)
                            if  nscore > mscore:
                                mscore = nscore
                                mi = tri_num
                            flips[tri_num] = flp
                            NCAND[tri_num] = ncand #already sorted in fcg()
                        except Exception as e:
                            print("error!", "count ", count,": ",  e)
                    print()
                    print(f"\tstep1 {time.time()-count_start:.2f}s")

                    futures=[]
                    #### step 2: fi == current_mi ####
                    ncand=[]
                    target_list = [j for j in range(num) if j!=current_mi]
                    e_idx_list = []
                    start2_0=time.time()
                    for fe_idx in range(0, len(F_E[current_mi]), chunk_size):
                        e_idx_list.append(F_E[current_mi][fe_idx:fe_idx + chunk_size])
                    print(f"\tchunk {time.time()-start2_0:.2f}s")
                    start2=time.time()
                    for chunk_num, edge_job_list in enumerate(e_idx_list):
                        f = exe.submit(self.my_total_flip_score, current_mi, mtriangulations, target_list, edge_job_list, chunk_num) #tri is mtriangulations[mi]
                        futures.append(f)
                    for f in as_completed(futures):
                        try:
                            chunk_num, e_job_total_score, ch_time = f.result()
                            print(f"C{chunk_num}, {ch_time:.2f}s", end=' ', flush=True)
                            for eidx, escore in enumerate(e_job_total_score):
                                if escore >0:
                                    e = e_idx_list[chunk_num][eidx]
                                    ncand.append((e, escore))
                        except Exception as e:
                            print("error! in step2", "count ", count, ": ", e )
                    print()
                    ncand.sort(key = lambda x:x[1], reverse=True)
                    futures=[]
                    print(f"\tstep2 takes {time.time()-start2:.2f}s, {len(F_E[current_mi])}, current_mi={current_mi}, len(ncand)={len(ncand)}")
                    marked = set()
                    flp =[]
                    nscore = 0
                    start3=time.time()
                    for (p1, p2), escore in ncand:
                        t1 = tri.find_face(p1, p2)
                        t2 = tri.find_face(p2, p1)
                        if t1 in marked or t2 in marked: continue
                        flp.append((p1, p2))
                        marked.add(t1)
                        marked.add(t2)
                        nscore += escore
                    flips[current_mi] = flp
                    NCAND[current_mi] = ncand

                    #### step 3: mi = tri_num or mi = current_mi
                    if nscore > mscore:
                        mscore = nscore
                        mi = current_mi

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
                    fe=[]
                    print(f"count {count} done: {time.time()-count_start:.2f}s")
                    count+=1
            print(f"\tsecond parallel takes {time.time()-start:.2f}s")
        self.pFlips=[]
        for i in range(num):
            if i< num-1:
                if mtriangulations[i].edges != mtriangulations[i+1].edges:
                    print("!!!!!", i, i+1)
                assert(mtriangulations[i].edges == mtriangulations[i+1].edges)
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


    def random_new_center(self, old_fcg, fcg_parallel1, fcg_parallel2, replace_parallel, cpus, chunk_size):
        print(old_fcg, fcg_parallel1, fcg_parallel2, replace_parallel, cpus, chunk_size)
        param = 1
        len_flips = [len(pFlip) for pFlip in self.pFlips]
        max_dist = max(len_flips)
        total_dist = sum(len_flips)
        while param < max_dist *2:
            print("prm ",param, ":")
            revnum = [random.randint(min(param, len(self.pFlips[i]), 1), min(param, len(self.pFlips[i]))) for i in range(len(self.triangulations))]
            newD = FastData()
            newD.pts = self.pts
            for i in range(self.num_tris):
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
                        p1 = int(row[(pi+1)%3])
                        p2 = int(row[(pi+2)%3])
                        newT.flip(p1, p2)
                newD.triangulations.append(newT)

            start=time.time()
            if old_fcg==True:
                print("\tserial old ver findCenterGlobal() takes ... ", end=' ', flush=True)
                self.center = newD.old_findCenterGlobal()
            else:
                if fcg_parallel1==False and fcg_parallel2==False:
                    print("\tserial new ver findCenterGlobal() takes ... ", end=' ', flush=True)
                    self.center = newD.findCenterGlobal(cpus, chunk_size,parallel1=False, parallel2=False)
                elif fcg_parallel1==False and fcg_parallel2==True:
                    print("\tpartial parallel findCenterGlobal() takes ... ", end=' ', flush=True)
                    self.center = newD.findCenterGlobal(cpus, chunk_size, parallel1=False, parallel2=True)
                else:
                    print("\tall parallel findCenterGlobal() takes ... ", end=' ', flush=True)
                    self.center = newD.findCenterGlobal(cpus, chunk_size,parallel1=True, parallel2=True)
            print(f"{time.time()-start:.2f}s")

            for i in range(self.num_tris):
                self.pFlips[i] = self.pFlips[i][:-revnum[i]] + newD.pFlips[i]

            start=time.time()
            if replace_parallel==False:
                print("\tserial random_compute_pfd_replace() takes ... ", end=' ', flush=True)
                self.old_random_compute_fpd_replace() # pFlip update
            else:
                print("\tparallel random_compute_pfd_replace() takes ... ", end=' ', flush=True)
                self.random_compute_fpd_replace(cpus) # pFlip update
            print(f"{time.time()-start:.2f}s", end='\n')

            new_pfp = [len(pFlip) for pFlip in self.pFlips]
            new_dist = sum(new_pfp)

            if total_dist != new_dist:
                self.dist = new_dist
                break
            if self.num_edges<100:
                param +=1
            else:
                param *=2


    def pfd_distribution(self):
        pfd = [len(x) for x in self.pFlips]
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
        dist = sum([len(pFlip) for pFlip in self.pFlips])
        inst["meta"] = {"dist": dist}

        folder = "solutions"
        with open(folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        '''
        #verify
        org_input = 'data/benchmark_instances/'+self.instance_uid+'.json'
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
        '''
        
        opt_folder = "optimal"
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
                if old_score>dist:
                    os.remove(opt_folder+"/"+sol)
                    with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                        json.dump(inst, f, indent='\t')
                    print(f"solution saved {old_score} -> {dist}")

        if not already_exist:
            with open(opt_folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
                json.dump(inst, f, indent='\t')
            print(f"solution saved")



def turn(p1, p2, p3):
    # negative: (p1, p2, p3) CW
    # positive: (p1, p2, p3) CCW
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])



def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def init_worker_fcg(pts):
    global gb_pts_coor
    gb_pts_coor = pts


def fcg(tri_num, prev_ncand, F_E, tri, prev_mtri, current_mtri):
    start=time.time()
    ncand=[]

    f_pts = tri.face_pts
    e2f=  process_typed_dict(tri.edge_to_face)

    prev_f_pts = prev_mtri.face_pts
    prev_f_nei = prev_mtri.face_nei
    prev_e2f=  process_typed_dict(prev_mtri.edge_to_face)
    prev_adj = prev_mtri.adj

    curr_f_pts = current_mtri.face_pts
    curr_f_nei = current_mtri.face_nei
    curr_e2f=  process_typed_dict(current_mtri.edge_to_face)
    curr_adj = current_mtri.adj

    for e in F_E:
        p1, p3 = e
        key13 = (p1 << 32) | p3
        key31 = (p3 << 32) | p1

        t1 = e2f[key13]
        t2 = e2f[key31]

        row1 = f_pts[t1]
        i=0
        if row1[0] == p3: i=0
        elif row1[1] == p3: i=1
        else: i=2
        row2 = f_pts[t2]
        j=0
        if row2[0] == p1: j=0
        elif row2[1] == p1: j=1
        else: j=2
        p4 = int(row1[(i+1)%3])
        p2 = int(row2[(j+1)%3])

        escore = 0
        prev_e_score, _ = _flip_score_fast(e, p2, p4, prev_f_pts, prev_f_nei, prev_e2f, prev_adj, 1, gb_pts_coor, 1)
        current_e_score, _ = _flip_score_fast(e, p2, p4, curr_f_pts, curr_f_nei, curr_e2f, curr_adj, 1, gb_pts_coor, 1)
        for cc in prev_ncand:
            if cc[0] == e:
                new_cc = cc[1] - prev_e_score + current_e_score
                escore += new_cc
                break
        if escore >0:
            ncand.append((e, escore))
    ncand.sort(key = lambda x:x[1], reverse=True)
    marked = set()
    flp =[]
    nscore = 0
    for (p1, p2), escore in ncand:
        t1 = tri.find_face(p1, p2)
        t2 = tri.find_face(p2, p1)
        if t1 in marked or t2 in marked: continue
        flp.append((p1, p2))
        marked.add(t1)
        marked.add(t2)
        nscore += escore
    return tri_num, nscore, flp, ncand, time.time()-start

def prepare_edge_array(F_E_tri_num, e2f, f_pts):
    prepared_edges = []
    for e in F_E_tri_num:
        p1, p3 = e
        key13 = (p1<<32)|p3
        key31 = (p3<<32)|p1
        if key13 not in e2f or key31 not in e2f: continue
        t1 = e2f[key13]
        t2 = e2f[key31]

        row1 = f_pts[t1]
        if row1[0] == p3: i=0
        elif row1[1] == p3: i=1
        else: i=2
        row2 = f_pts[t2]
        if row2[0] == p1: j=0
        elif row2[1] == p1: j=1
        else: j=2
        p4 = int(row1[(i+1)%3])
        p2 = int(row2[(j+1)%3])
        prepared_edges.append((p1, p3, p2, p4))
    return prepared_edges

def fcg2(tri_num,  j_list, prepared_E, mtriangulations):
    start=time.time()
    total_local_scores = np.zeros(len(prepared_E), dtype=np.int64)
    mtris_e2f=[process_typed_dict(mt.edge_to_face) for mt in mtriangulations]

    for j in j_list:
        if tri_num==j: continue
        tg_f_pts = mtriangulations[j].face_pts
        tg_f_nei = mtriangulations[j].face_nei
        tg_adj = mtriangulations[j].adj
        tg_e2f= mtris_e2f[j]
        scores = _T_flip_score_fast(1, gb_pts_coor, tg_f_pts, tg_f_nei, tg_adj, tg_e2f, prepared_E,1)
        total_local_scores += scores
    print(f"T{tri_num}, {time.time()-start:.2f}s", end=' ', flush=True)
    return total_local_scores

def init_worker_fcg3(pts, tris):
    global gb_pts_coor, gb_tris
    gb_pts_coor = pts
    gb_tris = tris

def fcg3(tri_num,  prepared_E):
    global gb_tris

    num = len(gb_tris)
    total_scores = np.zeros(len(prepared_E), dtype=np.int64)
    mtris_e2f=[process_typed_dict(mt.edge_to_face) for mt in gb_tris]

    for j in range(num):
        if tri_num==j: continue
        tg_f_pts = gb_tris[j].face_pts
        tg_f_nei = gb_tris[j].face_nei
        tg_adj = gb_tris[j].adj
        tg_e2f= mtris_e2f[j]
        scores = _T_flip_score_fast(1, gb_pts_coor, tg_f_pts, tg_f_nei, tg_adj, tg_e2f, prepared_E,1)
        total_scores += scores

    sorted_indices = np.argsort(total_scores)[::-1]
    ncand=[]
    for idx in sorted_indices:
        score = total_scores[idx]
        if score >0:
            p1, p3, _, _ = prepared_E[idx]
            ncand.append(((p1, p3), score))
    ncand.sort(key=lambda x:x[1], reverse=True)
    return tri_num, ncand


def init_worker_fcg4(pts, tris, candidate):
    global gb_pts, gb_tris, gb_cand
    gb_pts = pts
    gb_tris = tris
    gb_cand = candidate

def fcg4(tri_num):
    tri = gb_tris[tri_num]
    ncand = gb_cand[tri_num]
    marked = set()
    flp =[]
    nscore=0
    for (p1, p2), escore in ncand:
        t1 = tri.find_face(p1, p2)
        t2 = tri.find_face(p2, p1)
        if t1 in marked or t2 in marked: continue
        flp.append((p1, p2))
        marked.add(t1)
        marked.add(t2)
        nscore += escore
    return tri_num, nscore, flp




@numba.njit
def _T_flip_score_fast(depth, pts_coor, f_pts, f_nei, adj, e2f, prepared_E, ver):
    E_score = np.zeros(len(prepared_E), dtype=np.int64)

    for i in range(len(prepared_E)):
        p1, p3, p2, p4 = prepared_E[i]
        ### find_triangle_containing() ###
        key13 = (p1 << 32) | p3
        key31 = (p3 << 32) | p1

        if key13 in e2f or key31 in e2f:
            t1=-1
        else:
            p = int(adj[p1])
            key_a = (p1 << 32) | p
            key_b = (p << 32) | p1

            if key_a in e2f:
                t = e2f[key_a]
                t1 = _find_t_c_fast(f_pts, f_nei, t, (p1, p3), pts_coor)
            elif key_b in e2f:
                t = e2f[key_b]
                t1 = _find_t_c_fast(f_pts, f_nei, t, (p1, p3), pts_coor)
            else:
                t1 = -1
        #################################

        ori_cross=0
        new_cross=0

        #if t1 is None:
        if t1 == -1:
            ori_cross=0
        else:
            ori_cross = _numba_count_cross_fast(f_pts, f_nei, p1,p3,t1, pts_coor)
        if depth==0:
            E_score[i] = ori_cross
            break

        key24 = (p2 << 32) | p4
        key42 = (p4 << 32) | p2
        if key24 in e2f or key42 in e2f:
            t2=-1
        else:
            p = int(adj[p2])
            key_a = (p2 << 32) | p
            key_b = (p << 32) | p2

            if key_a in e2f:
                t = e2f[key_a]
                t2 = _find_t_c_fast(f_pts, f_nei, t, (p2, p4), pts_coor)
            elif key_b in e2f:
                t = e2f[key_b]
                t2 = _find_t_c_fast(f_pts, f_nei, t, (p2, p4), pts_coor)
            else:
                t2 = -1

        if t2 == -1:
            new_cross=0
        else:
            new_cross = _numba_count_cross_fast(f_pts, f_nei, p2,p4,t2, pts_coor)
        n_cross=0
        if ver==1:
            n_cross = ori_cross - new_cross
        elif ver==2:
            n_cross = (ori_cross - new_cross)/ori_cross if ori_cross>0 else -new_cross

        if depth==1:
            E_score[i] = n_cross
    return E_score


@numba.njit
def _numba_count_cross(f_pts, f_nei, pts_coor, q1, q2, t):
    row = f_pts[t]
    i=0
    if row[0] == q1: i=0
    elif row[1] == q1: i=1
    else: i=2

    idx_next = (i+1)%3
    tt = f_nei[t, idx_next]
    tmp = f_pts[t, idx_next] 
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
    A = (r4[1]-r1[1])
    B = (r4[0]-r1[0])
    while True:
        row = f_pts[t]
        if row[0] == q1: i=0
        elif row[1] == q1: i=1
        else: i=2
        r2 = pts_coor[row[(i+1)%3]]
        r3 = pts_coor[row[(i+2)%3]]

        turn_val1= (r2[0]-r1[0])*A - (r2[1]-r1[1])*B
        turn_val2= (r3[0]-r1[0])*A - (r3[1]-r1[1])*B
        if turn_val1 < 0:
            t = f_nei[t, i]
        elif turn_val2 >0:
            t = f_nei[t, (i+2)%3]
        else:
            return t 


