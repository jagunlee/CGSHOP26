import json, os
#from multiprocessing import Process, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from fast_Triangulation import *
import numba
from numba.typed import Dict, List
from numba import types as typ
import time
import random
import math
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
import copy

def init_worker(tri_data, center_data, pfp, pts):
    global gb_tris, gb_center, gb_pFlips, gb_pts_coor
    gb_tris = tri_data
    gb_center = center_data
    gb_pFlips = pfp
    gb_pts_coor = pts


def process_(tri_num):
    tri_obj = gb_tris[tri_num]
    local_T = gb_tris[tri_num].fast_copy()
    center = gb_center.fast_copy()

    seq = gb_pFlips[tri_num]
    for e in seq[0]:
        local_T.flip(e[0], e[1])
    updated_pFlips=seq
    for seq_iter in range(1, len(seq)):
        seq1 = _computePFS_total(tri_obj, local_T)
        seq2 = _computePFS_total(local_T, center)
        if (len(seq1) + len(seq2)) <= len(updated_pFlips):
            updated_pFlips = seq1 + seq2
        for e in seq[seq_iter]:
            local_T.flip(e[0], e[1])
    return tri_num, updated_pFlips

def _computePFS_total(T1, T2):
    pts = gb_pts_coor

    t1 = T1.fast_copy()
    t2 = T2.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=False
    pFlips_paired1 = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)


    t1 = T1.fast_copy()
    t2 = T2.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=False
    pFlips_paired2 = _parallel_flip_path2(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 0, reverse)


    t1 = T1.fast_copy()
    t2 = T2.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=False
    pFlips_paired3 = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)

    #reverse
    t1 = T2.fast_copy()
    t2 = T1.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=True
    tmp = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)
    pFlips_paired11 = tmp[::-1]

    #reverse
    t1 = T2.fast_copy()
    t2 = T1.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=True
    tmp = _parallel_flip_path2(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 0, reverse)
    pFlips_paired21 = tmp[::-1]

    #reverse
    t1 = T2.fast_copy()
    t2 = T1.fast_copy()
    e2f=  process_typed_dict(t1.edge_to_face)
    tg_e2f = process_typed_dict(t2.edge_to_face)
    reverse=True
    tmp = _parallel_flip_path(t1.face_pts, t1.face_nei, e2f, t1.adj,
                                         t2.face_pts, t2.face_nei, tg_e2f, t2.adj, pts, 1, reverse)
    pFlips_paired31 = tmp[::-1]


    path_list = [pFlips_paired1, pFlips_paired2, pFlips_paired3,pFlips_paired11,pFlips_paired21, pFlips_paired31]

    opt_ind = np.argmin([len(x) for x in path_list])
    pFlips_paired = path_list[opt_ind]
    prev_pFlips_i=[]
    for round_ in pFlips_paired:
        round_temp=[]
        for oneFlip in round_:
            round_temp.append(list(oneFlip))
        prev_pFlips_i.append(round_temp)
    return prev_pFlips_i

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
    if row1[0] == p2: i=0
    elif row1[1] == p2: i=1
    else: i=2
    p3 = int(row1[(i+1)%3])

    row2 = f_pts[t2]
    if row2[0] == p1: j=0
    elif row2[1] == p1: j=1
    else: j=2
    p4 = int(row2[(j+1)%3])

    # t1 측 이웃들
    n_p2p3 = f_nei[t1, i] # == tt2
    n_p3p1 = f_nei[t1, (i+1)%3]

    # t2측 이웃들
    m_p1p4 = f_nei[t2, j] # == tt1
    m_p4p2 = f_nei[t2, (j+1)%3]


    # 삼각형 정점 정보 업데이트 (새로운 간선 p3-p4 기준 CCW 정렬)
    f_pts[t1,i] = np.int32(p4)
    f_pts[t2,j] = np.int32(p3)

    # 삼각형 이웃 정보 업데이트
    f_nei[t1, i] = np.int32(t2)
    f_nei[t1, (i+2)%3] = m_p1p4
    f_nei[t2, j] = np.int32(t1)
    f_nei[t2, (j+2)%3] = n_p2p3


    if m_p1p4 != -1:
        row = f_pts[m_p1p4]
        if row[0]==p4: ii=0
        elif row[1]==p4: ii=1
        else: ii=2
        f_nei[m_p1p4, ii] = np.int32(t1)

    if n_p2p3 != -1:
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
    #edges.add(e1)
    #if (min(p1, p2), max(p1, p2)) in edges:
    #    edges.discard((min(p1, p2), max(p1, p2)))
    adj[p1] = np.int32(p3)
    adj[p2] = np.int32(p3)

    #assert(before == len(e2f))
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

#def _flip_score_fast(e, p2, p4, tri_target, depth):
@numba.njit
def _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor):
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

    #if t1 is None:
    if t1 == -1:
        ori_cross=0
    else:
        ori_cross = _numba_count_cross_fast(f_pts, f_nei, p1,p3,t1, pts_coor)
    if depth==0: return (ori_cross, 0)


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

    #if t2 is None:
    if t2 == -1:
        new_cross=0
    else:
        new_cross = _numba_count_cross_fast(f_pts, f_nei, p2,p4,t2, pts_coor)
    n_cross = ori_cross - new_cross
    m_score = (n_cross, depth)
    if depth==1:
        return m_score


#@numba.njit
def _error_parallel_flip_path(f_pts, f_nei, e2f, adj, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, pts_coor, depth, reverse):
    pfp = List()
    pfp.append(List([(1,2)]))
    pfp.pop(0)
    num_pts = len(pts_coor)
    all_keys = np.array(list(e2f.keys()), dtype=np.int64)
    all_keys.sort()
    visited = np.zeros(len(all_keys), dtype=np.bool_)
    while True:
        cand = List()
        cand.append((1.0,(1,2)))
        cand.pop(0)
        for E_key in e2f.keys():
            q1, q3 = int(E_key>>32), int(E_key & 0xFFFFFFFF)
            e = (min(q1, q3), max(q1, q3))
            key13 = (q1<<32)|q3
            key31 = (q3<<32)|q1
            if key13 not in e2f or key31 not in e2f: continue

            idx = np.searchsorted(all_keys, key13)
            if idx < len(all_keys) and all_keys[idx]==key13:
                if visited[idx]: continue
            idx = np.searchsorted(all_keys, key31)
            if idx < len(all_keys) and all_keys[idx]==key31:
                if visited[idx]: continue
                else: visited[idx]=True


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
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor)
                if score >0:
                    cand.append((score, e))
                    idx = np.searchsorted(all_keys, key13)
                    if idx < len(all_keys) and all_keys[idx]==key13:
                        visited[idx] = True
                    idx = np.searchsorted(all_keys, key31)
                    if idx < len(all_keys) and all_keys[idx]==key13:
                        visited[idx] = True
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
        all_keys = np.array(list(e2f.keys()), dtype=np.int64)
        all_keys.sort()
        visited = np.zeros(len(all_keys), dtype=np.bool_)
        if reverse:
            pfp.append(flips_reverse)
        else:
            pfp.append(flips)
    return pfp


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
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor)
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

    while True:
        cand = List()
        cand.append((1.0,(1,2)))
        cand.pop(0)
        for E_key, _ in e2f.items():
            q1, q3 = E_key>>32, E_key & 0xFFFFFFFF
            e = (q1, q3)
            if e in prev_flip: continue
            q1, q3 = e
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
                score, _ = _flip_score_fast(e, p2, p4, tg_f_pts, tg_f_nei, tg_e2f, tg_adj, depth, pts_coor)
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


def pool_computePFS_total(self, tri1, tri2):
    T1 = tri1.fast_copy()
    T2 = tri2.fast_copy()
    pool = Pool(processes=2)
    t1t2=(T1,T2)
    pFlips_paired1 = pool.apply_async(self.parallel_flip_path, t1t2)
    pFlips_paired11 = pool.apply_async(self.parallel_flip_path_reverse, t1t2)


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


            org_input = '/Users/hyeyun/Experiment/PFD/hyeyun_git/data/benchmark_instances/'+self.instance_uid+'.json'
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
            #self.center = self.make_triangulation(root["triangulations"][min_flip_ind])
            #for flip_seq in self.pFlips[min_flip_ind]:
            #    for flp in flip_seq:
            #        self.center.flip(flp[0], flp[1])
            print("center just from one of the T!!!!!")
            max_flip_ind = np.argmax([len(x) for x in self.pFlips])
            self.center = self.triangulations[min_flip_ind].fast_copy()
            for i in range(len(self.triangulations)):
                self.pFlips[i] = self.parallel_flip_path(self.triangulations[i], self.center)
            self.dist = sum([len(x) for x in self.pFlips])
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

        if depth == 0: return (ori_cross, 0)

        ftc = self.find_triangle_containing(tri_target, (p2, p4))
        if ftc is None:
            new_cross = 0
        else:
            new_cross = _numba_count_cross(f_pts,f_nei,pts_coor,p2,p4,ftc)

        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        if depth == 1:
            return m_score

        tri.flip(p1, p3)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(tri, pe):
                nsc = self.flip_score(tri, tri_target, pe, depth - 1)
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
                    score = self.flip_score(tri, tri_target, e, 1)
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
                    score = self.flip_score(tri, tri1, e, 1)
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
                    score = self.flip_score(tri, tri_target, e, 0)
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
            #edges = list(tri.edges)
            for e in tri.edges:
                if e in prev_flip: continue
                if self.flippable(tri, e):
                    score = self.flip_score(tri, tri1, e, 0)
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
                    score = self.flip_score(tri, tri_target, e, 1)
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
            #edges = list(tri.edges)
            for e in tri.edges:
                if self.flippable(tri, e):
                    #score = self.flip_score(tri, tri_target, e, 1)
                    score = self.flip_score(tri, tri1, e, 1)
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
        #assert(tri.edges == tri_target.edges)
        assert(tri.edges == tri1.edges)
        return pfp[::-1]

    def pool_computePFS_total(self, tri1, tri2):
        T1 = tri1.fast_copy()
        T2 = tri2.fast_copy()
        pool = Pool(processes=2)
        t1t2=(T1,T2)
        pFlips_paired1 = pool.apply_async(self.parallel_flip_path, t1t2)
        pFlips_paired11 = pool.apply_async(self.parallel_flip_path_reverse, t1t2)

        pFlips_paired2 = pool.apply_async(self.parallel_flip_path2, t1t2)
        pFlips_paired21 = pool.apply_async(self.parallel_flip_path2_reverse, t1t2)

        pFlips_paired3 = pool.apply_async(self.parallel_flip_path3, t1t2)
        pFlips_paired31 = pool.apply_async(self.parallel_flip_path3_reverse, t1t2)

        path_list = [pFlips_paired1.get(),pFlips_paired2.get(),pFlips_paired3.get(),
                     pFlips_paired11.get(),pFlips_paired21.get(),pFlips_paired31.get()]
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]

        prev_pFlips_i=[]
        for round_ in pFlips_paired:
            round_temp=[]
            for oneFlip in round_:
                round_temp.append(list(oneFlip))
            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i


    def computePFS_total(self, tri1, tri2):
        T1 = tri1.fast_copy()
        T2 = tri2.fast_copy()

        start=time.time()

        pFlips_paired1 = self.parallel_flip_path(T1, T2)
        #pFlips_paired11 = self.parallel_flip_path_reverse(T1, T2)

        pFlips_paired2 = self.parallel_flip_path2(T1, T2)
        #pFlips_paired21 = self.parallel_flip_path2_reverse(T1, T2)

        #pFlips_paired3 = self.parallel_flip_path3(T1, T2)
        #pFlips_paired31 = self.parallel_flip_path3_reverse(T1, T2)

        #path_list = [pFlips_paired1,pFlips_paired2,pFlips_paired3,pFlips_paired11,pFlips_paired21,pFlips_paired31]
        path_list = [pFlips_paired1,pFlips_paired2]
        opt_ind = np.argmin([len(x) for x in path_list])
        pFlips_paired = path_list[opt_ind]

        prev_pFlips_i=[]
        for round_ in pFlips_paired:
            round_temp=[]
            for oneFlip in round_:
                round_temp.append(list(oneFlip))
            prev_pFlips_i.append(round_temp)
        return prev_pFlips_i

    def old_random_compute_fpd_replace(self):
        prev_len = self.dist
        prev_best = prev_len
        tri_num=0

        while tri_num < self.num_tris:
            print("tri_num = ", tri_num)
            seq = self.pFlips[tri_num]
            if len(seq) == 0 or len(seq) == 1:
                tri_num +=1
                continue
            seq_iter =1
            local_T = self.triangulations[tri_num].fast_copy()
            for e in seq[0]:
                local_T.flip(e[0], e[1])
            while seq_iter < len(seq):
                print(seq_iter, end = ' ', flush =True)
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
            if seq_iter == len(seq):
                tri_num +=1


    def random_compute_fpd_replace(self):
        TN = [tri_num for tri_num in range(self.num_tris) if len(self.pFlips[tri_num])>1]

        #for tn in TN:
        #    process_(tn, self.pts, self.triangulations, self.center, self.pFlips)
        with ProcessPoolExecutor(
                initializer=init_worker,
                initargs=(self.triangulations, self.center, self.pFlips, self.pts),
                #max_workers=1
                ) as exe:
            futures=[]

            for tn in TN:
                f = exe.submit(process_, tn)
                futures.append(f)
            #print("completed job..:", end=' ', flush=True)
            for f in as_completed(futures):
                try:
                    tri_num, updated_seq = f.result()
                    #print("T", tri_num, end=' ', flush=True)
                    self.pFlips[tri_num] = updated_seq
                except Exception as e:
                    print(e)
            #print()


    def for_random_compute_fpd_replace(self):
        prev_len = self.dist
        prev_best = prev_len

        TN = [tri_num for tri_num in range(self.num_tris) if len(self.pFlips[tri_num])>1]
        for tri_num in TN:
            print("tri_num = ", tri_num)
            seq = self.pFlips[tri_num]
            seq_iter =1
            local_T = self.triangulations[tri_num].fast_copy()
            for e in seq[0]:
                local_T.flip(e[0], e[1])
            for seq_iter in range(1, len(seq)):
                start=time.time()
                seq1 = self.computePFS_total(self.triangulations[tri_num], local_T)
                seq2 = self.computePFS_total(local_T, self.center)
                print(f"{seq_iter}: {time.time()-start:.2f}s", end=' ', flush=True)
                if (len(seq1) + len(seq2)) <= len(seq):
                    self.pFlips[tri_num] = seq1 + seq2
                for e in seq[seq_iter]:
                    local_T.flip(e[0], e[1])
            print()

    def slower_random_compute_fpd_replace(self):
        prev_len = self.dist
        prev_best = prev_len
        tri_num=0

        with Pool(processes=2) as pool:
            while tri_num < self.num_tris:
                print("tri_num = ", tri_num)
                seq = self.pFlips[tri_num]
                if len(seq) == 0 or len(seq) == 1:
                    tri_num +=1
                    continue
                seq_iter =1
                local_T = self.triangulations[tri_num].fast_copy()
                for e in seq[0]:
                    local_T.flip(e[0], e[1])
                while seq_iter < len(seq):
                    print(seq_iter, end = ' ', flush =True)
                    start=time.time()
                    #seq1 = self.computePFS_total(self.triangulations[tri_num], local_T)
                    #seq2 = self.computePFS_total(local_T, self.center)
                    inputs = [(self.triangulations[tri_num], local_T), (local_T, self.center)]
                    two_seqs = pool.starmap(self.computePFS_total, inputs)
                    seq1 = inputs[0]
                    seq2 = inputs[1]
                    print(f"{time.time()-start:.2f}s", end=' ', flush=True)
                    if (len(seq1) + len(seq2)) <= len(seq):
                        self.pFlips[tri_num] = seq1 + seq2
                    for e in seq[seq_iter]:
                        local_T.flip(e[0], e[1])
                    seq_iter +=1
                print()
                if seq_iter == len(seq):
                    tri_num +=1

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
                    score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
                    escore +=score
                if escore >0:
                    ncand.append((e, escore))
            ncand.sort(key = lambda x:x[1], reverse=True)
            #print("T", i," ncand = ",  ncand)
            NCAND.append(ncand)
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
        #print("count, mscore = ", 0, mscore)
        #print()

        #count=1
        while True:
            mscore=0
            flips =[]
            current_mi = mi
            for i in range(num):
                ncand=[]
                nscore = 0
                tri = mtriangulations[i]
                if i!= current_mi:
                    prev_ncand = NCAND[i]
                    for e in F_E[i]:
                        escore = 0
                        prev_e_score, _ = self.flip_score(tri, prev_mtri, e, 1)
                        current_e_score, _ = self.flip_score(tri, mtriangulations[current_mi], e, 1)
                        for cc in prev_ncand:
                            if cc[0] == e:
                                new_cc = cc[1] - prev_e_score + current_e_score
                                escore += new_cc
                        if escore >0:
                            ncand.append((e, escore))
                else:#i==current_mi
                    ncand=[]
                    for e in F_E[i]:
                        escore=0
                        for j in range(num):
                            if i==j: continue
                            score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
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
            #count+=1
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
                        score, _ = self.flip_score(tri, mtriangulations[j], e, 1)
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
        print("param: ",end = ' ', flush=True)
        while param < max_dist *2:
            print(param, end=' ', flush=True)
            revnum = [min(param, len(pFlip)) for pFlip in self.pFlips]
            stop=True
            newD = FastData()
            newD.pts = self.pts
            for i in range(self.num_tris):
                #print("newT", i)
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
            #start=time.time()
            #print("findCenterGlobal() takes ... ", end=' ', flush=True)
            self.center = newD.findCenterGlobal()
            #print(f"{time.time()-start:.2f}s")
            for i in range(self.num_tris):
                self.pFlips[i] = self.pFlips[i][:-revnum[i]] + newD.pFlips[i]
            #start=time.time()
            #print("random_compute_pfd_replace()... ")
            self.random_compute_fpd_replace() # pFlip update
            #print(f"it takes {time.time()-start:.2f}s", end='\n')
            new_pfp = [len(pFlip) for pFlip in self.pFlips]
            new_dist = sum(new_pfp)
            if total_dist != new_dist:
                print()
                self.dist = new_dist
                print("-----dist update, newly count:", total_dist, new_dist)
                print()
                break
            param +=1


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

        path = '/Users/hyeyun/Experiment/PFD/hyeyun_git/'
        folder = "hy_solutions"
        with open(path+folder+"/"+self.instance_uid+".solution"+".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        #verify
        org_input = '/Users/hyeyun/Experiment/PFD/hyeyun_git/data/benchmark_instances/'+self.instance_uid+'.json'
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
            return t #face_idx


