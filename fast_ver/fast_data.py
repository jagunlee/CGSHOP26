import json, os
from fast_Triangulation import *
import numba
import time
import random
import math
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
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
            #print("make triangulation ...:")
            for i, t_data in enumerate(root["triangulations"]):
                self.triangulations[i] = self.make_triangulation(t_data)
                #print(f"T{i}", end=' ', flush=True)
            #print(end='\n')

            self.inst_info()


            # restore center
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.triangulations[min_flip_ind].fast_copy()
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])
            self.triangulations[-1] = self.center.fast_copy()
            new_dist = 0
            new_pfp = [None]*(self.num_tris)
            rev_better =0
            rev_lower_flippable =0
            rev_lower_highest_score=0
            rev_lower_total_flip=0
            for i in range(self.num_tris):
                start = time.time()
                new_pfp1, results1 = self.parallel_flip_path2(i, -1)
                print(f"T{i}: {time.time()-start:.2f}s", end=' ', flush=True)
                start = time.time()
                new_pfp2, results2 = self.parallel_flip_path_rev2(-1, i)
                #print("----------")
                print(f"rev: {time.time()-start:.2f}s", end=' ', flush=True)
                print(len(new_pfp1), len(new_pfp2), end='\n')
                if len(new_pfp1) < len(new_pfp2):
                    new_pfp[i] = new_pfp1
                else:
                    new_pfp[i] = new_pfp2
                    rev_better +=1
                new_dist+=len(new_pfp[i])
                #print("T", i)
                #print("(path) -- (rev)")
                #print(f" pfd: {len(new_pfp1)} -- {len(new_pfp2)}")
                #print(f" (total flippable)/(total for-loop): {results1[0]/results1[1]:.2f} --  {results2[0]/results2[1]:.2f}")
                #print(f" total candidates: {results1[2]} -- {results2[2]}")
                #print(f" total flips: {results1[3]} -- {results2[3]}")
                #print(f" (lowest~highest) score: ({results1[4]}~{results1[5]}) -- ({results2[4]}~{results2[5]})")

                #mean_of_mean_score1 = sum(results1[6])/len(results1[6])
                #mean_of_mean_score2 = sum(results2[6])/len(results2[6])
                #mean_of_std_score1 = sum(results1[7])/len(results1[7])
                #mean_of_std_score2 = sum(results2[7])/len(results2[7])
                #print(f" mean of E, std path(): {mean_of_mean_score1:.2f}, {mean_of_std_score1:.2f}")
                #print("  E(std): ", end=' ', flush=True)
                #for nn in range(len(results1[6])):
                #    print(f"{results1[6][nn]:.2f}({results1[7][nn]:.0f})", end=', ', flush=True)
                #print(end='\n')
                #print(f" mean of E, std rev(): {mean_of_mean_score2:.2f}, {mean_of_std_score2:.2f}")
                #print("  E(std): ", end=' ', flush=True)
                #for nn in range(len(results2[6])):
                #    print(f"{results2[6][nn]:.2f}({results2[7][nn]:.0f})", end=', ', flush=True)
                #print(end='\n')
                #print()

                #rev_lower_mean_score = "True" if mean_of_mean_score2 < mean_of_mean_score1 else "False"

                #if (results1[0]/results1[1]) > (results2[0]/results2[1]):
                #    rev_lower_flippable +=1
                #if results1[-1] > results2[-1]:
                #    rev_lower_highest_score+=1
                #if results1[3] > results2[3]:
                #    rev_lower_total_flip +=1
            #print()
            #print(f"(# of rev_better_dist)/(# tris) = {rev_better}/{self.num_tris}")
            #print(f"(# of rev_lower_flippable)/(# tris) = {rev_lower_flippable}/{self.num_tris}")
            #print(f"(# of rev_lower_total_flip)/(# tris) = {rev_lower_total_flip}/{self.num_tris}")
            #print(f"(# of rev_lower_highest_score)/(# tris) = {rev_lower_highest_score}/{self.num_tris}")
            #print(f"(rev has smaller mean of (mean_score)) = {rev_lower_mean_score}")
            #print()
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
        #prev_flip =set()
        cand_sum =0
        flip_sum = 0
        lowest_score = 1000000
        highest_score = -1000
        total_edges = 0
        total_flippable =0
        mean_score=[]
        std_score=[]
        while True:
            cand = []
            #edges = list(tri.edges)
            edges = list(tri.edges - tri_target.edges)
            #start= time.time()
            prev_flip = set()
            for e in edges:
                if e in prev_flip: continue
                if self.flippable(tri, e):
                    total_flippable +=1
                    score = self.flip_score(tri, tri_target, e, 1)
                    if score[0] >0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=set()
                    continue
                else: break
            cand.sort(key=lambda x: x[1], reverse=True)
            _, score1 = cand[0]
            _, score2 = cand[-1]
            highest_score = max(score1[0], highest_score)
            lowest_score = min(score2[0], lowest_score)
            cur_score = [s[0] for _, s in cand]
            cur_mean = sum(cur_score)/len(cand)
            cur_std = math.sqrt(sum([(cs-cur_mean)**2 for cs in cur_score])/len(cand))
            mean_score.append(cur_mean)
            std_score.append(cur_std)


            flips = []
            marked = set()
            #print(f"{count}: score takes:{time.time()-start:.2f}s", end=' ', flush=True)
            #start= time.time()
            cand_sum +=len(cand)
            for (p1, p2), _ in cand:
                t1 = tri.find_face(p1, p2)
                t2 = tri.find_face(p2, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)
            flip_sum +=len(flips)
            for e in flips:
                p1, p2 = e
                e1 = tri.flip(p1, p2)
                prev_flip.add(e1)
            pfp.append(flips)
            #print(f"flip takes:{time.time()-start:.2f}s", end='\n')
            count+=1
        tri2 = self.triangulations[target_idx]
        assert(tri.edges == tri2.edges)
        for_loop = len(tri.edges)*count
        #print("path2:")
        #print(f"\t (total flippable)/(edges x {count}) = {total_flippable}/{for_loop} = {total_flippable/for_loop:.2f}")
        #print(f"\t total candidates = {cand_sum}")
        #print(f"\t total flips = {flip_sum}")
        #print(f"\t lowest~highest score = {lowest_score}~{highest_score}")
        zip_result = [total_flippable, for_loop, cand_sum, flip_sum, lowest_score, highest_score, mean_score, std_score]
        return pfp, zip_result


    def parallel_flip_path_rev2(self, start_idx, target_idx):
        tri = self.triangulations[start_idx].fast_copy()
        tri_target = self.triangulations[target_idx].fast_copy()
        rev_pfp=[]
        #prev_flip =set()
        count=1
        cand_sum =0
        flip_sum = 0
        lowest_score = 1000000
        highest_score = -1000
        total_flippable =0
        mean_score=[]
        std_score=[]
        while True:
            cand = []
            #edges = list(tri.edges)
            edges = list(tri.edges - tri_target.edges)
            prev_flip2=[]
            prev_flip =set() # 아.. while 밖에 있는게 맞긴한데... 그러면 pfp가 크게 나온다..
            for e in edges:
                if e in prev_flip:
                    continue
                if self.flippable(tri, e):
                    total_flippable +=1
                    score = self.flip_score(tri, tri_target, e, 1)
                    if score[0] > 0:
                        cand.append((e, score))
            if not cand:
                if prev_flip:
                    prev_flip=set()
                    continue
                else: break
            cand.sort(key=lambda x: x[1],reverse=True)
            _, score1 = cand[0]
            _, score2 = cand[-1]
            highest_score = max(score1[0], highest_score)
            lowest_score = min(score2[0], lowest_score)
            cur_score = [s[0] for _, s in cand]
            cur_mean = sum(cur_score)/len(cand)
            cur_std = math.sqrt(sum([(cs-cur_mean)**2 for cs in cur_score])/len(cand))
            mean_score.append(cur_mean)
            std_score.append(cur_std)

            flips = []
            marked = set()
            cand_sum +=len(cand)
            for (p1, p3), _ in cand:
                t1 = tri.find_face(p1, p3)
                t2 = tri.find_face(p3, p1)
                if t1 in marked or t2 in marked: continue
                flips.append((p1, p3))
                marked.add(t1)
                marked.add(t2)
            flip_sum +=len(flips)
            for e in flips:
                p1, p3 = e
                e1 = tri.flip(p1, p3)
                prev_flip.add(e1)
                prev_flip2.append(e1)
            rev_pfp.append(prev_flip2)
            count+=1
        tri2 = self.triangulations[target_idx]
        assert(tri.edges == tri2.edges)
        rev_pfp.reverse()
        for_loop = len(tri.edges)*count
        #print("rev2:")
        #print(f"\t (total flippable)/(edges x {count}) = {total_flippable}/{len(tri.edges)*count} = {total_flippable/(len(tri.edges)*count):.2f}")
        #print(f"\t total candidates = {cand_sum}")
        #print(f"\t total flips = {flip_sum}")
        #print(f"\t lowest~highest score = {lowest_score}~{highest_score}")
        zip_result = [total_flippable, for_loop, cand_sum, flip_sum, lowest_score, highest_score, mean_score, std_score]
        return rev_pfp, zip_result

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
            #pFlips_paired1 = self.parallel_flip_path2(i, -1)
            pFlips_paired2 = self.parallel_flip_path_rev2(-1, i)
            pFlips_paired=[]
            #if len(pFlips_paired1) < len(pFlips_paired2):
            #    pFlips_paired=pFlips_paired1
            #else:
            #    pFlips_paired=pFlips_paired2
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
        inst["meta"] = {"dist": sum([len(self.pFlip) for pFlip in self.pFlips])} # , "input": self.input}

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


