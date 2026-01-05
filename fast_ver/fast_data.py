import json
from fast_Triangulation import *
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
            print(f"num_edges = {self.num_edges}, num_faces = {self.num_faces}")

            num_tris = len(root["triangulations"])
            self.triangulations = [None] * (num_tris+1)
            for i, t_data in enumerate(root["triangulations"]):
                self.triangulations[i] = self.make_triangulation(t_data)
                print(f"T{i}", end='\n')
            print(end='\n')

            # restore center
            min_flip_ind = np.argmin([len(x) for x in self.pFlips])
            self.center = self.triangulations[min_flip_ind].fast_copy()
            for flip_seq in self.pFlips[min_flip_ind]:
                for flp in flip_seq:
                    self.center.flip(flp[0], flp[1])
            self.triangulations[-1] = self.center.fast_copy()
            new_dist = 0
            new_pfp = [None]*len(self.triangulations)
            for i in range(len(self.triangulations)):
                new_pfp1 = self.parallel_flip_path2(i, -1)
                new_dist +=len(new_pfp1)
            print(f"Dist sum = {new_dist}")


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

    def find_triangle_containing(self, tri_idx, con:tuple):
        q1, q2 = con
        tri = self.triangulations[tri_idx]
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


    def count_cross(self, tri_idx, con:tuple):
        t = self.find_triangle_containing(tri_idx, con)
        if t is None:
            return 0
        q1, q2 = con
        tri = self.triangulations[tri_idx]

        f_pts = tri.face_pts
        f_nei = tri.face_nei
        pts_coor = self.pts
        p_q1 = pts_coor[q1]
        p_q2 = pts_coor[q2]

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

    def flip_score(self, tri:FastTriangulation, target_idx, e:tuple, depth:int):
        p1, p3 = e

        e2f = tri.edge_to_face
        f_pts = tri.face_pts

        key13 = (np.int64(p1) << 32) | np.int64(p3)
        key31 = (np.int64(p3) << 32) | np.int64(p1)

        t1 = e2f.get(key13)
        t2 = e2f.get(key31)

        #if t1 is None or t2 is None: return (-999, depth)

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
        ori_cross = self.count_cross(target_idx, (p1, p3))
        new_cross = self.count_cross(target_idx, (p2, p4))
        n_cross = ori_cross - new_cross
        m_score = (n_cross, depth)
        if depth == 1:
            return m_score
        tri.flip(p1, p3)
        for pe in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            if self.flippable(start_idx, pe):
                nsc = self.flip_score(tri, target_idx, pe, 1)
                m_score = max(m_score, (nsc[0] + m_score[0], nsc[1]))
        tri.flip(p2, p4)
        return m_score



    def parallel_flip_path2(self, start_idx, target_idx):
        tri = self.triangulations[start_idx].fast_copy()
        pfp = []
        while True:
            prev_flip =set()
            cand = []
            edges = list(tri.edges)
            for e in edges:
                if self.flippable(tri, e):
                    if e in prev_flip: continue
                    score = self.flip_score(tri, target_idx, e, 1)
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
                prev_flip.add(e1)
            pfp.append(flips)
        tri2 = self.triangulations[target_idx]
        assert(tri.edges == tri2.edges)
        return pfp




def turn(p1, p2, p3):
    # negative: (p1, p2, p3) CW
    # positive: (p1, p2, p3) CCW
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])


