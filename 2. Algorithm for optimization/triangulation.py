import numpy as np
class FastTriangulation:
    def __init__(self, num_pts, num_faces):
        self.face_pts = np.full((num_faces, 3), -1, dtype=np.int32)
        self.face_nei = np.full((num_faces, 3), -1, dtype=np.int32)
        self.edge_to_face = {} 
        self.adj = np.full(num_pts, -1, dtype=np.int32) 
        self.edges = set()

    def get_ind(self, face_idx, q):
        index = np.where(self.face_pts[face_idx] == q)
        if index[0].size ==0: return -1
        return index[0].item()

    def add_edge_to_face(self, face_idx, p1, p2):
        p1 , p2 = np.int64(p1), np.int64(p2)
        self.edge_to_face[(p1 << 32 | p2)] = face_idx

    def remove_edge_to_face(self, p1, p2):
        p1 , p2 = np.int64(p1), np.int64(p2)
        self.edge_to_face.pop((p1 << 32) | p2, None )
        self.edge_to_face.pop((p2 << 32) | p1, None )

    def find_face(self, p1, p2):
        p1 , p2 = np.int64(p1), np.int64(p2)
        key = (p1 << 32) | p2
        return self.edge_to_face.get(key)

    def fast_copy(self):
        new_triangulation = FastTriangulation(len(self.adj), len(self.face_pts))
        new_triangulation.face_pts[:] = self.face_pts.copy()
        new_triangulation.face_nei[:] = self.face_nei.copy()
        new_triangulation.edge_to_face = self.edge_to_face.copy()
        new_triangulation.adj[:] = list(self.adj)
        new_triangulation.edges = set(self.edges)
        return new_triangulation


    def flip(self, p1, p2):
        # flip (p1,p2) -> new edge (p3, p4)
        f_pts = self.face_pts
        f_nei = self.face_nei
        e2f = self.edge_to_face


        key12 = (p1 << 32) | (p2)
        key21 = (p2 << 32) | (p1)
        t1 = e2f[key12]
        t2 = e2f[key21]
        assert(t1!=None)
        assert(t2!=None)

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


        before = len(self.edge_to_face)
        del e2f[key12]
        del e2f[key21]

        e2f[(p1 << 32) | p4] = t1
        e2f[(p4 << 32) | p3] = t1
        e2f[(p2 << 32) | p3] = t2
        e2f[(p3 << 32) | p4] = t2

        u_old, v_old = (p1, p2) if p1<p2 else (p2, p1)
        u_new, v_new = (p3, p4) if p3<p4 else (p4, p3)
        self.edges.add((int(u_new), int(v_new)))
        self.edges.remove((int(u_old), int(v_old)))


        self.adj[p1] = np.int32(p3)
        self.adj[p2] = np.int32(p3)

        assert(before == len(self.edge_to_face))
        return (u_new, v_new)



