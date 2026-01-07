class Triangle:
    # p, q, r: node index
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
        self.dict = dict()
        self.adj = []

    def __del__(self):
        for t in self.triangles:
            del t
    def fast_copy(self):
        new = Triangulation()

        # 1) Triangle 복사 + 매핑
        tri_map = {}  # old Triangle -> new Triangle
        for t in self.triangles:
            nt = Triangle(t.pts[0], t.pts[1], t.pts[2])
            tri_map[t] = nt
            new.triangles.add(nt)

        # 2) neis 포인터 복원 (old -> new 로 연결)
        for t in self.triangles:
            nt = tri_map[t]
            for i in range(3):
                nei_t = t.neis[i]
                nt.neis[i] = tri_map[nei_t] if nei_t is not None else None

        # 3) edges 그대로 복사
        new.edges = set(self.edges)

        # 4) dict 재구성 (방향 있는 edge -> Triangle)
        new.dict = {}
        for nt in new.triangles:
            for i in range(3):
                a = nt.pt(i)
                b = nt.pt(i + 1)
                new.dict[(a, b)] = nt

        # 5) adj 리스트 얕은 복사
        new.adj = list(self.adj)

        return new

    def find_triangle(self, q1: int, q2: int):
        if (q1, q2) in self.dict:
            return self.dict[(q1,q2)]
        else:
            return None
            
    def flip(self, e):
        p1, p2 = e
        t1 = self.find_triangle(p1, p2)
        t2 = self.find_triangle(p2, p1)
        assert(t1)
        assert(t2)
        i = t1.get_ind(p2)
        p3 = t1.pt(i + 1)
        j = t2.get_ind(p1)
        p4 = t2.pt(j + 1)
        tt1 = t2.neis[j]
        tt2 = t1.neis[i]

        t1.pts[i] = p4
        t1.neis[i] = t2
        t1.neis[(i + 2) % 3] = tt1
        if tt1:
            ii = tt1.get_ind(p4)
            tt1.neis[ii] = t1
        t2.pts[j] = p3
        t2.neis[j] = t1
        t2.neis[(j + 2) % 3] = tt2
        if tt2:
            jj = tt2.get_ind(p3)
            tt2.neis[jj] = t2
        before = len(self.dict)
        del self.dict[(p1, p2)]
        del self.dict[(p2, p1)]
        self.adj[p1] = p3
        self.adj[p2] = p3
        for i in range(3):
            self.dict[(t1.pt(i), t1.pt(i+1))] = t1
            self.dict[(t2.pt(i), t2.pt(i+1))] = t2
        assert(before == len(self.dict))
        assert(self.find_triangle(p3, p1))
        assert(self.find_triangle(p2, p3))
        self.edges.add((min(p3, p4), max(p3, p4)))
        self.edges.remove((min(e), max(e)))
        return (min(p3, p4), max(p3, p4))

    def return_edge(self):
        return [list(e) for e in self.edges]
