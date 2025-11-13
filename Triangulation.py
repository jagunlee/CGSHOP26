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

    # need to fix, due to time issue
    def getEdges(self):
        edges = []

        for t in self.triangles:
            edges.append(sorted([t.pts[0], t.pts[1]]))
            edges.append(sorted([t.pts[1], t.pts[2]]))
            edges.append(sorted([t.pts[2], t.pts[0]]))

        unique_edges = []
        for e in edges:
            if e not in unique_edges:  # 이미 추가된 엣지가 아니면
                unique_edges.append(e)

        return unique_edges

        '''
        edges = list(set(edges))
        return edges
        '''

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