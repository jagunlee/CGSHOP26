from collections import defaultdict, deque
import copy
import cv2
import datetime
from heapq import *
import json
import math
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import pdb
import random
import time

def compute_center_dist(self, T1, max_total=None, diff_mode="grid_score", multi = False):
        """
        Returns (total_length, flips) same as original, but much faster.
        - fast_copy
        - edge_set equality
        - bbox-grid find_difference (exact)
        - early abort if exceeds max_total
        """
        if not T1:
            return float("INF"), None

        target_set = T1.edge_set
        total_length = 0
        flip = []

        for i,_T in enumerate(self.triangulations):
            if max_total is not None and total_length > max_total:
                return max_total + 1, None

            T = _T.fast_copy()
            step = 0
            res_e_list = []
            flip_list = []

            guard = 0
            while True:
                guard += 1
                if guard > 200000:
                    return max_total + 1 if max_total is not None else float("INF"), None

                if T.edge_set == target_set:
                    break

                E1, _ = T.find_difference(T1, mode=diff_mode, compute_l2_scores=False)
                if not E1:
                    break

                step += 1
                if max_total is not None and (total_length + step) > max_total:
                    return max_total + 1, None

                e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
                if not e_list:
                    break

                res_e_list = []
                f_iter = []
                for e in e_list:
                    f_iter.append([e[0], e[1]])
                    res_e_list.append(T.flip(e[0], e[1]))
                flip_list.append(f_iter)

            total_length += step
            flip.append(flip_list)
            print(f"Triangulation {i} to center: {step}")

        return total_length, flip


# ============================================================
# NOTE: Speed-first version
# - Triangulation edges store: dict[(u,v)] -> [nei0, nei1]  (no Diag objects in hot path)
# - edge_set maintained incrementally
# - fast_copy avoids deepcopy
# - find_difference uses bbox-grid pruning (exact strict-intersection semantics preserved)
# - compute_center_len early-aborts and avoids building flip sequences
# - compute_center_dist uses fast_copy + edge_set + faster find_difference
# - find_center_np uses global edge graph incremental updates (fast)
# ============================================================

# ----------------------------
# Basic geometry containers
# ----------------------------
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _canon_edge(u, v):
    return (u, v) if u < v else (v, u)


# ----------------------------
# Fast bbox grid for candidate pruning (exact after strict-intersection check)
# ----------------------------
class _GridIndex:
    __slots__ = ("cs", "cells")

    def __init__(self, cell_size: float):
        self.cs = float(cell_size) if cell_size and cell_size > 0 else 1.0
        self.cells = defaultdict(list)

    def _cell_range(self, mnx, mny, mxx, mxy):
        cs = self.cs
        ix0 = math.floor(mnx / cs)
        ix1 = math.floor(mxx / cs)
        iy0 = math.floor(mny / cs)
        iy1 = math.floor(mxy / cs)
        return ix0, ix1, iy0, iy1

    def insert(self, eidx, mnx, mny, mxx, mxy):
        ix0, ix1, iy0, iy1 = self._cell_range(mnx, mny, mxx, mxy)
        cells = self.cells
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cells[(ix, iy)].append(eidx)

    def query_set(self, mnx, mny, mxx, mxy):
        ix0, ix1, iy0, iy1 = self._cell_range(mnx, mny, mxx, mxy)
        out = set()
        cells = self.cells
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                out.update(cells.get((ix, iy), ()))
        return out

    def iter_candidates(self, mnx, mny, mxx, mxy):
        ix0, ix1, iy0, iy1 = self._cell_range(mnx, mny, mxx, mxy)
        cells = self.cells
        seen = set()
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for idx in cells.get((ix, iy), ()):
                    if idx not in seen:
                        seen.add(idx)
                        yield idx


# ============================================================
# Triangulation (speed-optimized)
# ============================================================
class Triangulation:
    """
    edges: dict[(u,v)] -> [nei0, nei1]
      - edge endpoints are always canonical u < v
      - nei array stores the "other neighbor" on each directed side. We keep the same meaning
        as your original Diag.nei_pts[i<nei] indexing.
    """

    def __init__(self, pts, T, px=None, py=None):
        self.pts = pts
        self.px = px
        self.py = py

        self.edges = {}
        self.edge_set = set()

        self.make_triangulation(T)
        self.times = {}

    def fast_copy(self):
        """Much faster than deepcopy: copy dict + 2-int lists, plus edge_set."""
        new = object.__new__(Triangulation)
        new.pts = self.pts
        new.px = self.px
        new.py = self.py
        new.times = {}

        # copy edge dict + 2-int lists
        new.edges = {e: [nei[0], nei[1]] for e, nei in self.edges.items()}
        new.edge_set = set(self.edge_set)
        return new

    def return_edge(self):
        return [list(e) for e in self.edges.keys()]

    def make_triangulation(self, T):
        edges = {}
        for e in T:
            a, b = _canon_edge(e[0], e[1])
            edges[(a, b)] = [None, None]  # [nei0, nei1]

        nei_dict = {i: [] for i in range(len(self.pts))}
        for e in T:
            nei_dict[e[0]].append(e[1])
            nei_dict[e[1]].append(e[0])

        for i in range(len(self.pts)):
            nei_dict[i], _, found = self.sort_cw_with_half_circle(nei_dict[i], i)
            m = len(nei_dict[i])
            if m == 0:
                continue

            for j in range(m - 1):
                v = nei_dict[i][j]
                nxt = nei_dict[i][j + 1]
                a, b = _canon_edge(i, v)
                # original: if i < v: edges[(i,v)].nei_pts[1] = nxt else edges[(v,i)].nei_pts[0] = nxt
                edges[(a, b)][1 if i < v else 0] = nxt

            if not found:
                lastv = nei_dict[i][-1]
                firstv = nei_dict[i][0]
                a, b = _canon_edge(i, lastv)
                edges[(a, b)][1 if i < lastv else 0] = firstv

        self.edges = edges
        self.edge_set = set(edges.keys())

    def sort_cw_with_half_circle(self, pts, center):
        cx, cy = self.pts[center].x, self.pts[center].y
        with_angles = [(pt, math.atan2(self.pts[pt].y - cy, self.pts[pt].x - cx)) for pt in pts]
        with_angles.sort(key=lambda pa: -pa[1])

        n = len(with_angles)
        if n == 0:
            return [], [], False

        angles = [a for _, a in with_angles]
        best_start = 0
        found = False
        doubled = angles + [a - 2 * math.pi for a in angles]
        for i in range(n):
            j = i + n - 1
            if doubled[i] - doubled[j] <= math.pi + 1e-12:
                best_start = i
                found = True
                break

        with_angles2 = with_angles + with_angles
        if found:
            reordered = with_angles2[best_start:best_start + n]
            reordered_angles = [a if a <= math.pi else a - 2 * math.pi for _, a in reordered]
        else:
            reordered = with_angles2[:n]
            reordered_angles = angles

        reordered_pts = [pt for pt, _ in reordered]
        return reordered_pts, reordered_angles, found

    # ----------------------------
    # Strict intersection (same semantics as your original)
    # ----------------------------
    def intersect(self, d11, d12, d21, d22):
        px = self.px
        py = self.py
        if px is None or py is None:
            # fallback (shouldn't happen if Data passes px/py)
            def _orient(a, b, c) -> float:
                return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

            EPS = 1e-9
            p1 = self.pts[d11]
            p2 = self.pts[d12]
            p3 = self.pts[d21]
            p4 = self.pts[d22]
            o1 = _orient(p1, p2, p3)
            o2 = _orient(p1, p2, p4)
            o3 = _orient(p3, p4, p1)
            o4 = _orient(p3, p4, p2)
            return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)

        ax = px[d11]; ay = py[d11]
        bx = px[d12]; by = py[d12]
        cx = px[d21]; cy = py[d21]
        dx = px[d22]; dy = py[d22]

        o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)

        EPS = 1e-9
        return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)

    def is_convex_quad(self, i, j):
        i, j = _canon_edge(i, j)
        nei = self.edges.get((i, j), None)
        if nei is None:
            return False
        d1, d2 = nei[0], nei[1]
        if d1 is None or d2 is None:
            return False
        return self.intersect(i, j, d1, d2)

    # ----------------------------
    # Next-stage optimization:
    # - local cache for is_convex_quad results per call
    # - avoid repeated dict lookups in blocking
    # ----------------------------
    def maximal_disjoint_convex_quad(self, E, prev_use=None):
        if not E:
            return []

        edges = self.edges
        prev = set(prev_use) if prev_use else set()
        blocked = set()
        res = []
        second_best = None

        # cache for convex check to avoid repeats during two-pass scans
        convex_cache = {}

        def is_cq(e):
            # e is canonical
            v = convex_cache.get(e)
            if v is not None:
                return v
            i, j = e
            nei = edges.get(e)
            if nei is None:
                convex_cache[e] = False
                return False
            d1, d2 = nei[0], nei[1]
            if d1 is None or d2 is None:
                convex_cache[e] = False
                return False
            val = self.intersect(i, j, d1, d2)
            convex_cache[e] = val
            return val

        def block_star(e):
            # block e and its 4 star edges around endpoints w.r.t quad neighbors
            nei = edges.get(e)
            if nei is None:
                return
            d1, d2 = nei[0], nei[1]
            if d1 is None or d2 is None:
                return
            a, b = e
            blocked.add(e)
            blocked.add(_canon_edge(a, d1))
            blocked.add(_canon_edge(a, d2))
            blocked.add(_canon_edge(b, d1))
            blocked.add(_canon_edge(b, d2))

        # first pick (same semantics)
        for ee in E:
            e = _canon_edge(ee[0], ee[1])
            if e in blocked:
                continue
            if is_cq(e):
                if e not in prev:
                    res.append(e)
                    block_star(e)
                    break
                else:
                    second_best = e

        # if none selected, pick second_best then greedy add
        if not res:
            if second_best is None:
                return []
            res.append(second_best)
            block_star(second_best)
            for ee in E:
                e = _canon_edge(ee[0], ee[1])
                if e in blocked:
                    continue
                if is_cq(e):
                    res.append(e)
                    block_star(e)
            return res

        # pass 1: not in prev_use
        for ee in E:
            e = _canon_edge(ee[0], ee[1])
            if e in blocked:
                continue
            if (e not in prev) and is_cq(e):
                res.append(e)
                block_star(e)

        # pass 2: allow prev_use
        for ee in E:
            e = _canon_edge(ee[0], ee[1])
            if e in blocked:
                continue
            if is_cq(e):
                res.append(e)
                block_star(e)

        return res

    # ----------------------------
    # flip (same semantics) + maintain edge_set
    # ----------------------------
    def flip(self, i, j):
        i, j = _canon_edge(i, j)
        if not self.is_convex_quad(i, j):
            return -1, -1

        edges = self.edges
        nei = edges[(i, j)]
        d1, d2 = nei[0], nei[1]

        # update the 4 surrounding edges' neighbor pointers (same logic as original)
        edges[_canon_edge(i, d1)][1 if i < d1 else 0] = d2
        edges[_canon_edge(d1, j)][1 if d1 < j else 0] = d2
        edges[_canon_edge(j, d2)][1 if j < d2 else 0] = d1
        edges[_canon_edge(d2, i)][1 if d2 < i else 0] = d1

        # add new diag (d1,d2) with neighbors [j,i] or [i,j]
        a, b = _canon_edge(d1, d2)
        edges[(a, b)] = [None, None]
        if d1 < d2:
            edges[(a, b)][0] = j
            edges[(a, b)][1] = i
        else:
            edges[(a, b)][0] = i
            edges[(a, b)][1] = j

        # delete old diag
        del edges[(i, j)]

        # maintain edge_set
        self.edge_set.remove((i, j))
        self.edge_set.add((a, b))

        return a, b

    # ----------------------------
    # Exact but faster find_difference:
    # - uses maintained edge_set
    # - uses bbox-grid pruning (still exact after strict check)
    # - supports mode="grid_score" (default) or mode="none" (fastest, changes ordering!)
    # ----------------------------
    def _edge_bbox(self, e):
        u, v = e
        px = self.px
        py = self.py
        ax = px[u]; ay = py[u]
        bx = px[v]; by = py[v]
        mnx = ax if ax < bx else bx
        mny = ay if ay < by else by
        mxx = bx if ax < bx else ax
        mxy = by if ay < by else ay
        return mnx, mny, mxx, mxy

    def _count_intersections_A_vs_B(self, A, B, cell_size):
        if not A or not B:
            return [0] * len(A)

        grid = _GridIndex(cell_size)

        Bu = [0] * len(B)
        Bv = [0] * len(B)
        Bb = [None] * len(B)

        for j, e in enumerate(B):
            u, v = e
            Bu[j] = u
            Bv[j] = v
            bb = self._edge_bbox(e)
            Bb[j] = bb
            grid.insert(j, bb[0], bb[1], bb[2], bb[3])

        counts = [0] * len(A)
        for i, e in enumerate(A):
            u, v = e
            bb = self._edge_bbox(e)
            c = 0
            for j in grid.iter_candidates(bb[0], bb[1], bb[2], bb[3]):
                u2 = Bu[j]; v2 = Bv[j]
                if u == u2 or u == v2 or v == u2 or v == v2:
                    continue
                if self.intersect(u, v, u2, v2):
                    c += 1
            counts[i] = c
        return counts

    def find_difference(self, T, mode="grid_score", compute_l2_scores=False):
        e1 = self.edge_set
        e2 = T.edge_set
        l1 = list(e1 - e2)
        l2 = list(e2 - e1)
        if not l1 and not l2:
            return ([], [])

        if mode == "none":
            # fastest; ordering differs from original
            return (l1, l2)

        # heuristic cell size from instance scale + target diff size
        px = self.px; py = self.py
        spanx = float(px.max() - px.min()) if px is not None else 1.0
        spany = float(py.max() - py.min()) if py is not None else 1.0
        area = max(spanx * spany, 1.0)
        denom = max(len(l2), 1)
        cell_size = max(math.sqrt(area / denom) * 4.0, 1.0)

        c1 = self._count_intersections_A_vs_B(l1, l2, cell_size)
        l1_scored = list(zip(c1, l1))
        l1_scored.sort(reverse=True)
        l1_sorted = [e for _, e in l1_scored]

        if compute_l2_scores:
            c2 = self._count_intersections_A_vs_B(l2, l1, cell_size)
            l2_scored = list(zip(c2, l2))
            l2_scored.sort(reverse=True)
            l2_sorted = [e for _, e in l2_scored]
        else:
            l2_sorted = l2

        return (l1_sorted, l2_sorted)

    def check_1pfd(self, T):
        E1, _ = self.find_difference(T)
        E1_in = self.maximal_disjoint_convex_quad(E1)
        return len(E1_in) == len(E1)


# ============================================================
# Data class
# ============================================================
class Data:
    def __init__(self, input):
        try:
            base = Path(__file__).resolve()
        except NameError:
            base = Path(os.getcwd()).resolve()

        self.application_path = str(base.parents[0])
        self.input = input
        self.df = None

        # ★ triangulation weight (coreset일 때 사용, 아니면 모두 1)
        self.tri_weights = None

        self.ReadData()

    def ReadData(self):
        print("--------------------ReadData--------------------")

        if "solution" not in self.input:
            # =========================
            #  (a) 인스턴스 JSON 직접 읽는 경우
            # =========================
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                print(f"instance: {self.instance_uid}")

                pts_x = root["points_x"]
                pts_y = root["points_y"]

                self.pts = [Point(pts_x[i], pts_y[i]) for i in range(len(pts_y))]
                self.px = np.array(pts_x, dtype=np.float64)
                self.py = np.array(pts_y, dtype=np.float64)

                self.triangulations = []
                Ts = root["triangulations"]
                for T in Ts:
                    self.triangulations.append(Triangulation(self.pts, T, self.px, self.py))

                print(f"num of pts: {len(self.pts)}")
                print(f"num of triangulations: {len(self.triangulations)}")

                # ★ NEW: coreset_weights 읽기 (없으면 모두 1)
                w = root.get("coreset_weights", None)
                if w is None:
                    # 일반 인스턴스 or weight 정보 없음 → weight=1
                    self.tri_weights = np.ones(len(self.triangulations), dtype=np.float64)
                else:
                    w_arr = np.asarray(w, dtype=np.float64)
                    if w_arr.shape[0] != len(self.triangulations):
                        print(
                            f"[WARN] coreset_weights length mismatch "
                            f"({w_arr.shape[0]} vs {len(self.triangulations)}); ignore weights."
                        )
                        self.tri_weights = np.ones(len(self.triangulations), dtype=np.float64)
                    else:
                        self.tri_weights = w_arr

            self.center = self.triangulations[0]
            self.dist = float("INF")
            self.flip = [[] for _ in range(len(self.triangulations))]

        else:
            # =========================
            #  (b) solution.json을 읽는 경우
            #      → 원본 인스턴스 JSON(root["meta"]["input"]) 다시 열기
            # =========================
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                print(f"instance: {self.instance_uid}")
                self.flip = root["flips"]
                self.dist = sum([len(x) for x in self.flip])
                org_input = root["meta"]["input"]

            self.input = org_input
            with open(self.input, "r", encoding="utf-8") as f:
                root = json.load(f)
                self.instance_uid = root["instance_uid"]
                pts_x = root["points_x"]
                pts_y = root["points_y"]
                self.pts = [Point(pts_x[i], pts_y[i]) for i in range(len(pts_y))]
                self.px = np.array(pts_x, dtype=np.float64)
                self.py = np.array(pts_y, dtype=np.float64)

                self.triangulations = []
                Ts = root["triangulations"]
                for T in Ts:
                    self.triangulations.append(Triangulation(self.pts, T, self.px, self.py))

                print(f"num of pts: {len(self.pts)}")
                print(f"num of triangulations: {len(self.triangulations)}")

                # ★ 여기서도 동일하게 weight 처리
                w = root.get("coreset_weights", None)
                if w is None:
                    self.tri_weights = np.ones(len(self.triangulations), dtype=np.float64)
                else:
                    w_arr = np.asarray(w, dtype=np.float64)
                    if w_arr.shape[0] != len(self.triangulations):
                        print(
                            f"[WARN] coreset_weights length mismatch "
                            f"({w_arr.shape[0]} vs {len(self.triangulations)}); ignore weights."
                        )
                        self.tri_weights = np.ones(len(self.triangulations), dtype=np.float64)
                    else:
                        self.tri_weights = w_arr

    # strict intersection same as Triangulation (kept for find_center_np global graph)
    def intersect(self, d11, d12, d21, d22):
        px = self.px; py = self.py
        ax = px[d11]; ay = py[d11]
        bx = px[d12]; by = py[d12]
        cx = px[d21]; cy = py[d21]
        dx = px[d22]; dy = py[d22]

        o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)

        EPS = 1e-9
        return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)
    def compute_center_dist(self, T1: Triangulation, max_total=None, diff_mode="grid_score", multi = False):
        """
        Returns (total_length, flips) same as original, but much faster.
        - fast_copy
        - edge_set equality
        - bbox-grid find_difference (exact)
        - early abort if exceeds max_total
        """
        if not T1:
            return float("INF"), None

        target_set = T1.edge_set
        total_length = 0
        flip = []

        for i,_T in enumerate(self.triangulations):
            if max_total is not None and total_length > max_total:
                return max_total + 1, None

            T = _T.fast_copy()
            step = 0
            res_e_list = []
            flip_list = []

            guard = 0
            while True:
                guard += 1
                if guard > 200000:
                    return max_total + 1 if max_total is not None else float("INF"), None

                if T.edge_set == target_set:
                    break

                E1, _ = T.find_difference(T1, mode=diff_mode, compute_l2_scores=False)
                if not E1:
                    break

                step += 1
                if max_total is not None and (total_length + step) > max_total:
                    return max_total + 1, None

                e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
                if not e_list:
                    break

                res_e_list = []
                f_iter = []
                for e in e_list:
                    f_iter.append([e[0], e[1]])
                    res_e_list.append(T.flip(e[0], e[1]))
                flip_list.append(f_iter)

            total_length += step
            flip.append(flip_list)
            print(f"Triangulation {i} to center: {step}")

        return total_length, flip
    # ========================================================
    # Next-stage compute_center_len / compute_center_dist
    # ========================================================
    def compute_center_len(self, T1: Triangulation, max_total=None, diff_mode="grid_score"):
        """
        Only returns total distance (no flip sequences).
        - fast_copy instead of deepcopy
        - early abort if sum exceeds max_total
        - uses edge_set equality
        """
        if not T1:
            return float("INF")

        target_set = T1.edge_set
        total = 0

        for _T in self.triangulations:
            if max_total is not None and total > max_total:
                return max_total + 1

            T = _T.fast_copy()
            step = 0
            res_e_list = []

            # safety loop bound (prevents accidental infinite loops)
            # (should not trigger if logic is correct)
            guard = 0
            while True:
                guard += 1
                if guard > 200000:
                    return max_total + 1 if max_total is not None else float("INF")

                if T.edge_set == target_set:
                    break

                E1, _ = T.find_difference(T1, mode=diff_mode, compute_l2_scores=False)
                if not E1:
                    break

                step += 1
                if max_total is not None and (total + step) > max_total:
                    return max_total + 1

                e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
                if not e_list:
                    break

                res_e_list = []
                for e in e_list:
                    res_e_list.append(T.flip(e[0], e[1]))

            total += step

        return total

    def compute_center_dist(self, T1: Triangulation, max_total=None, diff_mode="grid_score", multi = False):
        """
        Returns (total_length, flips) same as original, but much faster.
        - fast_copy
        - edge_set equality
        - bbox-grid find_difference (exact)
        - early abort if exceeds max_total
        """
        if not T1:
            return float("INF"), None

        target_set = T1.edge_set
        total_length = 0
        flip = []

        for i,_T in enumerate(self.triangulations):
            if max_total is not None and total_length > max_total:
                return max_total + 1, None

            T = _T.fast_copy()
            step = 0
            res_e_list = []
            flip_list = []

            guard = 0
            while True:
                guard += 1
                if guard > 200000:
                    return max_total + 1 if max_total is not None else float("INF"), None

                if T.edge_set == target_set:
                    break

                E1, _ = T.find_difference(T1, mode=diff_mode, compute_l2_scores=False)
                if not E1:
                    break

                step += 1
                if max_total is not None and (total_length + step) > max_total:
                    return max_total + 1, None

                e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
                if not e_list:
                    break

                res_e_list = []
                f_iter = []
                for e in e_list:
                    f_iter.append([e[0], e[1]])
                    res_e_list.append(T.flip(e[0], e[1]))
                flip_list.append(f_iter)

            total_length += step
            flip.append(flip_list)
            print(f"Triangulation {i} to center: {step}")

        return total_length, flip
    
    

    # ========================================================
    # Speed-first find_center_np (global edge usage/inter graph incremental)
    # ========================================================
        # ========================================================
    # Speed-first find_center_np (weighted coreset 대응)
    # ========================================================
    def find_center_np(self, debug=False, TOPK=3000, max_steps=300000):
        """
        NOTE:
          - self.tri_weights 를 사용하여 각 triangulation에 weight를 부여할 수 있음
            (예: tri_coreset에서 만든 weighted coreset).
          - tri_weights 가 정의되지 않았거나 길이가 안 맞으면 모두 weight=1로 동작.
        """
        step = 0
        T = [t.fast_copy() for t in self.triangulations]
        nT = len(T)
        res_e_lists = [[] for _ in range(nT)]

        # ★ triangulation weight (coreset에서 S_weights로 세팅)
        tri_w = getattr(self, "tri_weights", None)
        if tri_w is None or len(tri_w) != nT:
            tri_w = np.ones(nT, dtype=np.int64)
        else:
            tri_w = np.asarray(tri_w, dtype=np.int64)

        px = self.px
        py = self.py

        # ---- global edge indexing across triangulations ----
        edge_to_idx = {}
        idx_to_edge = []
        tri_sets = []

        for t in T:
            s = set()
            for e in t.edge_set:
                idx = edge_to_idx.get(e)
                if idx is None:
                    idx = len(idx_to_edge)
                    edge_to_idx[e] = idx
                    idx_to_edge.append(e)
                s.add(idx)
            tri_sets.append(s)

        M = len(idx_to_edge)
        cap = max(M, 4096)

        usage = np.zeros(cap, dtype=np.int64)   # ★ int64로 조금 여유 있게
        inter = np.zeros(cap, dtype=np.int64)
        weight = np.zeros(cap, dtype=np.float64)

        minx = np.empty(cap, dtype=np.float64)
        miny = np.empty(cap, dtype=np.float64)
        maxx = np.empty(cap, dtype=np.float64)
        maxy = np.empty(cap, dtype=np.float64)

        # ★ edge usage: "몇 개의 triangulation이 이 edge를 가지는가" → 가중치 합
        for ti, s in enumerate(tri_sets):
            w_t = tri_w[ti]
            if w_t == 0:
                continue
            for eidx in s:
                usage[eidx] += w_t

        # grid cell size heuristic
        spanx = float(px.max() - px.min())
        spany = float(py.max() - py.min())
        area = max(spanx * spany, 1.0)
        cell_size = max(math.sqrt(area / max(M, 1)) * 4.0, 1.0)
        grid = _GridIndex(cell_size)

        # bboxes & insert
        for idx, (u, v) in enumerate(idx_to_edge):
            ax, ay = px[u], py[u]
            bx, by = px[v], py[v]
            mnx = ax if ax < bx else bx
            mny = ay if ay < by else by
            mxx = bx if ax < bx else ax
            mxy = by if ay < by else ay
            minx[idx], miny[idx], maxx[idx], maxy[idx] = mnx, mny, mxx, mxy
            grid.insert(idx, mnx, mny, mxx, mxy)

        neighbors = [None] * cap
        for i in range(M):
            neighbors[i] = []

        # initial neighbor graph + inter
        for i in range(M):
            u1, v1 = idx_to_edge[i]
            cand = grid.query_set(minx[i], miny[i], maxx[i], maxy[i])
            for j in cand:
                if j <= i:
                    continue
                u2, v2 = idx_to_edge[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                if self.intersect(u1, v1, u2, v2):
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    inter[i] += usage[j]
                    inter[j] += usage[i]

        # Zobrist signature for fast done-check
        rng = np.random.default_rng(0)
        edge_hash = rng.integers(1, np.iinfo(np.uint64).max, size=cap, dtype=np.uint64)
        tri_hash = np.zeros(nT, dtype=np.uint64)
        tri_size = np.zeros(nT, dtype=np.int32)

        for ti, s in enumerate(tri_sets):
            h = np.uint64(0)
            for eidx in s:
                h ^= edge_hash[eidx]
            tri_hash[ti] = h
            tri_size[ti] = len(s)

        def all_same():
            return np.all(tri_hash == tri_hash[0]) and np.all(tri_size == tri_size[0])

        # edge -> triangulations containing it (incremental T_val update)
        edge_to_tris = [set() for _ in range(cap)]
        for ti, s in enumerate(tri_sets):
            for eidx in s:
                edge_to_tris[eidx].add(ti)

        def ensure_cap(newM):
            nonlocal cap, usage, inter, weight, minx, miny, maxx, maxy, neighbors, edge_hash, edge_to_tris
            if newM <= cap:
                return
            newcap = max(newM, int(cap * 1.6) + 4096)

            usage2 = np.zeros(newcap, dtype=np.int64); usage2[:cap] = usage
            inter2 = np.zeros(newcap, dtype=np.int64); inter2[:cap] = inter
            weight2 = np.zeros(newcap, dtype=np.float64); weight2[:cap] = weight
            minx2 = np.empty(newcap, dtype=np.float64); minx2[:cap] = minx
            miny2 = np.empty(newcap, dtype=np.float64); miny2[:cap] = miny
            maxx2 = np.empty(newcap, dtype=np.float64); maxx2[:cap] = maxx
            maxy2 = np.empty(newcap, dtype=np.float64); maxy2[:cap] = maxy

            edge_hash2 = np.zeros(newcap, dtype=np.uint64); edge_hash2[:cap] = edge_hash

            neighbors.extend([None] * (newcap - cap))
            edge_to_tris.extend([set() for _ in range(newcap - cap)])

            cap = newcap
            usage, inter, weight = usage2, inter2, weight2
            minx, miny, maxx, maxy = minx2, miny2, maxx2, maxy2
            edge_hash = edge_hash2

        def add_edge(u, v):
            nonlocal M
            key = _canon_edge(u, v)
            idx = edge_to_idx.get(key)
            if idx is not None:
                return idx

            idx = M
            M += 1
            ensure_cap(M)

            edge_to_idx[key] = idx
            idx_to_edge.append(key)

            usage[idx] = 0
            inter[idx] = 0
            weight[idx] = 0.0
            neighbors[idx] = []
            edge_hash[idx] = rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64)
            edge_to_tris[idx].clear()

            ax, ay = px[u], py[u]
            bx, by = px[v], py[v]
            mnx = ax if ax < bx else bx
            mny = ay if ay < by else by
            mxx = bx if ax < bx else ax
            mxy = by if ay < by else ay
            minx[idx], miny[idx], maxx[idx], maxy[idx] = mnx, mny, mxx, mxy
            grid.insert(idx, mnx, mny, mxx, mxy)

            cand = grid.query_set(mnx, mny, mxx, mxy)
            for j in cand:
                if j == idx:
                    continue
                u2, v2 = idx_to_edge[j]
                if u == u2 or u == v2 or v == u2 or v == v2:
                    continue
                if self.intersect(u, v, u2, v2):
                    neighbors[idx].append(j)
                    neighbors[j].append(idx)
                    inter[idx] += usage[j]
            return idx

        active = usage[:M] > 0
        mask = active
        usage_active = usage[:M][mask]
        inter_active = inter[:M][mask]
        weight[:M][mask] = (inter_active - usage_active) / usage_active

        # ★ T_val: triangulation weight * (edge weight 합)
        T_val = np.zeros(nT, dtype=np.float64)
        for ti, s in enumerate(tri_sets):
            if not s:
                continue
            arr = np.fromiter(s, dtype=np.int32)
            T_val[ti] = tri_w[ti] * weight[arr].sum()

        t_upt_list = []

        while True:
            if debug and step % 10 == 0:
                print(f"[{self.instance_uid}] step={step} bestT={float(T_val.max()):.6f} M={M}")

            if all_same():
                dist, flip = self.compute_center_dist(T[0])
                print(f"Total distance from center: {self.dist} -> {dist}")
                self.center = T[0]
                self.dist = dist
                self.flip = flip
                return T[0], dist

            if step >= max_steps:
                return (-1, -1)

            step += 1

            # try triangulations in descending objective order until we get non-empty flip list
            order = np.argsort(-T_val)
            update_t_ind = None
            flip_list = None

            for ti in order:
                s = tri_sets[ti]
                if not s:
                    continue
                arr = np.fromiter(s, dtype=np.int32)
                w = weight[arr]
                pos_mask = (w >= 0)
                if not np.any(pos_mask):
                    continue

                pos_edges = arr[pos_mask]
                if TOPK is not None and pos_edges.size > TOPK:
                    wpos = weight[pos_edges]
                    sel = np.argpartition(-wpos, TOPK - 1)[:TOPK]
                    pos_edges = pos_edges[sel]

                pos_edges = pos_edges[np.argsort(-weight[pos_edges])]
                update_e = [idx_to_edge[int(eidx)] for eidx in pos_edges]
                fl = T[ti].maximal_disjoint_convex_quad(update_e, res_e_lists[ti])
                if fl:
                    update_t_ind = ti
                    flip_list = [_canon_edge(e[0], e[1]) for e in fl]
                    break

            if update_t_ind is None:
                return (-1, -1)

            t = T[update_t_ind]

            local_res_list = []
            for e in flip_list:
                newe = t.flip(e[0], e[1])
                if newe == (-1, -1):
                    continue
                local_res_list.append(_canon_edge(newe[0], newe[1]))

            res_e_lists[update_t_ind] = local_res_list

            removed_idx = [edge_to_idx[e] for e in flip_list]
            added_idx = [add_edge(e[0], e[1]) for e in local_res_list]

            # loop detection
            if step < 100:
                t_upt_list.append(update_t_ind)
            else:
                t_upt_list.pop(0)
                t_upt_list.append(update_t_ind)
                if all(tt == t_upt_list[0] for tt in t_upt_list):
                    return (-1, -1)

            changed = set(removed_idx)
            changed.update(added_idx)
            affected = set(changed)
            for eidx in list(changed):
                for nb in neighbors[eidx]:
                    affected.add(nb)
            affected = np.fromiter(affected, dtype=np.int32)

            old_w = weight[affected].copy()

            # ★ membership이 바뀐 triangulation의 weight 반영
            w_t = tri_w[update_t_ind]
            T_val[update_t_ind] += w_t * (weight[added_idx].sum() - weight[removed_idx].sum())

            # update tri_set / signature / memberships
            s = tri_sets[update_t_ind]
            h = tri_hash[update_t_ind]

            for eidx in removed_idx:
                if eidx in s:
                    s.remove(eidx)
                    edge_to_tris[eidx].discard(update_t_ind)
                    h ^= edge_hash[eidx]

            for eidx in added_idx:
                if eidx not in s:
                    s.add(eidx)
                    edge_to_tris[eidx].add(update_t_ind)
                    h ^= edge_hash[eidx]

            tri_hash[update_t_ind] = h
            tri_size[update_t_ind] = len(s)

            # ★ usage / inter 업데이트에도 weight 반영
            delta = defaultdict(int)
            for eidx in removed_idx:
                delta[eidx] -= 1
            for eidx in added_idx:
                delta[eidx] += 1

            for eidx, d in delta.items():
                if d == 0:
                    continue
                usage[eidx] += d * w_t
                for nb in neighbors[eidx]:
                    inter[nb] += d * w_t

            # recompute weights only on affected and adjust T_val for tris containing these edges
            ua = usage[affected]
            ia = inter[affected]
            new_w = np.zeros_like(old_w, dtype=np.float64)
            mask2 = (ua > 0)
            new_w[mask2] = (ia[mask2] - ua[mask2]) / ua[mask2]
            dw = new_w - old_w
            weight[affected] = new_w

            for k in range(affected.size):
                dwi = dw[k]
                if dwi == 0.0:
                    continue
                eidx = int(affected[k])
                for tj in edge_to_tris[eidx]:
                    T_val[tj] += dwi * tri_w[tj]

            print(f"[{self.instance_uid}, {step} step] Triangulation {update_t_ind} flipped, {len(local_res_list)} edges")

    # ========================================================
    # Faster random_move:
    # - use compute_center_len first with early-abort
    # - only compute full flip list if it actually improves
    # ========================================================
    def random_move(self, diff_mode="grid_score"):
        prev_len, _ = self.compute_center_dist(self.center, diff_mode=diff_mode)
        total_best = prev_len
        T = self.center.fast_copy()

        print(f"Start with {prev_len}")

        step = 0
        total_step = 0
        end_step = 10 * len(self.triangulations) * len(self.pts)

        edges = list(T.edges.keys())
        starting_edge_ind = 0
        random.shuffle(edges)

        while total_step < end_step:
            total_step += 1
            do_random = random.random() > 0.999 ** step

            if do_random or starting_edge_ind == len(edges):
                random.shuffle(edges)
                _e_list = T.maximal_disjoint_convex_quad(edges)
                if _e_list:
                    random_choice = [random.random() for _ in range(len(_e_list))]
                    e_list = [e for i, e in enumerate(_e_list) if random_choice[i] > 0.5]
                    for e in e_list:
                        T.flip(e[0], e[1])

                edges = list(T.edges.keys())
                random.shuffle(edges)
                starting_edge_ind = 0

                # quick length-only evaluation
                new_len = self.compute_center_len(T, max_total=prev_len, diff_mode=diff_mode)
                total_best = min(total_best, new_len)
                prev_len = new_len
                step = 0
                continue

            # single flip try
            T1 = T.fast_copy()
            e = edges[starting_edge_ind]
            if not T1.is_convex_quad(e[0], e[1]):
                starting_edge_ind += 1
                continue

            T1.flip(e[0], e[1])

            new_len = self.compute_center_len(T1, max_total=prev_len, diff_mode=diff_mode)
            if new_len <= prev_len:
                step = 0
                T = T1
                edges = list(T.edges.keys())
                random.shuffle(edges)

                if new_len < prev_len:
                    # only now compute full flip list
                    new_len2, flip = self.compute_center_dist(T, max_total=new_len, diff_mode=diff_mode)
                    if new_len2 < prev_len:
                        self.center = T.fast_copy()
                        self.dist = new_len2
                        self.flip = flip
                        if new_len2 < total_best:
                            print(f"[{self.instance_uid} {total_step}/{end_step}] {total_best}->{new_len2}")
                            total_best = new_len2
                            self.WriteData()
                    prev_len = min(prev_len, new_len2)

                starting_edge_ind = 0
            else:
                step += 1
                starting_edge_ind += 1

        return self.center

    # ========================================================
    # compute_pfd (uses fast_copy)
    # ========================================================
    def compute_pfd(self, i, j, diff_mode="grid_score"):
        T = self.triangulations[i].fast_copy()
        T1 = self.triangulations[j].fast_copy()

        step = 0
        res_e_list = []
        flip_list = []

        guard = 0
        while True:
            guard += 1
            if guard > 200000:
                break

            E1, _ = T.find_difference(T1, mode=diff_mode, compute_l2_scores=False)
            if not E1:
                break
            step += 1

            e_list = T.maximal_disjoint_convex_quad(E1, res_e_list)
            if not e_list:
                break

            res_e_list = []
            flip_iter = []
            for e in e_list:
                flip_iter.append([e[0], e[1]])
                res_e_list.append(T.flip(e[0], e[1]))
            flip_list.append(flip_iter)

        # print(f"{i} -> {j} can be done in {step} step!")
        return step, flip_list, i, j

    # ========================================================
    # Write + Draw (unchanged; adapted to edge dict)
    # ========================================================
    def WriteData(self):
        inst = dict()
        inst["content_type"] = "CGSHOP2026_Solution"
        inst["instance_uid"] = self.instance_uid
        inst["flips"] = self.flip
        inst["meta"] = {"dist": self.dist, "input": self.input, "center": self.center.return_edge()}

        folder = "solutions"
        os.makedirs(folder, exist_ok=True)
        with open(folder + "/" + self.instance_uid + ".solution" + ".json", "w", encoding="utf-8") as f:
            json.dump(inst, f, indent='\t')

        opt_folder = "opt"
        os.makedirs(opt_folder, exist_ok=True)
        opt_list = os.listdir(opt_folder)
        already_exist = False
        for sol in opt_list:
            if self.instance_uid + ".solution.json" in sol:
                already_exist = True
                with open(opt_folder + "/" + sol, "r", encoding="utf-8") as ff:
                    root = json.load(ff)
                    try:
                        old_score = root["meta"]["dist"]
                    except:
                        old_flips = root["flips"]
                        old_score = sum([len(x) for x in old_flips])
                if old_score > self.dist:
                    os.remove(opt_folder + "/" + sol)
                    with open(opt_folder + "/" + self.instance_uid + ".solution" + ".json", "w", encoding="utf-8") as f:
                        json.dump(inst, f, indent='\t')
        if not already_exist:
            with open(opt_folder + "/" + self.instance_uid + ".solution" + ".json", "w", encoding="utf-8") as f:
                json.dump(inst, f, indent='\t')

        fname = "result.csv"
        if not os.path.exists(fname):
            df_dict = dict()
            df_dict["date"] = datetime.date.today()
            df_dict[self.instance_uid] = [self.dist]
            df = DataFrame(df_dict)
            df.to_csv("result.csv")
        else:
            df = pd.read_csv(fname, index_col=0)
            if self.instance_uid not in df.columns:
                df[self.instance_uid] = float("INF")

            today = datetime.date.today().isoformat()
            if df["date"].iloc[-1] != today:
                df.loc[len(df)] = list(df.iloc[-1])
                df.loc[len(df.index) - 1, "date"] = today

            df.loc[len(df.index) - 1, self.instance_uid] = min(df.loc[len(df.index) - 1, self.instance_uid], self.dist)
            df.to_csv("result.csv")

    def DrawTriangulation(self, T, colored_edges=(), name="", folder=""):
        if name:
            name = "_" + name

        minx = min(p.x for p in self.pts)
        miny = min(p.y for p in self.pts)
        maxx = max(p.x for p in self.pts)
        maxy = max(p.y for p in self.pts)

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        lineType = cv2.LINE_AA

        width = int(maxx - minx)
        height = int(maxy - miny)
        rad = 1000 / max(width, 1)

        add_size = 80
        width = int(width * rad) + add_size
        height = int(height * rad) + add_size
        minw = add_size // 2 - int(minx * rad)
        minh = height + int(miny * rad) - add_size // 2

        img = np.zeros((height, width, 3), dtype="uint8") + 255
        colored_edges = set(_canon_edge(e[0], e[1]) for e in colored_edges)

        for e in T.edges.keys():
            col = (0, 0, 255) if e in colored_edges else (0, 0, 0)
            cv2.line(
                img,
                (minw + int(rad * self.pts[e[0]].x), minh - int(rad * self.pts[e[0]].y)),
                (minw + int(rad * self.pts[e[1]].x), minh - int(rad * self.pts[e[1]].y)),
                col,
                2,
            )

        for i, p in enumerate(self.pts):
            cv2.circle(img, (minw + int(rad * p.x), minh - int(rad * p.y)), 5, (255, 0, 0), -1)
            cv2.putText(
                img,
                str(i),
                (minw + int(rad * p.x) + 10, minh - int(rad * p.y) + 10),
                fontFace,
                fontScale,
                (0, 0, 0),
                thickness,
                lineType,
            )

        loc = os.path.join(self.application_path, folder if folder else "solutions")
        os.makedirs(loc, exist_ok=True)
        cv2.imwrite(loc + "/" + self.instance_uid + ".triangulation" + name + ".png", img)


# ----------------------------
# helpers (unchanged)
# ----------------------------
def turn(p1: Point, p2: Point, p3: Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)


def cw_key(center):
    cx, cy = center.x, center.y

    def key(pt):
        angle = math.atan2(pt.y - cy, pt.x - cx)
        return -angle

    return key
