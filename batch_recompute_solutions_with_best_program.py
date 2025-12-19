#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch recompute triangulation-center solutions using evolved best_program.py.

Directory conventions (as you described):
  - Instances:  data/benchmark_instances/*.json
  - Solutions:  opt/*.solution.json
  - Output:     OpenEvolve_opt/  (created if missing)

Matching:
  - Solution file name: A.solution.json  -> instance is typically A.json
  - Also, solution JSON contains instance_uid; instance path is instances_dir/{instance_uid}.json

For each solution:
  1) Validate & apply old flips to recover the implied center triangulation.
  2) Use best_program.plan_parallel_flip_sequence(...) to compute NEW flips to that center.
  3) Compare total distance = sum_i len(sequence_i).
  4) If improved, write new solution JSON into out_dir with the same file name.

Usage:
  python batch_recompute_solutions_with_best_program.py --best openevolve_output/best/best_program.py

Optional:
  --instances_dir data/benchmark_instances
  --solutions_dir opt
  --out_dir OpenEvolve_opt
  --tries 30
  --max_steps 0   (0 => auto from symdiff lower bound)
  --seed 0
  --limit 0       (0 => no limit; else process only first N solutions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional, Iterable
import argparse
import importlib.util
import json
import math
import random
import sys
import time
import types
from pathlib import Path

Point = Tuple[float, float]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]

def canon_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)

def orient(p: Point, q: Point, r: Point) -> float:
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

def is_convex_quad(points: List[Point], u: int, v: int, a: int, b: int) -> bool:
    pu, pv, pa, pb = points[u], points[v], points[a], points[b]
    s1 = orient(pu, pv, pa)
    s2 = orient(pu, pv, pb)
    if s1 == 0 or s2 == 0:
        return False
    if s1 * s2 >= 0:
        return False
    if s1 > 0:
        order = (u, a, v, b)
    else:
        order = (u, b, v, a)
    p0, p1, p2, p3 = order
    o1 = orient(points[p0], points[p1], points[p2])
    o2 = orient(points[p1], points[p2], points[p3])
    o3 = orient(points[p2], points[p3], points[p0])
    o4 = orient(points[p3], points[p0], points[p1])
    return (o1 > 0 and o2 > 0 and o3 > 0 and o4 > 0) or (o1 < 0 and o2 < 0 and o3 < 0 and o4 < 0)

@dataclass
class Triangulation:
    points: List[Point]
    edges: Set[Edge]
    adj: List[Set[int]]
    _ver: List[int]
    _nbr_cache: Dict[int, Tuple[int, List[int]]]

    @classmethod
    def from_edges(cls, points: List[Point], edges: Set[Edge]) -> "Triangulation":
        n = len(points)
        adj: List[Set[int]] = [set() for _ in range(n)]
        for (u, v) in edges:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
        return cls(points=points, edges=set(edges), adj=adj, _ver=[0]*n, _nbr_cache={})

    def _sorted_neighbors_ccw(self, v: int) -> List[int]:
        ver = self._ver[v]
        cached = self._nbr_cache.get(v)
        if cached is not None and cached[0] == ver:
            return cached[1]
        pv = self.points[v]
        nbrs = list(self.adj[v])
        nbrs.sort(key=lambda u: math.atan2(self.points[u][1] - pv[1], self.points[u][0] - pv[0]))
        self._nbr_cache[v] = (ver, nbrs)
        return nbrs

    def _next_ccw(self, v: int, u_prev: int) -> Optional[int]:
        nbrs = self._sorted_neighbors_ccw(v)
        if not nbrs:
            return None
        try:
            i = nbrs.index(u_prev)
        except ValueError:
            return None
        return nbrs[(i + 1) % len(nbrs)]

    def _left_face_third(self, u: int, v: int) -> Optional[int]:
        """Third vertex of the *triangle* to the left of directed edge u->v, else None."""
        w = self._next_ccw(v, u)
        if w is None:
            return None
        back = self._next_ccw(w, v)
        if back != u:
            return None
        return w

    def flippable_support(self, e: Edge) -> Optional[Tuple[int, int, int, int]]:
        u, v = e
        if e not in self.edges:
            return None
        a = self._left_face_third(u, v)
        b = self._left_face_third(v, u)
        if a is None or b is None or a == b:
            return None
        if canon_edge(a, b) in self.edges:
            return None
        if not is_convex_quad(self.points, u, v, a, b):
            return None
        return (u, v, a, b)

    def incident_triangles_of_edge(self, u: int, v: int, a: int, b: int) -> Tuple[Triangle, Triangle]:
        return (tuple(sorted((u, v, a))), tuple(sorted((u, v, b))))

    def _bump_ver(self, v: int):
        self._ver[v] += 1
        # cache invalidated by version mismatch automatically

    def apply_parallel_flip_step(self, step_edges: Iterable[Edge]) -> bool:
        step = [canon_edge(int(u), int(v)) for (u, v) in step_edges]
        if not step:
            return True

        if len(step) != len(set(step)):
            return False  # duplicate edges in same step

        supports: Dict[Edge, Tuple[int, int, int, int]] = {}
        used_tri: Set[Triangle] = set()

        for e in step:
            sup = self.flippable_support(e)
            if sup is None:
                return False
            u, v, a, b = sup
            t1, t2 = self.incident_triangles_of_edge(u, v, a, b)
            if t1 in used_tri or t2 in used_tri:
                return False
            used_tri.add(t1); used_tri.add(t2)
            supports[e] = sup

        # Apply flips (diagonals), update adjacency and versions only for affected vertices.
        for e, (u, v, a, b) in supports.items():
            new_e = canon_edge(a, b)

            if e not in self.edges:
                return False
            self.edges.remove(e)
            self.adj[u].discard(v); self.adj[v].discard(u)
            self._bump_ver(u); self._bump_ver(v)

            if new_e in self.edges:
                return False
            self.edges.add(new_e)
            self.adj[a].add(b); self.adj[b].add(a)
            self._bump_ver(a); self._bump_ver(b)

        return True

def install_minimal_triangulation_ops():
    mod = types.ModuleType("triangulation_ops")
    mod.Triangulation = Triangulation
    mod.canon_edge = canon_edge
    mod.Edge = Edge
    mod.Point = Point
    sys.modules["triangulation_ops"] = mod

def load_best_program(best_py: str):
    install_minimal_triangulation_ops()
    best_py = str(Path(best_py).resolve())
    spec = importlib.util.spec_from_file_location("best_program", best_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import best program from: {best_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_instance(instance_path: Path):
    data = json.load(open(instance_path, "r", encoding="utf-8"))
    points: List[Point] = list(zip(data["points_x"], data["points_y"]))
    triangulations: List[Set[Edge]] = []
    for tri_edges in data["triangulations"]:
        E = {canon_edge(int(u), int(v)) for (u, v) in tri_edges}
        triangulations.append(E)
    return data, points, triangulations

def load_solution(solution_path: Path):
    sol = json.load(open(solution_path, "r", encoding="utf-8"))
    flips_raw = sol["flips"]
    flips: List[List[List[Edge]]] = []
    for tri_seq in flips_raw:
        steps: List[List[Edge]] = []
        for step in tri_seq:
            steps.append([canon_edge(int(u), int(v)) for (u, v) in step])
        flips.append(steps)
    return sol, flips

def apply_sequence(points: List[Point], edges_start: Set[Edge], seq: List[List[Edge]]) -> Set[Edge]:
    T = Triangulation.from_edges(points, set(edges_start))
    for step in seq:
        if not T.apply_parallel_flip_step(step):
            raise ValueError(f"Invalid flip step: {step}")
    return set(T.edges)

def try_plan(planner_fn, points: List[Point], start_edges: Set[Edge], goal_edges: Set[Edge],
             base_seed: int, max_steps: int, tries: int):
    symdiff = len(start_edges.symmetric_difference(goal_edges))
    lb = math.ceil(symdiff / 2)
    base_budget = max_steps if max_steps > 0 else (4 * lb + 10)

    for t in range(tries):
        seed = base_seed + 7919 * t
        budget = int(base_budget * (1 + 0.25 * t))
        random.seed(seed)
        seq = planner_fn(points, set(start_edges), set(goal_edges), max_steps=budget, seed=seed)
        try:
            end_edges = apply_sequence(points, start_edges, seq)
        except Exception:
            continue
        if end_edges == goal_edges:
            return seq
    return None

def to_json_steps(seq: List[List[Edge]]):
    return [[[int(u), int(v)] for (u, v) in step] for step in seq]

def resolve_instance_path(instances_dir: Path, solution_path: Path, sol_json: dict) -> Optional[Path]:
    uid = sol_json.get("instance_uid")
    if isinstance(uid, str):
        p = instances_dir / f"{uid}.json"
        if p.exists():
            return p
    # fallback: A.solution.json -> A.json
    name = solution_path.name
    if name.endswith(".solution.json"):
        base = name[:-len(".solution.json")]
        p2 = instances_dir / f"{base}.json"
        if p2.exists():
            return p2
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="Path to evolved best_program.py")
    ap.add_argument("--instances_dir", default="data/benchmark_instances")
    ap.add_argument("--solutions_dir", default="opt")
    ap.add_argument("--out_dir", default="OpenEvolve_opt")
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 => no limit")
    args = ap.parse_args()

    instances_dir = Path(args.instances_dir)
    solutions_dir = Path(args.solutions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mod = load_best_program(args.best)
    if not hasattr(best_mod, "plan_parallel_flip_sequence"):
        raise SystemExit("best_program.py must define plan_parallel_flip_sequence(...)")

    planner_fn = best_mod.plan_parallel_flip_sequence

    sol_paths = sorted(solutions_dir.glob("*.solution.json"))
    if args.limit and args.limit > 0:
        sol_paths = sol_paths[:args.limit]

    total = 0
    improved = 0
    skipped = 0
    failed = 0

    print(f"Found {len(sol_paths)} solution files in {solutions_dir}")
    print(f"Output dir: {out_dir} (created if missing)\n")

    for sol_path in sol_paths:
        total += 1
        try:
            sol_json, old_flips = load_solution(sol_path)
        except Exception as e:
            print(f"[SKIP] {sol_path.name}: cannot read solution json ({e})")
            skipped += 1
            continue

        inst_path = resolve_instance_path(instances_dir, sol_path, sol_json)
        if inst_path is None:
            print(f"[SKIP] {sol_path.name}: cannot find instance under {instances_dir}")
            skipped += 1
            continue

        try:
            inst_json, points, tris = load_instance(inst_path)
        except Exception as e:
            print(f"[SKIP] {sol_path.name}: cannot read instance ({e})")
            skipped += 1
            continue

        uid_i = inst_json.get("instance_uid")
        uid_s = sol_json.get("instance_uid")
        if uid_i != uid_s:
            print(f"[SKIP] {sol_path.name}: instance_uid mismatch (inst={uid_i}, sol={uid_s})")
            skipped += 1
            continue

        if len(old_flips) != len(tris):
            print(f"[SKIP] {sol_path.name}: #flips({len(old_flips)}) != #triangulations({len(tris)})")
            skipped += 1
            continue

        # 1) derive center from old flips
        try:
            centers = []
            for i, (E0, seq) in enumerate(zip(tris, old_flips)):
                centers.append(apply_sequence(points, E0, seq))
            center = centers[0]
            ok = all(Ei == center for Ei in centers[1:])
            if not ok:
                # find first mismatch
                for i, Ei in enumerate(centers):
                    if Ei != center:
                        sd = len(Ei.symmetric_difference(center))
                        raise ValueError(f"not a single center; mismatch at i={i}, symdiff={sd}")
        except Exception as e:
            print(f"[FAIL] {sol_path.name}: existing solution invalid ({e})")
            failed += 1
            continue

        old_total = sum(len(seq) for seq in old_flips)

        # 2) plan new flips to that center
        new_flips: List[List[List[Edge]]] = []
        new_total = 0
        t0 = time.time()
        ok_all = True
        for i, E0 in enumerate(tris):
            seq = try_plan(
                planner_fn=planner_fn,
                points=points,
                start_edges=E0,
                goal_edges=center,
                base_seed=args.seed + i * 104729,
                max_steps=args.max_steps,
                tries=args.tries,
            )
            if seq is None:
                ok_all = False
                break
            new_flips.append(seq)
            new_total += len(seq)
        dt = time.time() - t0

        if not ok_all:
            print(f"[FAIL] {sol_path.name}: planner could not reach center for some triangulation (tries={args.tries}).")
            failed += 1
            continue

        delta = new_total - old_total
        status = "IMPROVED" if new_total < old_total else ("TIED" if new_total == old_total else "WORSE")
        print(f"[{status}] {sol_path.name}: old={old_total}, new={new_total}, delta={delta}, time={dt:.2f}s")

        if new_total < old_total:
            out_path = out_dir / sol_path.name
            new_sol = {
                "content_type": sol_json.get("content_type", "CGSHOP2026_Solution"),
                "instance_uid": uid_i,
                "flips": [to_json_steps(seq) for seq in new_flips],
                "meta": dict(sol_json.get("meta", {})),
            }
            # Keep old and new totals for traceability
            new_sol["meta"]["dist_old"] = int(old_total)
            new_sol["meta"]["dist"] = int(new_total)
            new_sol["meta"]["improved_by"] = int(old_total - new_total)
            new_sol["meta"]["recomputed_by"] = "OpenEvolve best_program"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(new_sol, f, ensure_ascii=False, indent=2)
            improved += 1

    print("\n--- Summary ---")
    print(f"processed: {total}")
    print(f"improved:  {improved}")
    print(f"failed:    {failed}")
    print(f"skipped:   {skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
