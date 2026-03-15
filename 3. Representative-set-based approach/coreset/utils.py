import json
import sys
import types
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Basic Edge Helpers
def _norm_edge(u: int, v: int) -> Tuple[int, int]:
    return (int(u), int(v)) if u < v else (int(v), int(u))

def _is_edge_list(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(
        isinstance(e, (list, tuple)) and len(e) == 2 and 
        isinstance(e[0], (int, float)) and isinstance(e[1], (int, float)) for e in x
    )

def _as_edges(x: List[Any]) -> List[Tuple[int, int]]:
    return [_norm_edge(e[0], e[1]) for e in x]


# Solution Parsing & Center Reconstruction
def _find_opt_file(opt_dir: str, uid: str) -> Optional[Path]:
    opt_path = Path(opt_dir)
    if not opt_path.exists(): return None
    direct = opt_path / f"{uid}.solution.json"
    if direct.exists(): return direct
    cands = sorted(opt_path.glob(f"*{uid}*.solution.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def _extract_explicit_edges(sol: dict) -> Optional[List[Tuple[int, int]]]:
    meta = sol.get("meta", {})
    for key in ("center", "center_edges"):
        if key in meta and _is_edge_list(meta[key]): return _as_edges(meta[key])
    if "center" in sol and _is_edge_list(sol["center"]): return _as_edges(sol["center"])
    return None

def _parse_flips(flips: Any) -> List[List[Tuple[int, int]]]:
    if flips is None: return []
    if _is_edge_list(flips): return [_as_edges(flips)]
    rounds = []
    for item in flips:
        if _is_edge_list(item): rounds.append(_as_edges(item))
        elif isinstance(item, list):
            for r in item:
                if _is_edge_list(r): rounds.append(_as_edges(r))
    return rounds

def get_center_from_sol(data, opt_dir: str, uid: str) -> Tuple[Optional[object], dict]:
    sol_path = _find_opt_file(opt_dir, uid)
    if not sol_path: return None, {"method": "not_found"}
    meta_info = {"sol_file": sol_path.name, "method": None}
    try:
        sol = json.loads(sol_path.read_text(encoding="utf-8"))
    except Exception as e:
        meta_info.update({"method": "read_failed", "error": str(e)})
        return None, meta_info

    explicit_edges = _extract_explicit_edges(sol)
    if explicit_edges:
        try:
            center_tri = data.make_triangulation([[u, v] for u, v in explicit_edges])
            meta_info.update({"method": "explicit_edges", "edges_len": len(explicit_edges)})
            return center_tri, meta_info
        except Exception as e:
            meta_info["explicit_build_failed"] = str(e)

    rounds = _parse_flips(sol.get("flips"))
    if not rounds: return None, meta_info

    first_set = set(rounds[0])
    cands = [i for i, tri in enumerate(data.triangulations) if first_set.issubset(set(tri.edges))]
    if not cands: cands = list(range(min(50, len(data.triangulations))))

    for idx in cands:
        tri = data.triangulations[idx].fast_copy()
        try:
            for rnd in rounds:
                for u, v in rnd: tri.flip((u, v))
            meta_info.update({"method": "reconstructed", "start_idx": idx, "num_rounds": len(rounds)})
            return tri, meta_info
        except Exception: continue

    meta_info.update({"method": "reconstruct_failed", "num_rounds": len(rounds)})
    return None, meta_info

# Safe Distance & Data Monkey Patching
def _pfp_noassert(data, tri1, tri2, variant: int = 1, max_iters: int = 100000) -> Tuple[List[List[Tuple[int, int]]], bool]:
    """Executes greedy parallel flip path search. Returns actual path and success flag."""
    tri = tri1.fast_copy()
    pfp, prev_flip = [], []
    depth = int(getattr(sys.modules.get(data.__class__.__module__), "SEARCH_DEPTH", 1))

    for _ in range(max_iters):
        cand = []
        for e in tri.edges:
            if data.flippable(tri, e) and not (variant == 2 and e in prev_flip):
                score = data.flip_score(tri, tri2, e, depth if variant == 1 else 0)
                if score[0] > 0: cand.append((e, score))
        
        if not cand: break
        cand.sort(key=lambda x: x[1], reverse=True)
        
        flips, marked = [], set()
        for (p1, p2), _ in cand:
            t1, t2 = tri.find_triangle(p1, p2), tri.find_triangle(p2, p1)
            if t1 not in marked and t2 not in marked:
                flips.append((p1, p2))
                marked.update([t1, t2])
                
        for e in flips:
            if variant == 1: tri.flip(e)
            else: prev_flip.append(tri.flip(e))
        pfp.append(flips)

    return pfp, (tri.edges == tri2.edges)

def get_safe_distances(data, center_tri, mode: str = "min", weighted: bool = True):
    info = {"fallback": False, "success": 0, "fail": 0}
    try:
        dist_list, total = data.computePFS_total(center_tri, mode=mode, weighted=weighted)
        return list(dist_list), float(total), info
    except Exception as e:
        info.update({"fallback": True, "error": str(e)})

    weights = data.get_weights(len(data.triangulations)) if hasattr(data, 'get_weights') else [1.0] * len(data.triangulations)
    dist_list, total = [], 0.0

    for i, tri_i in enumerate(data.triangulations):
        if mode == "pfp2": pfp, ok = _pfp_noassert(data, center_tri, tri_i, 2)
        elif mode == "pfp": pfp, ok = _pfp_noassert(data, center_tri, tri_i, 1)
        else:
            pfp1, ok1 = _pfp_noassert(data, center_tri, tri_i, 1)
            pfp2, ok2 = _pfp_noassert(data, center_tri, tri_i, 2)
            pfp, ok = (pfp1 if len(pfp1) <= len(pfp2) else pfp2), (ok1 or ok2)
        
        dd = float(len(pfp)) * (float(weights[i]) if weighted else 1.0)
        dist_list.append(dd)
        total += dd
        info["success" if ok else "fail"] += 1

    return dist_list, total, info

def patch_noassert(dt):
    """Monkey-patches Data instance to return actual paths without crashing."""
    dt.parallel_flip_path = types.MethodType(lambda self, t1, t2: _pfp_noassert(self, t1, t2, 1)[0], dt)
    dt.parallel_flip_path2 = types.MethodType(lambda self, t1, t2: _pfp_noassert(self, t1, t2, 2)[0], dt)


def load_best_costs(csv_path: str) -> dict:
    """
    Parses the result CSV to extract the minimum cost (best) for each instance.
    Expected CSV format: Column 0 is irrelevant, Column 1 onwards are instance names.
    """
    path = Path(csv_path)
    best_costs = {}
    if not path.exists():
        print(f"[WARN] Reference CSV not found at {path}")
        return best_costs

    try:
        with open(path, "r", encoding="utf-8") as f:
            import csv
            import math
            reader = csv.reader(f)
            header = next(reader, None)
            if not header: return best_costs

            # Map instance names from header (skip the first empty/index column)
            inst_names = [h.strip() for h in header[1:]]
            n_inst = len(inst_names)
            best_vals = [math.inf] * n_inst

            for row in reader:
                if not row: continue
                # Parse values for each instance column
                for j in range(1, min(len(row), n_inst + 1)):
                    val_str = row[j].strip()
                    if not val_str: continue
                    try:
                        v = float(val_str)
                        if v < best_vals[j - 1]:
                            best_vals[j - 1] = v
                    except ValueError:
                        continue

            # Populate dictionary with valid finite costs
            for name, v in zip(inst_names, best_vals):
                if math.isfinite(v):
                    best_costs[name] = v
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")

    return best_costs

def get_core_weights(json_path: str, n: int) -> List[float]:
    """
    Retrieves coreset weights from the JSON file. 
    Defaults to 1.0 for all points if the key is missing or invalid.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            root = json.load(f)
        weights = [float(x) for x in root.get("coreset_weights", [])]
        # Pad with 1.0 if coreset_weights list is shorter than triangulation count
        if len(weights) < n:
            return weights + [1.0] * (n - len(weights))
        return weights[:n]
    except Exception:
        return [1.0] * n


def evaluate_distance_and_path(data_obj, center_tri):
    """
    Computes flip paths and total distance to the center.
    This version uses the already defined _pfp_noassert logic 
    to prevent crashes during flip sequence verification.
    """
    data_obj.pFlips = []
    
    for tri in data_obj.triangulations:
        # Use the internal _pfp_noassert to safely get path without crashes
        # variant=1 is the default greedy parallel flip path
        path, success = _pfp_noassert(data_obj, tri, center_tri, variant=1)
        
        # Sanitize: ensure no None elements exist in the rounds
        clean_path = []
        for rnd in path:
            if rnd:
                clean_rnd = [e for e in rnd if e is not None]
                if clean_rnd:
                    clean_path.append(clean_rnd)
        
        data_obj.pFlips.append(clean_path)
    
    # Set the total distance (sum of parallel flip rounds)
    data_obj.dist = sum(len(p) for p in data_obj.pFlips)