import argparse
import json
import time, math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from data import Data


# =========================================================
# Utilities
# =========================================================

def _norm_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _is_edge_list(x) -> bool:
    """True if x looks like [[u,v], ...] with ints."""
    if not isinstance(x, list) or len(x) == 0:
        return False
    for e in x:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            return False
        if not (isinstance(e[0], (int, float)) and isinstance(e[1], (int, float))):
            return False
    return True


def _as_edges(x) -> List[Tuple[int, int]]:
    return [(_norm_edge(int(e[0]), int(e[1]))) for e in x]


# =========================================================
# Find & parse opt solution
# =========================================================

def find_opt_solution_file(opt_dir: str, instance_uid: str) -> Optional[Path]:
    """
    Find matching *.solution.json in opt_dir for instance_uid.
    Prefers exact match {instance_uid}.solution.json, otherwise newest *{instance_uid}*.solution.json.
    """
    opt_path = Path(opt_dir)
    if not opt_path.exists():
        return None

    direct = opt_path / f"{instance_uid}.solution.json"
    if direct.exists():
        return direct

    cands = list(opt_path.glob(f"*{instance_uid}*.solution.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def extract_explicit_center_edges(sol: dict) -> Optional[List[Tuple[int, int]]]:
    """
    Some solutions may explicitly store the final triangulation edges as 'center'.
    We support:
      - sol["meta"]["center"] or sol["meta"]["center_edges"]
      - sol["center"]
    """
    if not isinstance(sol, dict):
        return None

    meta = sol.get("meta", {})
    if isinstance(meta, dict):
        for key in ("center", "center_edges"):
            if key in meta and _is_edge_list(meta[key]):
                return _as_edges(meta[key])

    if "center" in sol and _is_edge_list(sol["center"]):
        return _as_edges(sol["center"])

    return None


def normalize_flip_phases(flips) -> List[List[List[Tuple[int, int]]]]:
    """
    Normalize solution["flips"] into:
        phases -> rounds -> edges (normalized tuples)

    Accepts common shapes:
      (A) flips = [[u,v], ...]                          (single round)
      (B) flips = [ [[u,v],...], [[u,v],...], ... ]     (rounds)
      (C) flips = [ phase1, phase2, ... ],
          where phase = [ [[u,v],...], [[u,v],...], ... ]
    """
    if flips is None:
        return []

    # (A) single round edge list
    if _is_edge_list(flips):
        return [[[_as_edges(flips)]]]

    # (B) list of rounds
    if isinstance(flips, list) and len(flips) > 0 and _is_edge_list(flips[0]):
        rounds = []
        for r in flips:
            if _is_edge_list(r):
                rounds.append(_as_edges(r))
        return [[rounds]]

    # (C) list of phases
    if (
        isinstance(flips, list)
        and len(flips) > 0
        and isinstance(flips[0], list)
        and len(flips[0]) > 0
        and _is_edge_list(flips[0][0])
    ):
        phases = []
        for ph in flips:
            if not isinstance(ph, list):
                continue
            rounds = []
            for r in ph:
                if _is_edge_list(r):
                    rounds.append(_as_edges(r))
            if rounds:
                phases.append(rounds)
        return [phases]  # NOTE: wrap one more level to keep return type stable

    return []


def flatten_rounds_from_phases(phases_wrapped: List[List[List[List[Tuple[int, int]]]]]) -> List[List[Tuple[int, int]]]:
    """
    phases_wrapped is either:
      - [[ rounds ]] or
      - [[ phase1_rounds, phase2_rounds, ... ]]

    We flatten to a single list of rounds in chronological order.
    """
    if not phases_wrapped:
        return []
    top = phases_wrapped[0]
    # top could be "rounds" (list of round) OR "phases" (list of rounds-list)
    if len(top) == 0:
        return []
    if top and isinstance(top[0], list) and top and (len(top[0]) > 0) and isinstance(top[0][0], tuple):
        # It's already rounds: [round, round, ...]
        return top  # type: ignore
    # else it's phases: [phase_rounds, phase_rounds, ...]
    rounds: List[List[Tuple[int, int]]] = []
    for phase_rounds in top:  # type: ignore
        for r in phase_rounds:
            rounds.append(r)
    return rounds


# =========================================================
# Reconstruct center triangulation from flip sequence
# =========================================================

def try_apply_rounds(tri, rounds: List[List[Tuple[int, int]]]) -> Optional[object]:
    """
    Apply all rounds to a fast_copy of tri.
    Returns the resulting triangulation if succeeded, otherwise None.
    """
    t = tri.fast_copy()
    try:
        for rnd in rounds:
            for (u, v) in rnd:
                t.flip((u, v))
        return t
    except Exception:
        return None


def reconstruct_center_from_solution(data: Data, sol_path: Path) -> Tuple[Optional[object], dict]:
    """
    Build a center triangulation object using solution.json.

    Priority:
      1) If explicit center edges exist -> build with data.make_triangulation
      2) Else, use flip sequence:
         - Determine a valid starting triangulation by checking
           whether first round edges are contained, then verifying by applying all rounds.
         - Apply all flips sequentially to get the final center triangulation.

    Returns:
      (centerT or None, meta_info dict)
    """
    meta_info = {"sol_file": sol_path.name, "method": None}

    try:
        sol = json.loads(sol_path.read_text(encoding="utf-8"))
    except Exception as e:
        meta_info["method"] = "read_failed"
        meta_info["error"] = str(e)
        return None, meta_info

    # (1) explicit center
    center_edges = extract_explicit_center_edges(sol)
    if center_edges is not None:
        try:
            centerT = data.make_triangulation([[u, v] for (u, v) in center_edges])
            meta_info["method"] = "explicit_center_edges"
            meta_info["center_edges_len"] = len(center_edges)
            return centerT, meta_info
        except Exception as e:
            meta_info["explicit_center_build_failed"] = str(e)

    # (2) flip sequence
    flips = sol.get("flips", None)
    phases_wrapped = normalize_flip_phases(flips)
    rounds = flatten_rounds_from_phases(phases_wrapped)
    if not rounds:
        meta_info["method"] = "no_flips"
        return None, meta_info

    first_set = set(rounds[0])

    # Candidate starts: triangulations containing all edges in the first round
    candidates = []
    for idx, tri in enumerate(data.triangulations):
        try:
            if first_set.issubset(set(tri.edges)):
                candidates.append(idx)
        except Exception:
            continue

    # If nothing matches, fall back to trying a small prefix (deterministic)
    if not candidates:
        candidates = list(range(min(50, len(data.triangulations))))

    # Try candidates deterministically until flip application succeeds
    for idx in candidates:
        tri0 = data.triangulations[idx]
        res = try_apply_rounds(tri0, rounds)
        if res is not None:
            meta_info["method"] = "reconstructed_from_flips"
            meta_info["start_index"] = idx
            meta_info["num_rounds"] = len(rounds)
            meta_info["first_round_edges"] = len(rounds[0])
            return res, meta_info

    meta_info["method"] = "reconstruct_failed"
    meta_info["num_rounds"] = len(rounds)
    meta_info["candidates_tried"] = len(candidates)
    return None, meta_info


# =========================================================
# Safe distance computation (avoid AssertionError in data.py)
# =========================================================

def _pfp_len_noassert(data: Data, tri1, tri2, variant: int = 1, max_iters: int = 100000) -> Tuple[int, bool]:
    """
    Replicate data.parallel_flip_path / parallel_flip_path2 but DO NOT assert at the end.
    Returns (len_rounds, success).
    """
    tri = tri1.fast_copy()
    pfp = []
    iters = 0

    while True:
        iters += 1
        if iters > max_iters:
            break

        cand = []
        edges = list(tri.edges)

        # mimic original behavior
        prev_flip = []

        for e in edges:
            if data.flippable(tri, e):
                if variant == 2 and e in prev_flip:
                    continue
                depth = getattr(data, "SEARCH_DEPTH", 1)
                score = data.flip_score(tri, tri2, e, depth if variant == 1 else 0)
                if score[0] > 0:
                    cand.append((e, score))

        if not cand:
            break

        cand.sort(key=lambda x: x[1], reverse=True)

        flips = []
        marked = set()
        for (p1, p2), _ in cand:
            t1 = tri.find_triangle(p1, p2)
            t2 = tri.find_triangle(p2, p1)
            if t1 in marked or t2 in marked:
                continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)

        # apply flips
        for e in flips:
            if variant == 1:
                tri.flip(e)
            else:
                e1 = tri.flip(e)
                prev_flip.append(e1)

        pfp.append(flips)

    success = (tri.edges == tri2.edges)
    return len(pfp), success


def computePFS_total_safe(data: Data, centerT, mode: str, weighted: bool) -> Tuple[List[float], float, dict]:
    """
    Try data.computePFS_total(centerT,...). If AssertionError occurs,
    fall back to no-assert greedy lengths.

    Returns: (dist_list, total, info)
    """
    info = {"used_fallback": False, "fallback_success_cnt": 0, "fallback_fail_cnt": 0}

    try:
        dist_list, total = data.computePFS_total(centerT, mode=mode, weighted=weighted)
        return list(dist_list), float(total), info
    except AssertionError:
        info["used_fallback"] = True
    except Exception as e:
        info["used_fallback"] = True
        info["fallback_error"] = str(e)

    # fallback: compute per-triangulation
    weights = None
    try:
        weights = data.get_weights(len(data.triangulations))
    except Exception:
        weights = [1.0] * len(data.triangulations)

    dist_list = []
    total = 0.0

    for i, tri_i in enumerate(data.triangulations):
        if mode == "pfp2":
            d, ok = _pfp_len_noassert(data, centerT, tri_i, variant=2)
        elif mode == "pfp":
            d, ok = _pfp_len_noassert(data, centerT, tri_i, variant=1)
        else:
            d1, ok1 = _pfp_len_noassert(data, centerT, tri_i, variant=1)
            d2, ok2 = _pfp_len_noassert(data, centerT, tri_i, variant=2)
            d = min(d1, d2)
            ok = ok1 or ok2

        dd = float(d)
        if weighted:
            dd *= float(weights[i])

        dist_list.append(dd)
        total += dd

        if ok:
            info["fallback_success_cnt"] += 1
        else:
            info["fallback_fail_cnt"] += 1

    return dist_list, float(total), info


# =========================================================
# Coreset builder (ring + 1D grid)
# =========================================================

def _dist_between_tris_safe(data, triA, triB, mode: str = "min", max_iters: int = 100000) -> float:
    """
    Metric distance assumed. We compute a greedy PFP length (no-assert) similarly to your fallback.
    mode in {"pfp","pfp2","min"}.
    """
    def _pfp_len_variant(tri1, tri2, variant: int) -> int:
        tri = tri1.fast_copy()
        iters = 0
        prev_flip = []

        while True:
            iters += 1
            if iters > max_iters:
                break

            cand = []
            edges = list(tri.edges)
            for e in edges:
                if data.flippable(tri, e):
                    if variant == 2 and e in prev_flip:
                        continue
                    depth = getattr(data, "SEARCH_DEPTH", 1)
                    score = data.flip_score(tri, tri2, e, depth if variant == 1 else 0)
                    if score[0] > 0:
                        cand.append((e, score))

            if not cand:
                break

            cand.sort(key=lambda x: x[1], reverse=True)

            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)

            for e in flips:
                if variant == 1:
                    tri.flip(e)
                else:
                    e1 = tri.flip(e)
                    prev_flip.append(e1)

        return iters  # NOTE: rounds length in your code is len(pfp); here we use iters proxy? better: count rounds.

    # Better: count "rounds" like your _pfp_len_noassert
    def _pfp_rounds(tri1, tri2, variant: int) -> int:
        tri = tri1.fast_copy()
        pfp = []
        iters = 0
        prev_flip = []
        while True:
            iters += 1
            if iters > max_iters:
                break

            cand = []
            edges = list(tri.edges)
            for e in edges:
                if data.flippable(tri, e):
                    if variant == 2 and e in prev_flip:
                        continue
                    depth = getattr(data, "SEARCH_DEPTH", 1)
                    score = data.flip_score(tri, tri2, e, depth if variant == 1 else 0)
                    if score[0] > 0:
                        cand.append((e, score))

            if not cand:
                break

            cand.sort(key=lambda x: x[1], reverse=True)

            flips = []
            marked = set()
            for (p1, p2), _ in cand:
                t1 = tri.find_triangle(p1, p2)
                t2 = tri.find_triangle(p2, p1)
                if t1 in marked or t2 in marked:
                    continue
                flips.append((p1, p2))
                marked.add(t1)
                marked.add(t2)

            for e in flips:
                if variant == 1:
                    tri.flip(e)
                else:
                    e1 = tri.flip(e)
                    prev_flip.append(e1)

            pfp.append(flips)

        return len(pfp)

    if mode == "pfp":
        return float(_pfp_rounds(triA, triB, variant=1))
    if mode == "pfp2":
        return float(_pfp_rounds(triA, triB, variant=2))
    # mode == "min"
    d1 = float(_pfp_rounds(triA, triB, variant=1))
    d2 = float(_pfp_rounds(triA, triB, variant=2))
    return min(d1, d2)


def build_triangulation_coreset_cake(
    data,
    centerT,
    eps: float = 0.1,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    distance_mode: str = "min",
    # cake slicing params
    angle_beta: float = 4.0,
    max_anchors_per_ring: int = 32,
    fps_sample_limit: int = 250,
    use_radial_cell: bool = False,
    seed: int = 0,
):
    """
    'Cake-like' coreset:
      1) ring by dist-to-center
      2) inside each ring, pick anchor set via farthest-point sampling (a tau-net)
      3) assign each point to nearest anchor -> angular/wedge-like buckets
      4) (optional) also split by radial 1D cell
      5) keep 1 representative per bucket, aggregate weights.

    Returns:
      S_idx: np.ndarray[int]   representative indices (in data.triangulations)
      S_wts: np.ndarray[float] aggregated weights
      info: dict (stats)
    """
    rng = np.random.default_rng(seed)
    n = len(data.triangulations)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float), {"empty": True}

    weights = np.asarray(data.get_weights(n), dtype=float)
    W = float(weights.sum())

    # --- compute pure center distances delta_i (do NOT multiply weights here; keep metric clean) ---
    # We reuse your safe center distance code path idea:
    # Use Data.computePFS_total if available; otherwise fallback to greedy.
    # Here we call your computePFS_total_safe but force weighted=False if you paste it in same file.
    # If not available, do it directly with safe pair distance.
    try:
        # if your computePFS_total_safe is in scope:
        delta_list, _, _ = computePFS_total_safe(data, centerT, mode=distance_mode, weighted=False)
        delta = np.asarray(delta_list, dtype=float)
    except NameError:
        # fallback: compute delta by pairwise safe distance
        delta = np.zeros(n, dtype=float)
        for i in range(n):
            delta[i] = _dist_between_tris_safe(data, centerT, data.triangulations[i], mode=distance_mode)

    # scale R and ring count M (same spirit as your code)
    nu_A = float(np.sum(weights * delta))
    R = max(nu_A / max(W, 1e-12), 1e-12)
    Dmax = float(delta.max())

    if Dmax < R:
        rep = int(np.argmin(delta))
        return np.array([rep], dtype=int), np.array([W], dtype=float), {
            "method": "singleton",
            "R": R,
            "Dmax": Dmax,
        }

    M = max(1, int(math.ceil(math.log2((Dmax + R) / R))))

    # distance cache for inter-PFD in rings
    dist_cache: Dict[Tuple[int, int], float] = {}

    def d_idx(i: int, j: int) -> float:
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in dist_cache:
            return dist_cache[key]
        dij = _dist_between_tris_safe(data, data.triangulations[a], data.triangulations[b], mode=distance_mode)
        dist_cache[key] = dij
        return dij

    # ring assignment
    ring_of = np.zeros(n, dtype=int)
    for i in range(n):
        dist = float(delta[i])
        if dist < R:
            j = 0
        else:
            j = int(math.ceil(math.log2(max(dist, 1e-12) / R)))
        ring_of[i] = min(j, M)

    reps: List[int] = []
    rep_wts: List[float] = []
    stats = {"R": R, "Dmax": Dmax, "M": M, "rings": {}}

    # buckets: key -> (rep_idx, weight_sum)
    buckets: Dict[Tuple[int, int, int], List[float]] = {}  # (j, anchor_id, radial_cell) -> [rep, wsum]

    for j in range(0, M + 1):
        I = np.where(ring_of == j)[0].tolist()
        if not I:
            continue

        # define tau_j: target cluster radius within this ring
        # radius scale ~ R*2^j; keep tau proportional to eps * scale
        scale = R * (2 ** j)
        tau = (eps * scale) / float(angle_beta)
        tau = max(tau, (eps * R) / float(alpha_min))  # safety floor

        # --- choose anchors by farthest-point sampling until coverage <= tau or cap ---
        # start anchor: pick farthest from center in this ring (deterministic)
        start = max(I, key=lambda idx: float(delta[idx]))
        anchors = [start]

        # helper: compute min-dist-to-anchors for an index
        def min_dist_to_anchors(idx: int) -> float:
            return min(d_idx(idx, a) for a in anchors)

        while True:
            if len(anchors) >= max_anchors_per_ring:
                break

            # to speed up, optionally sample candidates
            cand = I
            if fps_sample_limit is not None and len(I) > fps_sample_limit:
                cand = rng.choice(I, size=fps_sample_limit, replace=False).tolist()

            farthest = None
            farthest_md = -1.0
            for idx in cand:
                md = min_dist_to_anchors(idx)
                if md > farthest_md:
                    farthest_md = md
                    farthest = idx

            # if even the farthest point is within tau -> coverage achieved
            if farthest is None or farthest_md <= tau:
                break

            anchors.append(int(farthest))

        # --- assign each point to nearest anchor (wedge-like partition) ---
        # optional: radial cell width like your original
        if use_radial_cell:
            r_rad = (eps * R * (2 ** j)) / float(alpha)
            r_rad = max(r_rad, (eps * R) / float(alpha_min))
        else:
            r_rad = None

        # For statistics
        stats["rings"][j] = {"size": len(I), "anchors": len(anchors), "tau": float(tau)}

        for idx in I:
            # nearest anchor id
            best_aid = 0
            best_da = float("inf")
            for aid, a in enumerate(anchors):
                da = d_idx(idx, a) if idx != a else 0.0
                if da < best_da:
                    best_da = da
                    best_aid = aid

            # radial cell inside ring (optional)
            if r_rad is None:
                cell = 0
            else:
                cell = int(math.floor(float(delta[idx]) / float(r_rad)))

            key = (j, best_aid, cell)
            wi = float(weights[idx])

            if key not in buckets:
                buckets[key] = [float(idx), wi]  # rep index stored as float in list for mutability
            else:
                buckets[key][1] += wi

    # finalize reps
    for rep_idx_f, wsum in buckets.values():
        reps.append(int(rep_idx_f))
        rep_wts.append(float(wsum))

    return np.array(reps, dtype=int), np.array(rep_wts, dtype=float), {
        **stats,
        "num_buckets": len(buckets),
        "num_reps": len(reps),
        "sum_weights": float(np.sum(rep_wts)),
        "dist_cache_size": len(dist_cache),
        "params": {
            "eps": eps,
            "alpha": alpha,
            "alpha_min": alpha_min,
            "angle_beta": angle_beta,
            "max_anchors_per_ring": max_anchors_per_ring,
            "fps_sample_limit": fps_sample_limit,
            "use_radial_cell": use_radial_cell,
            "distance_mode": distance_mode,
        },
    }



# =========================================================
# Main: build & save coresets
# =========================================================

def build_and_save_coresets_for_all_instances(
    input_dir: str = "./data/benchmark_instances",
    output_dir: str = "./data/coreset_instance",
    opt_dir: str = "../opt",
    eps: float = 0.1,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    distance_mode: str = "min",
    weighted_distance: bool = True,
    include_rirs: bool = False,
):
    """
    For each instance.json in input_dir:
      - find opt solution in opt_dir
      - reconstruct center triangulation from solution flips
      - build coreset using that center triangulation object
      - save to *_coreset.json (with coreset_weights)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_path.glob("*.json"))
    print(f"[INFO] Found {len(json_paths)} json files in {input_dir}")

    for json_file in json_paths:
        stem = json_file.stem
        if (not include_rirs) and stem.lower().startswith("rirs"):
            print(f"[SKIP] {json_file.name} (starts with 'rirs'; include_rirs=False)")
            continue

        print("\n==============================================")
        print(f"[PROCESS] {json_file.name}")
        start_time = time.time()

        # load instance json (for triangulation edge lists to export)
        try:
            inst = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] failed to read json: {json_file} ({e}) -> skip")
            continue

        base_uid = inst.get("instance_uid", json_file.stem)
        tri_list = inst.get("triangulations", [])
        if not isinstance(tri_list, list) or len(tri_list) == 0:
            print(f"[WARN] no triangulations in {json_file} -> skip")
            continue

        # Load Data (Triangulation objects & distance routines)
        data_full = Data(str(json_file))

        # Find opt solution
        sol_path = find_opt_solution_file(opt_dir, base_uid)
        if sol_path is None:
            print(f"[WARN] opt solution NOT FOUND for {base_uid} in {opt_dir} -> skip (need flips to get center)")
            continue

        # Reconstruct center
        centerT, center_meta = reconstruct_center_from_solution(data_full, sol_path)
        if centerT is None:
            print(f"[WARN] failed to reconstruct center from {sol_path.name} -> skip")
            print(f"       detail: {center_meta}")
            continue

        # Build coreset using the reconstructed center triangulation object
        S_idx, S_weights, dist_info = build_triangulation_coreset_cake(
            data_full,
            centerT=centerT,
            eps=0.5,
            alpha=8.0,
            alpha_min=16.0,
            distance_mode="min",
            angle_beta=1.0,
            max_anchors_per_ring=8,
            fps_sample_limit=250,
            use_radial_cell=False,   # 케이크 느낌이면 보통 False 추천
        )


        S_idx_list = np.asarray(S_idx, dtype=int).tolist()
        coreset_tris = [tri_list[i] for i in S_idx_list]
        coreset_wts = np.asarray(S_weights, dtype=float).tolist()

        coreset_uid = f"{base_uid}-coreset"
        coreset_inst = {
            "content_type": inst.get("content_type", "CGSHOP2026_Instance"),
            "instance_uid": coreset_uid,
            "points_x": inst["points_x"],
            "points_y": inst["points_y"],
            "triangulations": coreset_tris,
            "coreset_weights": coreset_wts,
            "meta": {
                "eps": eps,
                "alpha": alpha,
                "alpha_min": alpha_min,
                "distance_mode": distance_mode,
                "weighted_distance": weighted_distance,
                "include_rirs": include_rirs,
                "opt_dir": opt_dir,
                **center_meta,
                **dist_info,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        out_file = output_path / (json_file.stem + "_coreset.json")
        out_file.write_text(json.dumps(coreset_inst, indent=2), encoding="utf-8")

        elapsed = time.time() - start_time
        print(f"[DONE] reps={len(S_idx_list)}  sum_w={sum(coreset_wts):.3f}")
        print(f"       center_method={center_meta.get('method')}, sol={sol_path.name}")
        if dist_info.get("used_fallback"):
            print(f"       [WARN] fallback distance used: ok={dist_info.get('fallback_success_cnt')} fail={dist_info.get('fallback_fail_cnt')}")
        print(f"       saved -> {out_file}")
        print(f"       time  -> {elapsed:.3f}s")
        print("==============================================")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="./data/benchmark_instances")
    p.add_argument("--output_dir", type=str, default="./data/coreset_instance")
    p.add_argument("--opt_dir", type=str, default="../opt")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--alpha_min", type=float, default=16.0)
    p.add_argument("--distance_mode", type=str, default="min", choices=["pfp", "pfp2", "min"])
    p.add_argument("--weighted_distance", action="store_true", default=True)
    p.add_argument("--no_weighted_distance", action="store_false", dest="weighted_distance")
    p.add_argument("--include_rirs", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_save_coresets_for_all_instances(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        opt_dir=args.opt_dir,
        eps=args.eps,
        alpha=args.alpha,
        alpha_min=args.alpha_min,
        distance_mode=args.distance_mode,
        weighted_distance=args.weighted_distance,
        include_rirs=args.include_rirs,
    )
