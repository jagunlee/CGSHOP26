import copy
import math
import time
import numpy as np
from pathlib import Path
import json

# IMPORTANT: use the same Data + distance logic as generate_coreset.py
from data import Data


# -------------------------------------------------------
# 거리 계산: generate_coreset.py의 _pfp_len_noassert 로직 그대로 사용
# -------------------------------------------------------
def _pfp_len_noassert(data: Data, tri1, tri2, variant: int = 1, max_iters: int = 100000):
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


# -------------------------------------------------------
# 1) triangulation + PFP 거리 기반 1-median coreset (샘플링 + CAKE 버전)
# -------------------------------------------------------
def build_triangulation_coreset_practical(
    data: Data,
    eps: float = 0.5,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    use_random_center: bool = True,
    center_seed: int = 0,
    sample_size: int = 32,
    # ---- cake params ----
    distance_mode: str = "min",       # {"pfp","pfp2","min"}
    angle_beta: float = 4.0,          # 작을수록 tau↑, 조각 굵어짐(앵커↓)
    max_anchors_per_ring: int = 16,   # 링당 앵커 상한
    use_radial_cell: bool = False,    # True면 (ring,wedge) 내부 반지름 cell까지 추가 분할
    symmetrize: bool = True,          # True면 d(i,j)=min(d(i->j),d(j->i))로 대칭화
):
    """
    예전 코드 구조 유지:
      1) 전체 n 중 sample_size개 샘플링
      2) 샘플 내부에서 center 하나 선택
      3) center까지의 거리로 ring 나눔
      4) 각 ring에서 inter-PFD 기반 앵커(FPS)로 wedge(케이크 조각) 생성
      5) 각 (ring,wedge[,radial]) 버킷당 대표 1개 + 샘플 count를 weight로 사용 (전체 n으로 스케일)

    반환:
      S_idx      : 대표 triangulation 인덱스들 (np.ndarray[int], 원본 인덱스)
      S_weights  : 각 대표가 대표하는 "대략적인" 개수 (np.ndarray[int])
    """
    n = len(data.triangulations)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(center_seed)

    # ---------------------------------------------------
    # 1) 샘플링
    # ---------------------------------------------------
    if sample_size is None or sample_size >= n:
        idx_list = np.arange(n, dtype=int)  # 전체 사용
    else:
        idx_list = rng.choice(n, size=sample_size, replace=False)
    m = len(idx_list)

    # ---------------------------------------------------
    # 2) center 선택 (샘플 안에서)
    # ---------------------------------------------------
    if use_random_center:
        center_pos = int(rng.integers(0, m))
    else:
        center_pos = 0
    center_idx = int(idx_list[center_pos])  # 원본 인덱스
    centerT = data.triangulations[center_idx]

    # ---------------------------------------------------
    # 거리 캐시: (global_i, global_j) -> dist
    # ---------------------------------------------------
    dist_cache = {}

    def _direct_dist(g_i: int, g_j: int, variant: int) -> float:
        if g_i == g_j:
            return 0.0
        tri1 = data.triangulations[g_i]
        tri2 = data.triangulations[g_j]
        d, _ = _pfp_len_noassert(data, tri1, tri2, variant=variant)
        return float(d)

    def dist(g_i: int, g_j: int) -> float:
        if g_i == g_j:
            return 0.0
        a, b = (g_i, g_j) if g_i < g_j else (g_j, g_i)
        key = (a, b, distance_mode, symmetrize)
        if key in dist_cache:
            return dist_cache[key]

        def one_mode(var: int) -> float:
            if symmetrize:
                return min(_direct_dist(a, b, var), _direct_dist(b, a, var))
            return _direct_dist(a, b, var)

        if distance_mode == "pfp":
            outv = one_mode(1)
        elif distance_mode == "pfp2":
            outv = one_mode(2)
        else:
            outv = min(one_mode(1), one_mode(2))

        dist_cache[key] = float(outv)
        return float(outv)

    # ---------------------------------------------------
    # 3) 샘플에 대해 center까지 거리 배열
    # ---------------------------------------------------
    dist_to_center = np.zeros(m, dtype=float)
    for pos in range(m):
        g = int(idx_list[pos])
        dist_to_center[pos] = dist(center_idx, g)

    nu_A = float(dist_to_center.sum())
    R = max(nu_A / max(1, m), 1e-12)
    Dmax = float(dist_to_center.max())

    if Dmax < R:
        return np.array([center_idx], dtype=int), np.array([n], dtype=int)

    # ---------------------------------------------------
    # 4) ring + cake(wedge via anchors) 버킷팅
    # ---------------------------------------------------
    from math import ceil, log2
    tiny = 1e-12
    M = max(1, ceil(log2((Dmax + R) / R)))

    # ring index for each sampled position
    ring_of = np.zeros(m, dtype=int)
    for pos in range(m):
        d0 = float(dist_to_center[pos])
        if d0 < R:
            j = 0
        else:
            j = int(math.ceil(math.log2(max(d0, tiny) / R)))
        ring_of[pos] = min(j, M)

    buckets = {}  # key -> [rep_global_idx, count_in_sample]

    # helper: distance between two sample positions (by global indices)
    def d_pos(p: int, q: int) -> float:
        return dist(int(idx_list[p]), int(idx_list[q]))

    for j in range(0, M + 1):
        I = np.where(ring_of == j)[0]
        if len(I) == 0:
            continue

        scale = R * (2 ** j)
        tau = (eps * scale) / float(angle_beta)
        tau = max(tau, (eps * R) / float(alpha_min))

        # farthest-point sampling anchors in this ring
        start = int(I[np.argmax(dist_to_center[I])])
        anchors = [start]

        # maintain best distance to anchors + best anchor id for each member
        best_dist = np.array([d_pos(int(pos), start) for pos in I], dtype=float)
        best_aid = np.zeros(len(I), dtype=int)

        for t, pos in enumerate(I):
            if int(pos) == start:
                best_dist[t] = 0.0

        while True:
            if len(anchors) >= max_anchors_per_ring:
                break
            t_far = int(np.argmax(best_dist))
            farthest_pos = int(I[t_far])
            farthest_md = float(best_dist[t_far])

            if farthest_md <= tau:
                break

            anchors.append(farthest_pos)
            new_aid = len(anchors) - 1

            # update assignment against new anchor
            for t, pos in enumerate(I):
                pos = int(pos)
                dnew = 0.0 if pos == farthest_pos else d_pos(pos, farthest_pos)
                if dnew < best_dist[t]:
                    best_dist[t] = dnew
                    best_aid[t] = new_aid

        # optional radial cell (원하면 켜서 더 잘게)
        if use_radial_cell:
            r_rad = (eps * R * (2 ** j)) / float(alpha)
            r_rad = max(r_rad, (eps * R) / float(alpha_min))
        else:
            r_rad = None

        for t, pos in enumerate(I):
            pos = int(pos)
            wedge_id = int(best_aid[t])

            if r_rad is None:
                key = (int(j), wedge_id)
            else:
                cell = int(math.floor(float(dist_to_center[pos]) / float(r_rad)))
                key = (int(j), wedge_id, cell)

            if key not in buckets:
                buckets[key] = [int(idx_list[pos]), 1]
            else:
                buckets[key][1] += 1

    # ---------------------------------------------------
    # 5) reps / weights 추출 (샘플 count -> 전체 n 근사 스케일링)
    # ---------------------------------------------------
    if buckets:
        reps = []
        wts = []
        scale = float(n) / float(m)
        for rep_idx, cnt in buckets.values():
            reps.append(int(rep_idx))
            approx_weight = max(1, int(round(cnt * scale)))
            wts.append(approx_weight)
        return np.array(reps, dtype=int), np.array(wts, dtype=int)

    return np.array([center_idx], dtype=int), np.array([n], dtype=int)


# -------------------------------------------------------
# 2) 코어셋 triangulation 들만 가지고 find_center 실행
# -------------------------------------------------------
def make_coreset_data(data_full: Data, S_idx):
    coreset_data = copy.copy(data_full)  # 얕은 복사: pts, etc. 공유

    new_tris = [copy.deepcopy(data_full.triangulations[i]) for i in S_idx]
    coreset_data.triangulations = new_tris

    coreset_data.center = coreset_data.triangulations[0]
    coreset_data.flip = [[] for _ in coreset_data.triangulations]
    coreset_data.dist = float("inf")

    return coreset_data


# -------------------------------------------------------
# 3) 모든 인스턴스에 대해 코어셋 생성 후 저장 (per-instance 시간 출력)
# -------------------------------------------------------
def build_and_save_coresets_for_all_instances(
    input_dir: str = "./data/benchmark_instances",
    output_dir: str = "./data/coreset_instance",
    eps: float = 0.5,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    sample_size: int = 32,
    center_seed: int = 0,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_paths)} json files in {input_dir}")

    for json_file in json_paths:
        print(f"\n=== Processing {json_file.name} ===")
        start_time = time.time()

        data_full = Data(str(json_file))
        S_idx, S_weights = build_triangulation_coreset_practical(
            data_full,
            eps=eps,
            alpha=alpha,
            alpha_min=alpha_min,
            use_random_center=True,
            center_seed=center_seed,
            sample_size=sample_size,
            # cake params can be set here if you want
            distance_mode="min",
            angle_beta=1.0,
            max_anchors_per_ring=8,
            use_radial_cell=False,
            symmetrize=True,
        )
        print(f"  -> coreset reps: {len(S_idx)} / approx weights sum: {int(S_weights.sum())}")

        with open(json_file, "r") as f:
            inst = json.load(f)

        points_x = inst["points_x"]
        points_y = inst["points_y"]
        tri_list = inst["triangulations"]

        S_idx_list = np.asarray(S_idx, dtype=int).tolist()
        coreset_tris = [tri_list[i] for i in S_idx_list]

        base_uid = inst.get("instance_uid", json_file.stem)
        coreset_uid = f"{base_uid}-coreset"

        coreset_inst = {
            "content_type": inst.get("content_type", "CGSHOP2026_Instance"),
            "instance_uid": coreset_uid,
            "points_x": points_x,
            "points_y": points_y,
            "triangulations": coreset_tris,
            # "coreset_weights": np.asarray(S_weights, dtype=int).tolist(),  # 필요하면 주석 해제
        }

        out_name = json_file.stem + "_coreset.json"
        out_file = output_path / out_name

        with open(out_file, "w") as f:
            json.dump(coreset_inst, f, indent=2)

        elapsed = time.time() - start_time
        print(f"  -> saved to: {out_file}")
        print(f"  -> time taken: {elapsed:.3f} seconds")


if __name__ == "__main__":
    build_and_save_coresets_for_all_instances(
        input_dir="./data/benchmark_instances",
        output_dir="./data/coreset_instance",
        eps=0.5,
        alpha=8.0,
        alpha_min=16.0,
        sample_size=32,
        center_seed=0,
    )
