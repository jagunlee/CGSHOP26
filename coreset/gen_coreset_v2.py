import copy
import math
import time   # ★ 추가
import numpy as np
from pathlib import Path
import json
from data import Data

# -------------------------------------------------------
# 1) triangulation + compute_pfd 거리 기반 1-median coreset (매우 빠른 버전)
# -------------------------------------------------------

def build_triangulation_coreset_practical(
    data,
    eps: float = 0.1,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    use_random_center: bool = True,
    center_seed: int = 0,
    sample_size: int = 128,

    # --- cake params (추가) ---
    angle_beta: float = 4.0,          # 작을수록 조각 굵어짐(앵커↓, reps↓)
    max_anchors_per_ring: int = 16,   # 링당 최대 케이크 조각 수(앵커 수 상한)
    use_radial_cell: bool = False,    # True면 (ring, wedge) 안에서 반지름 1D cell까지 추가 분할
):
    """
    (샘플링 유지) + cake 방식 버킷팅:
      - ring: center까지 거리로 스케일 구간 나눔
      - wedge: ring 내부에서 inter-PFD로 farthest-point sampling 앵커를 뽑고,
               각 점을 가장 가까운 앵커에 할당(= metric Voronoi) → 케이크 조각
      - (optional) radial cell: 기존처럼 반지름 1D 셀까지 추가

    반환:
      S_idx      : 대표 triangulation 인덱스들 (np.ndarray[int], 원본 인덱스)
      S_weights  : 각 대표가 대표하는 "대략적인" 개수 (np.ndarray[int])
    """
    n = len(data.triangulations)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # --- compute_pfd(i, j) 캐시 ---
    pfd_cache = {}

    def get_pfd(i, j):
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key not in pfd_cache:
            steps, _, _, _ = data.compute_pfd(a, b)
            pfd_cache[key] = float(steps)
        return pfd_cache[key]

    # ---------------------------------------------------
    # 1) 샘플링
    # ---------------------------------------------------
    if sample_size is None or sample_size >= n:
        idx_list = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(center_seed)
        idx_list = rng.choice(n, size=sample_size, replace=False)

    m = len(idx_list)
    rng = np.random.default_rng(center_seed)

    # ---------------------------------------------------
    # 2) center 선택 (샘플 안에서)
    # ---------------------------------------------------
    if use_random_center:
        center_pos = int(rng.integers(0, m))
    else:
        center_pos = 0
    center_idx = int(idx_list[center_pos])  # 원본 인덱스

    # ---------------------------------------------------
    # 3) center 기준 거리 배열 (샘플에 대해서만): d[pos] = pfd(center, idx_list[pos])
    # ---------------------------------------------------
    dist_to_center = np.zeros(m, dtype=float)
    for pos in range(m):
        global_i = int(idx_list[pos])
        dist_to_center[pos] = get_pfd(center_idx, global_i)

    nu_A = float(dist_to_center.sum())
    R = max(nu_A / max(1, m), 1e-12)
    Dmax = float(dist_to_center.max())

    if Dmax < R:
        return np.array([center_idx], dtype=int), np.array([n], dtype=int)

    # ---------------------------------------------------
    # 4) ring + cake(wedge via anchors) 버킷팅
    # ---------------------------------------------------
    from math import ceil, log2

    M = max(1, ceil(log2((Dmax + R) / R)))
    tiny = 1e-12

    # ring index for each sampled position
    ring_of = np.zeros(m, dtype=int)
    for pos in range(m):
        dist = float(dist_to_center[pos])
        if dist < R:
            j = 0
        else:
            j = int(math.ceil(math.log2(max(dist, tiny) / R)))
        ring_of[pos] = min(j, M)

    # buckets: key -> [rep_global_idx, count_in_sample]
    # key = (ring j, wedge_id) or (ring j, wedge_id, radial_cell)
    buckets = {}

    # helper: inter-PFD between two sampled positions
    def d_pos(p, q) -> float:
        return get_pfd(int(idx_list[p]), int(idx_list[q]))

    for j in range(0, M + 1):
        # positions in this ring
        I = np.where(ring_of == j)[0]
        if len(I) == 0:
            continue

        # tau: wedge(조각) 반경 목표(커질수록 조각 굵어짐)
        scale = R * (2 ** j)
        tau = (eps * scale) / float(angle_beta)
        tau = max(tau, (eps * R) / float(alpha_min))

        # --- farthest-point sampling anchors in this ring ---
        # start anchor = farthest from center within ring
        start = int(I[np.argmax(dist_to_center[I])])
        anchors = [start]

        # maintain best distance to chosen anchors & its anchor-id (wedge assignment)
        best_dist = np.array([d_pos(pos, start) for pos in I], dtype=float)
        best_aid  = np.zeros(len(I), dtype=int)

        # anchor itself distance should be 0 if included
        for t, pos in enumerate(I):
            if pos == start:
                best_dist[t] = 0.0

        while True:
            if len(anchors) >= max_anchors_per_ring:
                break
            # farthest uncovered point
            t_far = int(np.argmax(best_dist))
            farthest_pos = int(I[t_far])
            farthest_md = float(best_dist[t_far])

            if farthest_md <= tau:
                break

            anchors.append(farthest_pos)
            new_aid = len(anchors) - 1

            # update best_dist / best_aid with new anchor
            for t, pos in enumerate(I):
                dnew = 0.0 if pos == farthest_pos else d_pos(pos, farthest_pos)
                if dnew < best_dist[t]:
                    best_dist[t] = dnew
                    best_aid[t] = new_aid

        # optional: radial cell size (기존 1D 그리드 폭)
        if use_radial_cell:
            r_rad = (eps * R * (2 ** j)) / float(alpha)
            r_rad = max(r_rad, (eps * R) / float(alpha_min))
        else:
            r_rad = None

        # --- fill buckets ---
        for t, pos in enumerate(I):
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

        S_idx = np.array(reps, dtype=int)
        S_weights = np.array(wts, dtype=int)
    else:
        S_idx = np.array([center_idx], dtype=int)
        S_weights = np.array([n], dtype=int)

    return S_idx, S_weights


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
    eps: float = 0.4,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    sample_size: int = 128,
    center_seed: int = 0,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_paths)} json files in {input_dir}")

    for json_file in json_paths:
        print(f"\n=== Processing {json_file.name} ===")
        start_time = time.time()  # ★ 시작 시간

        # (1) Data 로딩 + 코어셋 생성
        data_full = Data(str(json_file))
        S_idx, S_weights = build_triangulation_coreset_practical(
            data_full,
            eps=eps,
            alpha=alpha,
            alpha_min=alpha_min,
            use_random_center=True,
            center_seed=center_seed,
            sample_size=sample_size,
        )
        print(f"  -> coreset reps: {len(S_idx)} / approx weights sum: {int(S_weights.sum())}")

        # (2) 원본 인스턴스 json 읽기
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
            # "coreset_weights": np.asarray(S_weights, dtype=int).tolist(),
        }

        out_name = json_file.stem + "_coreset.json"
        out_file = output_path / out_name

        with open(out_file, "w") as f:
            json.dump(coreset_inst, f, indent=2)

        elapsed = time.time() - start_time  # ★ 경과 시간
        print(f"  -> saved to: {out_file}")
        print(f"  -> time taken: {elapsed:.3f} seconds")  # ★ 시간 출력


if __name__ == "__main__":
    build_and_save_coresets_for_all_instances(
        input_dir="./data/benchmark_instances",
        output_dir="./data/coreset_instance",
        eps=0.5,
        alpha=8.0,
        alpha_min=16.0,
        sample_size=32,   # 더 빠르게: 64, 32 등으로 조절
        center_seed=0,
    )