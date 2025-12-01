import copy
import math
import numpy as np
from data import Data

# -------------------------------------------------------
# 1) triangulation + compute_pfd 거리 기반 1-median coreset
# -------------------------------------------------------

def build_triangulation_coreset_practical(
    data,
    eps: float = 0.1,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    center_candidates: int = 5,
):
    """
    data.triangulations 위에서 compute_pfd를 거리로 사용하는
    1-median coreset 생성기.

    반환:
      S_idx      : 대표 triangulation 인덱스들 (np.ndarray[int])
      S_weights  : 각 대표가 대표하는 개수 (np.ndarray[int])
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

    # --- 1. 대략적인 center 후보 하나 잡기 (discrete 1-median 근사) ---
    m = n
    cand_num = min(m, center_candidates)
    best_center_idx = 0
    best_total_dist = float("inf")

    for c in range(cand_num):
        total = 0.0
        for j in range(m):
            total += get_pfd(c, j)
        if total < best_total_dist:
            best_total_dist = total
            best_center_idx = c

    center_idx = best_center_idx

    # --- 2. center_idx 기준 거리 배열 d(i) = pfd(center_idx, i) ---
    nearest_dist = np.zeros(m, dtype=float)
    for i in range(m):
        nearest_dist[i] = get_pfd(center_idx, i)

    nu_A = float(nearest_dist.sum())
    R = max(nu_A / max(1, m), 1e-12)
    Dmax = float(nearest_dist.max())

    # 모든 triangulation이 center와 너무 가깝다면 -> 하나로 요약
    if Dmax < R:
        return np.array([center_idx], dtype=int), np.array([m], dtype=int)

    # --- 3. 거리 스케일(ring) + 1D grid로 버킷팅 ---
    from math import ceil, log2
    M = max(1, ceil(log2((Dmax + R) / R)))

    buckets = {}
    tiny = 1e-12

    for i in range(m):
        dist = nearest_dist[i]
        inf_norm = dist  # 1D라 그냥 dist

        # level j 결정 (R, 2R, 4R, ...)
        if inf_norm < R:
            j = 0
        else:
            j = int(math.ceil(math.log2(max(inf_norm, tiny) / R)))
        j = min(j, M)

        # level j에서의 cell 크기 rj
        rj = (eps * R * (2 ** j)) / alpha
        rj = max(rj, (eps * R) / alpha_min)

        # 1D grid cell index
        cell_1d = int(math.floor(inf_norm / rj))
        key = (j, cell_1d)

        # 같은 버킷에 들어오는 triangulation들을 하나로 묶음
        if key not in buckets:
            buckets[key] = [i, 1]  # 대표 index, count
        else:
            buckets[key][1] += 1

    if buckets:
        reps = []
        wts = []
        for rep_idx, cnt in buckets.values():
            reps.append(rep_idx)
            wts.append(cnt)
        S_idx = np.array(reps, dtype=int)
        S_weights = np.array(wts, dtype=int)
    else:
        # 이론상 거의 안 나오는 경우지만, 방어적으로
        S_idx = np.array([center_idx], dtype=int)
        S_weights = np.array([m], dtype=int)

    return S_idx, S_weights

# -------------------------------------------------------
# 2) 코어셋 triangulation 들만 가지고 find_center 실행
# -------------------------------------------------------

def make_coreset_data(data_full: Data, S_idx):
    """
    data_full과 같은 pts / geometry 정보를 공유하지만,
    triangulations 리스트만 코어셋으로 교체한 Data 객체를 만든다.
    (weights는 일단 find_center에서는 쓰지 않고, 평가할 때만 사용)
    """
    coreset_data = copy.copy(data_full)  # 얕은 복사: pts, etc. 공유

    # 코어셋 triangulation만 복사해서 넣기
    new_tris = [copy.deepcopy(data_full.triangulations[i]) for i in S_idx]
    coreset_data.triangulations = new_tris

    # center / flip / dist 초기화
    coreset_data.center = coreset_data.triangulations[0]
    coreset_data.flip = [[] for _ in coreset_data.triangulations]
    coreset_data.dist = float("inf")

    return coreset_data

# -------------------------------------------------------
# 3) full vs coreset에서 find_center 결과 비교
# -------------------------------------------------------

def compare_center_with_coreset(
    instance_path: str = "random_instance_440_160_20.json",
    eps: float = 0.4,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    center_candidates: int = 5,
):
    # (A) 원본 인스턴스 로딩
    data_full = Data(instance_path)
    # n_full = len(data_full.triangulations)
    # print(f"# full triangulations: {n_full}")

    # (B) full 데이터에서 continuous center (find_center)
    # print("\n[1] full 데이터에서 center 찾는 중...")
    # center_full, dist_full_on_full = data_full.find_center_np()
    # 안전하게 한 번 더 평가
    # dist_full_on_full_check, _ = data_full.compute_center_dist(center_full)
    # print(f" full center_dist (full 데이터 기준): {dist_full_on_full_check}")

    # (C) 1-median coreset 생성 (compute_pfd 기반)
    print("\n[2] 1-median coreset 생성 중...")
    S_idx, S_weights = build_triangulation_coreset_practical(
        data_full,
        eps=eps,
        alpha=alpha,
        alpha_min=alpha_min,
        center_candidates=center_candidates,
    )
    print(f" coreset reps: {len(S_idx)} / weights 합: {S_weights.sum()}")

    # (D) 코어셋만 가지고 continuous center (find_center)
    print("\n[3] coreset 데이터에서 center 찾는 중...")
    data_core = make_coreset_data(data_full, S_idx)
    center_core, dist_core_on_core = data_core.find_center_np()
    print(f" center_dist (coreset 데이터 기준): {dist_core_on_core}")

    # (E) coreset center를 full 데이터에서 평가
    print("\n[4] 두 center를 full 데이터에서 평가...")
    dist_core_on_full, _ = data_full.compute_center_dist(center_core)

    # 결과 요약
    # print("\n===== 결과 요약 =====")
    # print(f"full center_dist (full 기준):      {dist_full_on_full_check}")
    print(f"coreset center_dist (full 기준):   {dist_core_on_full}")
    # abs_diff = dist_core_on_full - dist_full_on_full_check
    # ratio = dist_core_on_full / dist_full_on_full_check if dist_full_on_full_check > 0 else float("nan")
    # print(f"절대 차이: {abs_diff}")
    # print(f"비율(coreset/full): {ratio:.6f}")
    # print(f"상대 오차(rel_error): {ratio - 1.0:.6f}")

if __name__ == "__main__":
    # 인스턴스 경로는 실제 위치에 맞게 수정해줘
    compare_center_with_coreset(
        instance_path="./data/benchmark_instances/rirs-1500-75-10c039f8.json",
        eps=0.4,
        alpha=8.0,
        alpha_min=16.0,
        center_candidates=5,
    )


# if __name__ == "__main__":
#     # 인스턴스 경로는 실제 위치에 맞게 수정해줘
#     compare_center_with_coreset(
#         instance_path="./data/benchmark_instances/rirs-500-75-9322678f.json",
#         eps=0.4,
#         alpha=8.0,
#         alpha_min=16.0,
#         center_candidates=5,
#     )

