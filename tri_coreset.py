import copy
import math
import numpy as np
from data2 import Data

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
):
    """
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
    # 1. 샘플링: 전체 n개 중에서 sample_size개만 뽑아서 coreset 빌드 대상으로 사용
    # ---------------------------------------------------
    if sample_size is None or sample_size >= n:
        idx_list = np.arange(n, dtype=int)  # 전체 사용
    else:
        rng = np.random.default_rng(center_seed)
        idx_list = rng.choice(n, size=sample_size, replace=False)
    m = len(idx_list)

    # RNG 하나 더 쓰고 싶지 않으면 위에서 만든 rng 재사용
    rng = np.random.default_rng(center_seed)

    # ---------------------------------------------------
    # 2. center triangulation 하나 선택 (샘플 안에서만 선택)
    # ---------------------------------------------------
    if use_random_center:
        center_pos = int(rng.integers(0, m))
    else:
        center_pos = 0
    center_idx = int(idx_list[center_pos])  # 원본 인덱스

    # ---------------------------------------------------
    # 3. center_idx 기준 거리 배열 d(i) = pfd(center_idx, idx_list[i])
    #    -> compute_pfd 호출 횟수: O(m) (m = 샘플 크기)
    # ---------------------------------------------------
    nearest_dist = np.zeros(m, dtype=float)
    for pos in range(m):
        global_i = int(idx_list[pos])
        nearest_dist[pos] = get_pfd(center_idx, global_i)

    nu_A = float(nearest_dist.sum())
    R = max(nu_A / max(1, m), 1e-12)
    Dmax = float(nearest_dist.max())

    # 샘플들 기준으로도 center에 너무 다 붙어 있으면 → center 하나로 요약
    if Dmax < R:
        # 이때 weight는 전체 n개를 대표한다고 보는게 자연스러움
        return np.array([center_idx], dtype=int), np.array([n], dtype=int)

    # ---------------------------------------------------
    # 4. 거리 스케일(ring) + 1D grid로 버킷팅 (샘플에 대해서만)
    # ---------------------------------------------------
    from math import ceil, log2
    M = max(1, ceil(log2((Dmax + R) / R)))

    buckets = {}
    tiny = 1e-12

    for pos in range(m):
        dist = nearest_dist[pos]
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
            # 대표는 샘플의 이 위치에 해당하는 원본 인덱스로
            buckets[key] = [int(idx_list[pos]), 1]  # (대표 index, sample 내 count)
        else:
            buckets[key][1] += 1

    # ---------------------------------------------------
    # 5. 버킷에서 대표 인덱스, 가중치 추출
    #    - count는 샘플에서의 개수이므로, 전체 n개에 대한 근사 weight로 scale
    # ---------------------------------------------------
    if buckets:
        reps = []
        wts = []
        scale = float(n) / float(m)  # 샘플에서 전체로 확장하는 비율

        for rep_idx, cnt in buckets.values():
            reps.append(rep_idx)
            # 샘플에서 cnt개면, 전체에서는 대략 cnt * (n/m) 개라고 근사
            approx_weight = max(1, int(round(cnt * scale)))
            wts.append(approx_weight)

        S_idx = np.array(reps, dtype=int)
        S_weights = np.array(wts, dtype=int)
    else:
        # 이론상 거의 안 나오는 경우지만, 방어적으로
        S_idx = np.array([center_idx], dtype=int)
        S_weights = np.array([n], dtype=int)

    return S_idx, S_weights

# -------------------------------------------------------
# 2) 코어셋 triangulation 들만 가지고 find_center 실행
# -------------------------------------------------------

def make_coreset_data(data_full: Data, S_idx, S_weights):
    """
    data_full과 같은 pts / geometry 정보를 공유하지만,
    triangulations 리스트만 코어셋으로 교체한 Data 객체를 만든다.
    tri_weights는 coreset weight(S_weights)로 설정.
    """
    coreset_data = copy.copy(data_full)  # 얕은 복사: pts, etc. 공유

    # 코어셋 triangulation만 복사해서 넣기
    new_tris = [copy.deepcopy(data_full.triangulations[i]) for i in S_idx]
    coreset_data.triangulations = new_tris

    # center / flip / dist 초기화
    coreset_data.center = coreset_data.triangulations[0]
    coreset_data.flip = [[] for _ in coreset_data.triangulations]
    coreset_data.dist = float("INF")

    # ★ coreset weight 설정 (Data.find_center_np에서 자동 사용)
    coreset_data.tri_weights = np.asarray(S_weights, dtype=np.int64)

    return coreset_data


# -------------------------------------------------------
# 3) full vs coreset에서 find_center 결과 비교
# -------------------------------------------------------

def compare_center_with_coreset(
    instance_path: str = "random_instance_440_160_20.json",
    eps: float = 0.4,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    sample_size: int = 128,
):
    # (A) 원본 인스턴스 로딩
    data_full = Data(instance_path)

    # (C) 1-median coreset 생성 (compute_pfd 기반, 샘플링 버전)
    print("\n[1] 1-median coreset 생성 중...")
    S_idx, S_weights = build_triangulation_coreset_practical(
        data_full,
        eps=eps,
        alpha=alpha,
        alpha_min=alpha_min,
        use_random_center=True,
        center_seed=0,
        sample_size=sample_size,
    )
    print(f" coreset reps: {len(S_idx)} / approx weights 합: {S_weights.sum()}")

    # (D) 코어셋만 가지고 continuous center (find_center_np)
    print("\n[2] coreset 데이터에서 center 찾는 중...")
    data_core = make_coreset_data(data_full, S_idx, S_weights)
    center_core, dist_core_on_core = data_core.find_center_np()
    print(f" center_dist (coreset 데이터 기준): {dist_core_on_core}")

    # data_core.WriteData()
    # (E) coreset center를 full 데이터에서 평가
    print("\n[3] coreset center를 full 데이터에서 평가...")
    dist_core_on_full, _a = data_full.compute_center_dist(center_core)
    data_full.center = center_core
    data_full.dist = dist_core_on_full
    data_full.flip = _a
    data_full.WriteData()

    print(f"coreset center_dist (full 기준):   {dist_core_on_full}")


if __name__ == "__main__":
    # 인스턴스 경로 / sample_size는 상황에 맞게 조절
    compare_center_with_coreset(
        instance_path="./data/benchmark_instances/rirs-5000-75-9bd3bd51.json",
        eps=0.5,
        alpha=8.0,
        alpha_min=16.0,
        sample_size=32,   # 더 빠르게 하고 싶으면 64, 32 이런 식으로 줄여도 됨
    )
