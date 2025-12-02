import copy
import math
import time   # ★ 추가
import numpy as np
from pathlib import Path
import json
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
    data.triangulations 위에서 compute_pfd를 거리로 사용하는
    1-median coreset 생성기 (속도 극단적으로 줄인 버전).

    아이디어:
      - triangulation 전체 n개 중에서 sample_size개만 랜덤 샘플링해서
        그 샘플들에 대해서만 거리 기반 코어셋을 만든다.
      - center도 이 샘플들 중 하나로 선택한다.
      - compute_pfd 호출 횟수 ~ O(sample_size) 로 제한.

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
        print(f"-> coreset reps: {len(S_idx)} / approx weights sum: {int(S_weights.sum())}")

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
            "coreset_weights": np.asarray(S_weights, dtype=int).tolist(),
        }

        out_name = json_file.stem + "_coreset.json"
        out_file = output_path / out_name

        with open(out_file, "w") as f:
            json.dump(coreset_inst, f, indent=2)

        elapsed = time.time() - start_time  # ★ 경과 시간
        print(f"-> saved to: {out_file}")
        print(f"-> time taken: {elapsed:.3f} seconds")  # ★ 시간 출력


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
