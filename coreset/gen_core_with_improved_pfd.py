import json
import time
from pathlib import Path

import numpy as np
from data import Data  # parallel_flip_path, triangulations 등을 쓰는 Data 클래스

# -------------------------------------------------------
# 1) triangulation + parallel_flip_path 기반 비랜덤 1-median coreset
# -------------------------------------------------------

def build_triangulation_coreset_practical(
    data,
    eps: float = 0.1,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    center_candidates: int = 5,
):
    """
    data.triangulations 위에서 parallel_flip_path 를 거리로 사용하는
    1-median coreset 생성기 (비랜덤 버전).

    - 모든 triangulation을 다 사용 (샘플링 X)
    - center 후보는 앞에서부터 center_candidates개
      (0,1,2,... 순서라 완전히 deterministic)
    - 각 bucket에서 대표 triangulation 1개 + 해당 bucket 크기(weight)를 반환

    distance 정의:
      dist(i, j) = len( parallel_flip_path( T_i, T_j ) )
      (parallel flip round 의 개수 = pfd)

    반환:
      S_idx      : 대표 triangulation 인덱스들 (np.ndarray[int])
      S_weights  : 각 대표가 대표하는 개수 (np.ndarray[int])
    """

    n = len(data.triangulations)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # --- parallel_flip 기반 pfd 캐시 ---
    pfd_cache = {}

    def get_pfd(i, j):
        """
        i, j : triangulation index
        거리 = len( parallel_flip_path( triangulations[i], triangulations[j] ) )
        """
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key not in pfd_cache:
            tri_a = data.triangulations[a]
            tri_b = data.triangulations[b]
            # parallel flip path: list of rounds, 각 round 는 여러 flip edge 집합
            pfp = data.parallel_flip_path(tri_a, tri_b)
            steps = len(pfp)  # parallel flip distance = round 수
            pfd_cache[key] = float(steps)
        return pfd_cache[key]

    # --- 1. discrete 1-median 근사용 center 후보 선택 (비랜덤) ---
    m = n
    cand_num = min(m, center_candidates)
    best_center_idx = 0
    best_total_dist = float("inf")

    # 후보: triangulation 0,1,2,...,cand_num-1
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
        inf_norm = dist  # 1D 값이라 그대로 사용

        # level j 결정 (R, 2R, 4R, ...)
        if inf_norm < R:
            j = 0
        else:
            j = int(np.ceil(np.log2(max(inf_norm, tiny) / R)))
        j = min(j, M)

        # level j에서의 cell 크기 rj
        rj = (eps * R * (2 ** j)) / alpha
        rj = max(rj, (eps * R) / alpha_min)

        # 1D grid cell index
        cell_1d = int(np.floor(inf_norm / rj))
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
# 2) 모든 인스턴스에 대해 코어셋 생성 후 저장 (비랜덤 버전)
# -------------------------------------------------------

def build_and_save_coresets_for_all_instances(
    input_dir: str = "./data/benchmark_instances",
    output_dir: str = "./data/coreset_instance",
    eps: float = 0.4,
    alpha: float = 8.0,
    alpha_min: float = 16.0,
    center_candidates: int = 5,
):
    """
    input_dir 안에 있는 모든 *.json 인스턴스에 대해
    triangulation 1-median 코어셋(비랜덤 버전)을 만들고,
    output_dir 안에 예시 인스턴스와 동일한 포맷의 json으로 저장.

    저장 포맷:
      {
        "content_type": "CGSHOP2026_Instance",
        "instance_uid": "<원래 instance_uid>-coreset",
        "points_x": [...],
        "points_y": [...],
        "triangulations": [  # 코어셋 triangulations (subset)
            [[u, v], [u2, v2], ...],
            ...
        ],
        "coreset_weights": [w0, w1, ...]  # 각 triangulation weight
      }
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_paths)} json files in {input_dir}")

    for json_file in json_paths:
        print("\n==============================================")
        print(f"Processing {json_file.name}")

        start_time = time.time()

        # (1) Data 객체로 로드해서 coreset 계산
        data_full = Data(str(json_file))

        S_idx, S_weights = build_triangulation_coreset_practical(
            data_full,
            eps=eps,
            alpha=alpha,
            alpha_min=alpha_min,
            center_candidates=center_candidates,
        )

        print(f"  -> coreset reps: {len(S_idx)} / weights sum: {int(S_weights.sum())}")

        # (2) 원래 인스턴스 json 읽기
        with open(json_file, "r", encoding="utf-8") as f:
            inst = json.load(f)

        points_x = inst["points_x"]
        points_y = inst["points_y"]
        tri_list = inst["triangulations"]

        S_idx_list = np.asarray(S_idx, dtype=int).tolist()
        coreset_tris = [tri_list[i] for i in S_idx_list]
        coreset_wts = np.asarray(S_weights, dtype=int).tolist()

        base_uid = inst.get("instance_uid", json_file.stem)
        coreset_uid = f"{base_uid}-coreset"

        # (3) coreset 인스턴스 딕셔너리 구성
        coreset_inst = {
            "content_type": inst.get("content_type", "CGSHOP2026_Instance"),
            "instance_uid": coreset_uid,
            "points_x": points_x,
            "points_y": points_y,
            "triangulations": coreset_tris,
            "coreset_weights": coreset_wts,
        }

        # (4) 파일명: 원래 이름 + "_coreset.json"
        out_name = json_file.stem + "_coreset.json"
        out_file = output_path / out_name

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(coreset_inst, f, indent=2)

        elapsed = time.time() - start_time
        print(f"  -> saved coreset instance to: {out_file}")
        print(f"  -> time taken: {elapsed:.3f} seconds")
        print("==============================================")


if __name__ == "__main__":
    build_and_save_coresets_for_all_instances(
        input_dir="./data/benchmark_instances",
        output_dir="./data/coreset_instance-251219",
        eps=0.1,
        alpha=8.0,
        alpha_min=16.0,
        center_candidates=0,  # 비랜덤 후보 개수 (0~4번 triangulation 기준)
    )
