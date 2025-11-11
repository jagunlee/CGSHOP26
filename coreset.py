import numpy as np

# ----- 비용 계산 유틸 -----
def cost_on_points(data_pts, center):
    """원본(무가중) 데이터에서의 비용 Σ‖x - c‖"""
    return float(np.linalg.norm(data_pts - center, axis=1).sum())

def weighted_cost_on_points(rep_pts, rep_w, center):
    """가중 데이터에서의 비용 Σ w_i‖s_i - c‖"""
    return float((rep_w * np.linalg.norm(rep_pts - center, axis=1)).sum())

# ----- discrete 1-median (후보 = C, 데이터 = D) -----
def discrete_1_median(candidates, data_pts, data_w=None, batch=0):
    """
    후보 집합 candidates 중에서, 데이터(data_pts, data_w)에 대한 합거리 비용을 최소화하는 후보를 반환.
    - data_w가 None이면 모두 가중치 1
    - 메모리 문제가 있으면 batch>0 으로 배치 계산 가능(후보를 배치로 나눠 처리)
    반환: (best_idx, best_center, best_cost_on_data)
    """
    C = candidates
    D = data_pts
    w = np.ones(len(D)) if data_w is None else data_w.astype(float)

    best_idx = -1
    best_cost = np.inf

    nC = len(C)
    if batch and batch < nC:
        # 배치 처리
        for start in range(0, nC, batch):
            end = min(start + batch, nC)
            Cb = C[start:end]                          # (B,d)
            # 거리: (B, |D|)
            dists = np.linalg.norm(Cb[:, None, :] - D[None, :, :], axis=2)
            costs = (dists * w[None, :]).sum(axis=1)   # (B,)
            j = np.argmin(costs)
            if costs[j] < best_cost:
                best_cost = float(costs[j])
                best_idx = start + int(j)
    else:
        # 전체 후보 한번에
        dists = np.linalg.norm(C[:, None, :] - D[None, :, :], axis=2)  # (|C|, |D|)
        costs = (dists * w[None, :]).sum(axis=1)                        # (|C|,)
        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])

    return best_idx, C[best_idx], best_cost

# ----- 비교 래퍼 -----
def compare_discrete_1median_with_coreset(P, S_points, S_weights, batch=0):
    """
    (1) 코어셋 기준 discrete 1-median: argmin_{p∈P} Σ_i w_i‖p - s_i‖
    (2) 원본 기준 discrete 1-median: argmin_{p∈P} Σ_x∈P ‖p - x‖
    을 각각 구하고, 두 중심을 원본 P에서의 비용으로 비교한다.
    반환 딕셔너리:
      {
        'center_coreset': ...,      # 코어셋으로 구한 중심 (P 중 하나)
        'center_original': ...,     # 원본으로 구한 중심 (P 중 하나)
        'cost_on_P_coreset': ...,   # center_coreset의 원본 비용
        'cost_on_P_original': ...,  # center_original의 원본 비용(최적)
        'abs_diff': ...,            # 비용 차이
        'ratio': ...,               # cost_on_P_coreset / cost_on_P_original
        'rel_error': ...            # (ratio - 1)
      }
    """
    # (1) 코어셋으로 구한 discrete 1-median (후보=P, 데이터=S)
    idx_c, c_core, _ = discrete_1_median(
        candidates=P, data_pts=S_points, data_w=S_weights, batch=batch
    )

    # (2) 원본으로 구한 discrete 1-median (후보=P, 데이터=P)
    idx_o, c_orig, _ = discrete_1_median(
        candidates=P, data_pts=P, data_w=None, batch=batch
    )

    # 두 중심을 원본 데이터에서 평가
    cost_core_on_P = cost_on_points(P, c_core)
    cost_orig_on_P = cost_on_points(P, c_orig)

    ratio = cost_core_on_P / cost_orig_on_P if cost_orig_on_P > 0 else np.nan
    result = {
        'center_coreset': c_core,
        'center_original': c_orig,
        'cost_on_P_coreset': cost_core_on_P,
        'cost_on_P_original': cost_orig_on_P,
        'abs_diff': cost_core_on_P - cost_orig_on_P,
        'ratio': ratio,
        'rel_error': ratio - 1.0 if np.isfinite(ratio) else np.nan,
        'idx_coreset_in_P': idx_c,
        'idx_original_in_P': idx_o,
    }
    return result

# ----- 사용 예시 -----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = np.vstack([
        rng.normal(0, 1.0, size=(600, 2)),
        rng.normal(10, 1.2, size=(600, 2)),
    ])

    # 당신이 만든 코어셋 생성기를 사용하세요.
    # (아래는 이전 답변의 practical 버전 예시)
    def geometric_median_weizfeld(P, iters=30, eps=1e-8):
        x = P[np.random.randint(len(P))].astype(float)
        for _ in range(iters):
            diff = P - x
            dist = np.linalg.norm(diff, axis=1)
            w = 1.0 / np.maximum(dist, eps)
            x_new = (w[:, None] * P).sum(axis=0) / w.sum()
            if np.linalg.norm(x_new - x) < 1e-7:
                break
            x = x_new
        return x

    def build_1median_coreset_practical(P, eps=0.1, alpha=8.0, alpha_min=16.0):
        n, d = P.shape
        if n == 0:
            return P.copy(), np.array([], dtype=int)
        x = geometric_median_weizfeld(P)
        nearest_dist = np.linalg.norm(P - x, axis=1)
        nu_A = float(nearest_dist.sum())
        R = max(nu_A / max(1, n), 1e-12)
        Dmax = float(np.max(np.linalg.norm(P - x, axis=1)))
        if Dmax < R:
            return x[None, :], np.array([n], dtype=int)
        from math import ceil, log2
        M = max(1, ceil(log2((Dmax + R) / R)))
        buckets = {}
        tiny = 1e-12
        for i in range(n):
            v = P[i] - x
            inf_norm = np.max(np.abs(v))
            j = 0 if inf_norm < R else int(np.ceil(np.log2(max(inf_norm, tiny) / R)))
            j = min(j, M)
            rj = (eps * R * (2 ** j)) / alpha
            rj = max(rj, (eps * R) / alpha_min)
            cell_idx = tuple(np.floor(v / rj).astype(int))
            key = (j, cell_idx)
            if key not in buckets:
                buckets[key] = [P[i].copy(), 1]
            else:
                buckets[key][1] += 1
        reps, wts = zip(*buckets.values()) if buckets else ([], [])
        S_points = np.array(reps) if reps else x[None, :].copy()
        S_weights = np.array(wts, dtype=int) if wts else np.array([n], dtype=int)
        return S_points, S_weights

    # 코어셋 만들기
    S_points, S_weights = build_1median_coreset_practical(P, eps=0.4, alpha=8, alpha_min=16)

    # 비교
    out = compare_discrete_1median_with_coreset(P, S_points, S_weights, batch=0)
    print(f"# P 크기: {len(P)}, 코어셋 크기: {len(S_points)}, 가중치합: {S_weights.sum()}")
    print(f"원본 기준 최적 비용: {out['cost_on_P_original']:.6f}")
    print(f"코어셋으로 구한 중심의 원본 비용: {out['cost_on_P_coreset']:.6f}")
    print(f"절대 차이: {out['abs_diff']:.6f}")
    print(f"비율(coreset/original): {out['ratio']:.6f}, 상대오차: {out['rel_error']:.6f}")
    # 선택된 인덱스(둘 다 P의 한 점)
    print(f"원본-최적 중심 index: {out['idx_original_in_P']}, 코어셋-유도 중심 index: {out['idx_coreset_in_P']}")
