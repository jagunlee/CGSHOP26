import time
import csv
import math
import json
from pathlib import Path

from data import Data

def load_instance_best_costs_from_columns(result_csv: str):
    """
    result.csv 형식 (가정):

        (레이블/날짜), inst1,        inst2,        inst3, ...
        algo1/날짜,   cost11,      cost12,       cost13, ...
        algo2,        cost21,      cost22,       cost23, ...
        ...

    -> 첫 행의 각 열 이름(inst1, inst2, ...)을 인스턴스 이름으로 사용하고,
       해당 열의 최소값을 그 인스턴스의 best cost 로 본다.
    """
    path = Path(result_csv)
    best_costs = {}

    if not path.exists():
        print(f"[WARN] result.csv not found at {path}")
        return best_costs

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return best_costs

        inst_names = [h.strip() for h in header[1:]]
        n_inst = len(inst_names)
        best_vals = [math.inf] * n_inst

        for row in reader:
            if not row:
                continue
            for j in range(1, min(len(row), n_inst + 1)):
                val_str = row[j].strip()
                if not val_str:
                    continue
                try:
                    v = float(val_str)
                except ValueError:
                    continue
                if v < best_vals[j - 1]:
                    best_vals[j - 1] = v

        for name, v in zip(inst_names, best_vals):
            if math.isfinite(v):
                best_costs[name] = v

    return best_costs


def read_coreset_weights(coreset_json_path: str, n: int):
    """
    coreset json (두번째 파일 형식)에서 coreset_weights 읽기.
    없거나 길이가 다르면 안전하게 [1]*n 로 보정.
    """
    weights = None
    try:
        with open(coreset_json_path, "r", encoding="utf-8") as f:
            root = json.load(f)
        if "coreset_weights" in root:
            weights = [float(x) for x in root["coreset_weights"]]
    except Exception:
        weights = None

    if weights is None:
        return [1.0] * n

    if len(weights) < n:
        weights = weights + [1.0] * (n - len(weights))
    elif len(weights) > n:
        weights = weights[:n]
    return weights


def solve_all_from_coresets(
    benchmark_dir: str = "./data/benchmark_instances",
    coreset_dir: str = "./data/coreset_instance",
    result_csv: str = "./result.csv",
    log_path: str = "./logs/solve_from_coreset_findCenterGlobal2.log",
):
    """
    coreset_dir 안의 *_coreset.json 에 대해:

      (1) coreset 인스턴스 Data(data_core)에서 data.py 의 findCenterGlobal2() 실행 (weights 반영)
      (2) centerT로부터 coreset까지의 weighted sum(Σ w_i * dist_i)도 계산해 로그 저장
      (3) 얻은 centerT 를 full 인스턴스 Data(data_full)에 대해 computeDistanceSum(centerT) 으로 평가 (full 기준 cost)
      (4) data_full.center = centerT 로 맞춰준 뒤 data_full.WriteData() 저장
      (5) result.csv 에 있는 해당 인스턴스 best cost 와 비교해서 .log 기록
    """

    benchmark_path = Path(benchmark_dir)
    coreset_path = Path(coreset_dir)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    inst_best = load_instance_best_costs_from_columns(result_csv)
    print(f"[INFO] Loaded {len(inst_best)} instance best costs from {result_csv}")

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("========== solve_from_coreset (findCenterGlobal2 + coreset weighted sum) ==========\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"benchmark_dir = {benchmark_dir}\n")
        lf.write(f"coreset_dir   = {coreset_dir}\n")
        lf.write(f"result_csv    = {result_csv}\n\n")

    coreset_files = sorted(coreset_path.glob("*_coreset.json"))
    print(f"[INFO] Found {len(coreset_files)} coreset json files in {coreset_dir}")

    # ✅ rirs 스킵 함수
    def is_rirs(name: str) -> bool:
        return str(name).lower().startswith("rirs") or str(name).lower().startswith("rirs")

    for co_file in coreset_files:
        print("\n=======================================================")
        print(f"[CORESET] {co_file.name}")

        stem = co_file.stem
        base_stem = stem[:-len("_coreset")] if stem.endswith("_coreset") else stem

        # ✅ (1) 파일명 기준으로 rirs 스킵 (빠름)
        if is_rirs(base_stem):
            print(f"  [SKIP] base_stem starts with 'rirs': {base_stem}")
            continue

        full_file = benchmark_path / f"{base_stem}.json"
        if not full_file.exists():
            print(f"  [WARN] full instance not found: {full_file} -> skip")
            continue

        # (A) coreset / full 로딩
        data_core = Data(str(co_file))
        data_full = Data(str(full_file))

        inst_uid = getattr(data_full, "instance_uid", base_stem)

        # ✅ (2) instance_uid 기준으로도 rirs 스킵 (더 정확)
        if is_rirs(inst_uid):
            print(f"  [SKIP] instance_uid starts with 'rirs': {inst_uid}")
            continue

        # coreset weights 로딩
        weights = read_coreset_weights(str(co_file), len(data_core.triangulations))
        w_sum = float(sum(weights))

        # (B) ✅ coreset Data 에서 center 찾기: findCenterGlobal2()
        t0 = time.time()
        centerT = data_core.findCenterGlobal2()
        t1 = time.time()
        find_center_time = t1 - t0
        print(f"      findCenterGlobal2 done. time = {find_center_time:.3f} s")

        # (B-2) ✅ centerT 기준 coreset weighted sum 계산 (Σ w_i * dist_i)
        t_core0 = time.time()
        data_core.computeDistanceSum(centerT)  # data_core.pFlips 채움
        t_core1 = time.time()

        # dist_i = len(pFlips[i]) (parallel flip rounds)
        if getattr(data_core, "pFlips", None) is None:
            coreset_unweighted = math.inf
            coreset_weighted = math.inf
        else:
            coreset_unweighted = float(sum(len(p) for p in data_core.pFlips))
            coreset_weighted = float(
                sum(weights[i] * len(data_core.pFlips[i]) for i in range(len(data_core.pFlips)))
            )

        core_eval_time = t_core1 - t_core0
        print(f"      coreset dist sum (unweighted) : {coreset_unweighted}")
        print(f"      coreset dist sum (weighted)   : {coreset_weighted}   (sum_w={w_sum})")
        print(f"      time for coreset computeDistanceSum: {core_eval_time:.3f} s")

        # (C) full Data 에서 computeDistanceSum(centerT)
        t2 = time.time()
        data_full.computeDistanceSum(centerT)
        t3 = time.time()
        eval_time = t3 - t2

        if getattr(data_full, "pFlips", None) is None:
            coreset_dist_on_full = math.inf
        else:
            coreset_dist_on_full = float(sum(len(pFlip) for pFlip in data_full.pFlips))

        print(f"      center_dist on FULL (dist sum): {coreset_dist_on_full}")
        print(f"      time for FULL computeDistanceSum: {eval_time:.3f} s")

        # ✅ WriteData()가 meta에 self.center를 쓰는 경우가 많아서 center를 맞춰줌
        try:
            data_full.center = centerT.fast_copy()
        except Exception:
            data_full.center = centerT

        # (D) 저장
        data_full.WriteData()

        # (E) result.csv 기준 best cost 와 비교
        ref_cost = inst_best.get(inst_uid) or inst_best.get(base_stem)

        diff = ratio = improvement = None
        comparison = "N/A"

        if ref_cost is not None:
            diff = coreset_dist_on_full - ref_cost
            improvement = ref_cost - coreset_dist_on_full
            ratio = (coreset_dist_on_full / ref_cost) if ref_cost != 0 else math.inf

            if coreset_dist_on_full < ref_cost:
                comparison = "better_than_result_csv"
            elif coreset_dist_on_full > ref_cost:
                comparison = "worse_than_result_csv"
            else:
                comparison = "equal_to_result_csv"

            print(f"      best cost in result.csv for [{inst_uid}]: {ref_cost}")
            print(f"      diff(ours - best)            = {diff}")
            print(f"      improvement(best - ours)     = {improvement}")
            print(f"      ratio(ours / best)           = {ratio:.6f}")
            print(f"      comparison                   = {comparison}")
        else:
            print(f"      [WARN] No header column matching '{inst_uid}' or '{base_stem}' in result.csv")

        # (F) 로그 기록 (✅ coreset weighted sum 추가)
        lines = []
        lines.append("-------------------------------------------------------")
        lines.append(f"[{co_file.name}]")
        lines.append(f"instance_uid                 = {inst_uid}")
        lines.append(f"center_dist_on_full          = {coreset_dist_on_full:.6f}")
        lines.append(f"coreset_dist_sum_unweighted  = {coreset_unweighted:.6f}")
        lines.append(f"coreset_dist_sum_weighted    = {coreset_weighted:.6f}")
        lines.append(f"coreset_weights_sum          = {w_sum:.6f}")
        lines.append(f"findCenterGlobal2_time       = {find_center_time:.3f}s")
        lines.append(f"coreset_computeDistSum_time  = {core_eval_time:.3f}s")
        lines.append(f"full_computeDistanceSum_time = {eval_time:.3f}s")

        if ref_cost is not None:
            lines.append(f"best_cost(result.csv)        = {ref_cost:.6f}")
            lines.append(f"diff(ours - best)            = {diff:.6f}")
            lines.append(f"improvement(best - ours)     = {improvement:.6f}")
            lines.append(f"ratio(ours / best)           = {ratio:.6f}")
            lines.append(f"comparison                   = {comparison}")
        else:
            lines.append("best_cost(result.csv)        = None (no matching header column)")

        lines.append("-------------------------------------------------------")

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + "\n")

        print("=======================================================")

    print(f"\n[LOG] Saved results to {log_file}")


if __name__ == "__main__":
    solve_all_from_coresets(
        benchmark_dir="../data/benchmark_instances",
        coreset_dir="../data/coreset_instance_260123",
        result_csv="../result.csv",
        log_path="../logs/solve_from_coreset_260123.log",
    )
