import time
import csv
import math
from pathlib import Path

from data2 import Data  # 가중치 포함 find_center_np / ReadData 가 들어있는 버전


def load_instance_best_costs_from_columns(result_csv: str):
    """
    result.csv를 아래와 같은 형식으로 가정하고 읽는다.

        (빈칸/label), inst1, inst2, inst3, ...
        algo1,        c11,   c12,   c13,   ...
        algo2,        c21,   c22,   c23,   ...
        ...

    즉,
      - 첫 행: 각 열(column)의 인스턴스 이름
      - 첫 열: 알고리즘/실행 이름(무시)
      - (i>=1, j>=1) 셀: j번째 인스턴스에 대한 cost

    반환:
      dict: { "inst_name": 그 인스턴스 열의 최소 cost }
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

        # header[1:] 에 인스턴스 이름들이 있다고 가정
        inst_names = [h.strip() for h in header[1:]]
        n_inst = len(inst_names)
        best_vals = [math.inf] * n_inst

        for row in reader:
            if not row:
                continue
            # row[0] 은 알고리즘/실행 이름 → 무시
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

        # inf인 것은 값이 없었다는 뜻이므로 제외
        for name, v in zip(inst_names, best_vals):
            if math.isfinite(v):
                best_costs[name] = v

    return best_costs


def solve_all_from_coresets(
    benchmark_dir: str = "./data/benchmark_instances",
    coreset_dir: str = "./data/coreset_instance",
    result_csv: str = "./result.csv",
    log_path: str = "./solve_from_coreset.log",
):
    """
    coreset_dir 안에 있는 모든 *_coreset.json에 대해:

      (1) 코어셋 데이터(data_core)에서 weighted find_center_np 실행
      (2) 얻은 center를 full 인스턴스(data_full)에 대해 compute_center_dist로 평가
      (3) result.csv에서 해당 인스턴스 열(column)의 best cost를 찾아 비교
      (4) data_full.WriteData() 호출 + 비교 내용을 log 파일에 남김
    """

    benchmark_path = Path(benchmark_dir)
    coreset_path = Path(coreset_dir)
    log_file = Path(log_path)

    # --- result.csv에서 각 인스턴스별 best cost 읽기 (열 기준) ---
    inst_best = load_instance_best_costs_from_columns(result_csv)

    # --- 로그 헤더 기록 (append 모드) ---
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("========== solve_from_coreset run ==========\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"Loaded {len(inst_best)} instance best costs from {result_csv}\n\n")

    # --- 코어셋 파일들 탐색 ---
    coreset_files = sorted(coreset_path.glob("*_coreset.json"))
    print(f"Found {len(coreset_files)} coreset json files in {coreset_dir}")

    for co_file in coreset_files:
        print("\n=======================================================")
        print(f"[CORESET] {co_file.name}")

        # 원본 인스턴스 파일 이름 유추
        stem = co_file.stem
        base_stem = stem[:-len("_coreset")] if stem.endswith("_coreset") else stem
        full_file = benchmark_path / f"{base_stem}.json"
        if not full_file.exists():
            print(f"  [WARN] full instance not found: {full_file} -> skip")
            continue

        # (A) 데이터 로딩
        data_full = Data(str(full_file))
        data_core = Data(str(co_file))  # tri_weights = coreset_weights 로 읽힘

        # 인스턴스 이름: result.csv에서 어떤 이름으로 들어갔는지와 맞춰야 함
        # 보통 instance_uid 가 header의 이름일 가능성이 높다고 가정
        inst_uid = getattr(data_full, "instance_uid", base_stem)

        # (B) 코어셋 데이터에서 center 찾기
        t0 = time.time()
        center_core, dist_core_on_core = data_core.find_center_np()
        t1 = time.time()
        print(f"      center_dist (coreset 기준): {dist_core_on_core}")
        print(f"      time for find_center_np:   {t1 - t0:.3f} s")

        # (C) coreset center를 full 데이터에서 평가
        t2 = time.time()
        dist_core_on_full, flips = data_full.compute_center_dist(center_core)
        t3 = time.time()

        data_full.center = center_core
        data_full.dist = dist_core_on_full
        data_full.flip = flips
        data_full.WriteData()

        print(f"      coreset center_dist (full 기준): {dist_core_on_full}")
        print(f"      time for compute_center_dist:    {t3 - t2:.3f} s")
        print("=======================================================")

        # (D) result.csv의 해당 인스턴스 best cost와 비교
        ref_cost = inst_best.get(inst_uid)
        if ref_cost is None:
            # 혹시 header 에 instance_uid 대신 파일 stem이 들어갔을 수도 있으니 한 번 더 체크
            ref_cost = inst_best.get(base_stem)

        diff = None
        ratio = None
        comparison = "N/A"

        if ref_cost is not None:
            diff = dist_core_on_full - ref_cost
            ratio = dist_core_on_full / ref_cost if ref_cost != 0 else math.inf
            if dist_core_on_full < ref_cost:
                comparison = "better_than_result_csv"
            elif dist_core_on_full > ref_cost:
                comparison = "worse_than_result_csv"
            else:
                comparison = "equal_to_result_csv"

            print(f"      best cost in result.csv for [{inst_uid}]: {ref_cost}")
            print(f"      diff = {diff}, ratio = {ratio}, comparison = {comparison}")
        else:
            print(f"      [WARN] No header column matching '{inst_uid}' (or '{base_stem}') in result.csv")

        # (E) 로그 파일에 기록
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{co_file.name}]\n")
            lf.write(f"instance_uid            = {inst_uid}\n")
            lf.write(f"coreset_dist_on_full    = {dist_core_on_full:.6f}\n")
            lf.write(f"find_center_time        = {t1 - t0:.3f}s\n")
            lf.write(f"evaluate_time           = {t3 - t2:.3f}s\n")
            if ref_cost is not None:
                lf.write(f"best_cost(result.csv)   = {ref_cost:.6f}\n")
                lf.write(f"diff(coreset - best)    = {diff:.6f}\n")
                lf.write(f"ratio(coreset / best)   = {ratio:.6f}\n")
                lf.write(f"comparison              = {comparison}\n")
            else:
                lf.write("best_cost(result.csv)   = None (no matching header column)\n")
            lf.write("-------------------------------------------------------\n")

    print(f"\n[LOG] Saved results to {log_file}")


if __name__ == "__main__":
    solve_all_from_coresets(
        benchmark_dir="./data/benchmark_instances",
        coreset_dir="./data/coreset_instance",
        result_csv="./result.csv",
        log_path="./solve_from_coreset.log",
    )
