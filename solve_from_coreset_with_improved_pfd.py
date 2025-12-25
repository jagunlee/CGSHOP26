import time
import csv
import math
from pathlib import Path

from data import Data  # ✅ data2 말고 data.py 의 Data 사용


def load_instance_best_costs_from_columns(result_csv: str):
    """
    result.csv 형식 (가정):

        (레이블/날짜), inst1,        inst2,        inst3, ...
        algo1/날짜,   cost11,      cost12,       cost13, ...
        algo2,        cost21,      cost22,       cost23, ...
        ...

    -> 첫 행의 각 열 이름(inst1, inst2, ...)을 인스턴스 이름으로 사용하고,
       해당 열의 최소값을 그 인스턴스의 best cost 로 본다.

    반환:
      dict: { "instance_name": best_cost }
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


def solve_all_from_coresets(
    benchmark_dir: str = "./data/benchmark_instances",
    coreset_dir: str = "./data/coreset_instance",
    result_csv: str = "./result.csv",
    log_path: str = "./solve_from_coreset_findCenter.log",
):
    """
    coreset_dir 안에 있는 *_coreset.json 에 대해:

      (1) coreset 인스턴스 Data(data_core)에서 data.py 의 findCenter() 실행
      (2) 얻은 centerT 를 full 인스턴스 Data(data_full)에 대해
          computeDistanceSum(centerT) 으로 평가 (full 기준 cost)
      (3) data_full.WriteData() 로 솔루션/결과 저장 (data.py 의 WriteData 그대로 사용)
      (4) result.csv 에 있는 해당 인스턴스의 best cost 와 비교해서 .log 파일에 기록

    주의:
      - result.csv 는 스크립트 시작 시 한 번만 읽어서
        "이전 best" 를 고정해두고, 그 기준으로 improvement 를 계산.
      - Data.WriteData() 가 내부적으로 result.csv 를 업데이트하지만,
        그건 이후 run 에서 반영될 뿐, 이번 비교에는 초기값만 사용.
    """

    benchmark_path = Path(benchmark_dir)
    coreset_path = Path(coreset_dir)
    log_file = Path(log_path)

    # --- result.csv 에서 각 인스턴스별 best cost 읽기 ---
    inst_best = load_instance_best_costs_from_columns(result_csv)
    print(f"[INFO] Loaded {len(inst_best)} instance best costs from {result_csv}")

    # --- 로그 헤더 ---
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("========== solve_from_coreset (findCenter / computeDistanceSum) ==========\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"benchmark_dir = {benchmark_dir}\n")
        lf.write(f"coreset_dir   = {coreset_dir}\n")
        lf.write(f"result_csv    = {result_csv}\n\n")

    coreset_files = sorted(coreset_path.glob("*_coreset.json"))
    print(f"[INFO] Found {len(coreset_files)} coreset json files in {coreset_dir}")

    for co_file in coreset_files:
        print("\n=======================================================")
        print(f"[CORESET] {co_file.name}")

        # 원본 instance 파일 이름 찾기
        stem = co_file.stem  # e.g., random_instance_995_320_10_coreset
        if stem.endswith("_coreset"):
            base_stem = stem[:-len("_coreset")]  # random_instance_995_320_10
        else:
            base_stem = stem

        full_file = benchmark_path / f"{base_stem}.json"
        if not full_file.exists():
            print(f"  [WARN] full instance not found: {full_file} -> skip")
            continue

        # (A) coreset / full Data 로딩 (data.py 의 Data)
        data_core = Data(str(co_file))
        data_full = Data(str(full_file))

        # result.csv 에서 사용할 인스턴스 이름 (instance_uid 기준 가정)
        inst_uid = getattr(data_full, "instance_uid", base_stem)

        # (B) coreset Data 에서 center 찾기: findCenter()
        t0 = time.time()
        centerT = data_core.findCenter()
        t1 = time.time()
        find_center_time = t1 - t0

        print(f"      findCenter done. time = {find_center_time:.3f} s")

        # (C) full Data 에서 computeDistanceSum(centerT) 실행
        t2 = time.time()
        data_full.computeDistanceSum(centerT)
        t3 = time.time()
        eval_time = t3 - t2

        # cost = full 데이터에서 centerT까지의 parallel flip distance 합
        # (각 triangulation마다 pFlips[i] 길이를 더함)
        if getattr(data_full, "pFlips", None) is None:
            coreset_dist_on_full = math.inf
        else:
            coreset_dist_on_full = float(sum(len(pFlip) for pFlip in data_full.pFlips))

        print(f"      coreset center_dist (full 기준, dist sum): {coreset_dist_on_full}")
        print(f"      time for computeDistanceSum:             {eval_time:.3f} s")

        # (D) data.py 의 WriteData 사용 (solution + result.csv update)
        data_full.WriteData()

        # (E) result.csv 기준 best cost 와 비교 (스크립트 시작 시점 기준)
        ref_cost = inst_best.get(inst_uid)
        if ref_cost is None:
            ref_cost = inst_best.get(base_stem)

        diff = None
        ratio = None
        comparison = "N/A"
        improvement = None

        if ref_cost is not None:
            diff = coreset_dist_on_full - ref_cost          # 양수면 현재가 더 나쁨
            improvement = ref_cost - coreset_dist_on_full   # 양수면 개선
            ratio = coreset_dist_on_full / ref_cost if ref_cost != 0 else math.inf

            if coreset_dist_on_full < ref_cost:
                comparison = "better_than_result_csv"
            elif coreset_dist_on_full > ref_cost:
                comparison = "worse_than_result_csv"
            else:
                comparison = "equal_to_result_csv"

            print(f"      best cost in result.csv for [{inst_uid}]: {ref_cost}")
            print(f"      diff(coreset - best)      = {diff}")
            print(f"      improvement(best - coreset)= {improvement}")
            print(f"      ratio(coreset / best)     = {ratio:.6f}")
            print(f"      comparison                = {comparison}")
        else:
            print(f"      [WARN] No header column matching '{inst_uid}' or '{base_stem}' in result.csv")

        # (F) .log 파일에 기록
        lines = []
        lines.append("-------------------------------------------------------")
        lines.append(f"[{co_file.name}]")
        lines.append(f"instance_uid            = {inst_uid}")
        lines.append(f"coreset_dist_on_full    = {coreset_dist_on_full:.6f}")
        lines.append(f"findCenter_time         = {find_center_time:.3f}s")
        lines.append(f"computeDistanceSum_time = {eval_time:.3f}s")

        if ref_cost is not None:
            lines.append(f"best_cost(result.csv)   = {ref_cost:.6f}")
            lines.append(f"diff(coreset - best)    = {diff:.6f}")
            lines.append(f"improvement(best - coreset) = {improvement:.6f}")
            lines.append(f"ratio(coreset / best)   = {ratio:.6f}")
            lines.append(f"comparison              = {comparison}")
        else:
            lines.append("best_cost(result.csv)   = None (no matching header column)")

        lines.append("-------------------------------------------------------")

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + "\n")

        print("=======================================================")

    print(f"\n[LOG] Saved results to {log_file}")


if __name__ == "__main__":
    solve_all_from_coresets(
        benchmark_dir="./data/benchmark_instances",
        coreset_dir="./data/coreset_instance",
        result_csv="./result.csv",
        log_path="./logs/solve_from_coreset_findCenter.log",
    )
