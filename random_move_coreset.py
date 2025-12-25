import os
import csv
import math
import time
from pathlib import Path

from data2 import Data  # tri_weights / compute_center_dist / random_move 포함 버전 사용


def load_instance_best_costs_from_columns(result_csv: str):
    """
    result.csv 형식:

        (label), inst1,        inst2,        inst3, ...
        algo1,   cost11,       cost12,       cost13, ...
        algo2,   cost21,       cost22,       cost23, ...
        ...

    각 인스턴스(열)별로 최소 cost를 읽어서
    { "inst_name": best_cost } 딕셔너리를 반환.
    """
    path = Path(result_csv)
    best_costs = {}

    if not path.exists():
        print(f"[WARN] result.csv not found at: {path}")
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


def process_one_coreset(
    co_file: Path,
    benchmark_dir: Path,
    inst_best: dict,
    log_file: Path,
):
    """
    하나의 coreset json 파일에 대해:

      1) coreset Data 에서 random_move() 실행
      2) coreset의 center를 full Data 의 compute_center_dist 로 평가해서
         full 기준 center_dist 를 얻음
      3) result.csv 의 best cost와 비교해서 로그 기록
    """
    print(f"\n[CORESET-RANDOM-MOVE] processing {co_file.name}")

    stem = co_file.stem
    # 예: random_instance_995_320_10_coreset → random_instance_995_320_10
    if stem.endswith("_coreset"):
        base_stem = stem[: -len("_coreset")]
    else:
        base_stem = stem

    full_file = benchmark_dir / f"{base_stem}.json"
    if not full_file.exists():
        print(f"  [WARN] full instance not found: {full_file} -> skip")
        return

    # (1) 코어셋 로드 후 random_move 실행
    data_core = Data(str(co_file))
    core_uid = getattr(data_core, "instance_uid", stem)

    t0 = time.time()
    data_core.random_move()
    t1 = time.time()
    random_move_time = t1 - t0

    # random_move 후 center 가 설정되어 있다고 가정
    center_core = getattr(data_core, "center", None)
    if center_core is None:
        print("  [WARN] data_core.center is None after random_move() -> skip")
        return

    # (2) full 데이터에서 center_core 를 평가
    data_full = Data(str(full_file))
    full_uid = getattr(data_full, "instance_uid", base_stem)

    t2 = time.time()
    dist_on_full, flips = data_full.compute_center_dist(center_core)
    t3 = time.time()
    eval_time = t3 - t2

    # full 인스턴스에 결과 저장 (선택사항)
    data_full.center = center_core
    data_full.dist = dist_on_full
    data_full.flip = flips
    data_full.WriteData()

    # (3) result.csv 의 best cost 와 비교
    ref_cost = inst_best.get(full_uid)
    if ref_cost is None:
        # 혹시 header 에 instance_uid 대신 파일 이름이 들어간 경우
        ref_cost = inst_best.get(base_stem)

    diff = None
    improv = None
    ratio = None
    comparison = "N/A"

    if ref_cost is not None:
        diff = dist_on_full - ref_cost               # 양수면 worse
        improv = ref_cost - dist_on_full             # 양수면 better
        ratio = dist_on_full / ref_cost if ref_cost != 0 else math.inf

        if dist_on_full < ref_cost:
            comparison = "better_than_result_csv"
        elif dist_on_full > ref_cost:
            comparison = "worse_than_result_csv"
        else:
            comparison = "equal_to_result_csv"

    # (4) 요약 로그 블록 작성
    lines = []
    lines.append("-------------------------------------------------------")
    lines.append(f"[{co_file.name}]")
    lines.append(f"instance_uid            = {full_uid}")
    lines.append(f"center_dist_on_full     = {dist_on_full:.6f}")
    lines.append(f"random_move_time        = {random_move_time:.3f}s")
    lines.append(f"evaluate_time           = {eval_time:.3f}s")

    if ref_cost is not None:
        lines.append(f"best_cost(result.csv)   = {ref_cost:.6f}")
        lines.append(f"delta(new - best)       = {diff:.6f}")
        lines.append(f"improvement(best - new) = {improv:.6f}")
        lines.append(f"ratio(new / best)       = {ratio:.6f}")
        lines.append(f"comparison              = {comparison}")
    else:
        lines.append("best_cost(result.csv)   = None (no matching header column)")

    lines.append("-------------------------------------------------------")
    summary_block = "\n".join(lines)

    # 콘솔 출력
    print(summary_block)

    # .log 파일에 append
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(summary_block + "\n")


def random_move_from_coresets_and_log(
    benchmark_dir: str = "./data/benchmark_instances",
    coreset_dir: str = "./data/coreset_instance",
    result_csv: str = "./result.csv",
    log_path: str = "./random_move_from_coreset.log",
):
    """
    coreset_dir 안의 *_coreset.json 들에 대해:

      - coreset 에서 random_move()
      - 나온 center 를 full data 기준으로 compute_center_dist 로 평가
      - result.csv 의 best cost 와 비교해서 .log 에 기록
    """
    benchmark_path = Path(benchmark_dir)
    coreset_path = Path(coreset_dir)
    log_file = Path(log_path)

    # result.csv 헤더에서 인스턴스별 best cost 로딩
    inst_best = load_instance_best_costs_from_columns(result_csv)
    print(f"[INFO] Loaded {len(inst_best)} instance best costs from {result_csv}")

    # 로그 헤더
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\n========== random_move_from_coreset run ==========\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"benchmark_dir = {benchmark_dir}\n")
        lf.write(f"coreset_dir   = {coreset_dir}\n")
        lf.write(f"result_csv    = {result_csv}\n\n")

    coreset_files = sorted(coreset_path.glob("*_coreset.json"))
    print(f"[INFO] Found {len(coreset_files)} coreset json files in {coreset_dir}")

    # 안전하게 순차 처리 (다른 곳에서 이미 multiprocessing을 쓰고 있을 수 있으니까)
    for co_file in coreset_files:
        process_one_coreset(co_file, benchmark_path, inst_best, log_file)


if __name__ == "__main__":
    random_move_from_coresets_and_log(
        benchmark_dir="./data/benchmark_instances",
        coreset_dir="./data/coreset_instance",
        result_csv="./result.csv",
        log_path="./random_move_from_coreset.log",
    )
