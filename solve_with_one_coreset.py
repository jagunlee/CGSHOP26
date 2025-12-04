import time
from pathlib import Path

from data2 import Data  # 가중치 포함 find_center_np / ReadData가 들어있는 버전

def solve_from_coreset(
    benchmark_dir: str = "./data/benchmark_instances",
    coreset_dir: str = "./data/coreset_instance",
    coreset_name: str = "rirs-3000-75-d0dd4d29_coreset.json",
):
    """
    coreset_name 하나에 대해서만:

      (1) 코어셋 데이터(data_core)에서 weighted find_center_np 실행
      (2) 얻은 center를 full 인스턴스(data_full)에 대해 compute_center_dist로 평가
      (3) data_full.center / dist / flip 설정 후 data_full.WriteData() 호출
    """

    benchmark_path = Path(benchmark_dir)
    coreset_path = Path(coreset_dir)

    co_file = coreset_path / coreset_name
    if not co_file.exists():
        print(f"[ERROR] coreset file not found: {co_file}")
        return

    print("\n=======================================================")
    print(f"[CORESET] {co_file.name}")

    # 원본 인스턴스 파일 이름 유추
    # 예: rirs-500-75-9322678f_coreset.json -> rirs-500-75-9322678f.json
    stem = co_file.stem
    if stem.endswith("_coreset"):
        base_stem = stem[: -len("_coreset")]
    else:
        base_stem = stem  # 혹시 _coreset 없는 경우도 처리

    full_file = benchmark_path / f"{base_stem}.json"
    if not full_file.exists():
        print(f"  [WARN] full instance not found: {full_file}  -> abort")
        return

    # (A) 데이터 로딩
    data_full = Data(str(full_file))
    data_core = Data(str(co_file))  # 여기서 tri_weights = coreset_weights 로 읽힘

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

    # solution json 저장 (data2.py의 WriteData 구현대로 저장됨)
    data_full.WriteData()

    print(f"      coreset center_dist (full 기준): {dist_core_on_full}")
    print(f"      time for compute_center_dist:    {t3 - t2:.3f} s")
    print("=======================================================")


if __name__ == "__main__":
    solve_from_coreset(
        benchmark_dir="./data/benchmark_instances",
        coreset_dir="./data/coreset_instance",
        coreset_name="rirs-9000-20-41491eba_coreset.json",
    )
