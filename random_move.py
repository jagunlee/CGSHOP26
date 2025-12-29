import argparse
import sys
from pathlib import Path
from multiprocessing import Pool

from data import Data


def _infer_full_instance_path(coreset_json: Path, benchmark_dir: Path) -> Path | None:
    """
    Infer full instance json path from a coreset json name.

    Example:
      random_instance_4_40_2_coreset.json -> random_instance_4_40_2.json

    Search order:
      1) same directory as coreset file
      2) benchmark_dir
    """
    name = coreset_json.name

    if name.endswith("_coreset.json"):
        full_name = name.replace("_coreset.json", ".json")
    elif name.endswith("_coreset.solution.json"):
        full_name = name.replace("_coreset.solution.json", ".json")
    else:
        stem = coreset_json.stem
        if stem.endswith("_coreset"):
            full_name = stem[:-len("_coreset")] + ".json"
        else:
            full_name = (stem.replace("_coreset", "") + ".json") if "_coreset" in stem else (stem + ".json")

    cand1 = coreset_json.parent / full_name
    if cand1.exists():
        return cand1

    cand2 = benchmark_dir / full_name
    if cand2.exists():
        return cand2

    return None


def _maybe_disable_writes(dt: Data, enable: bool):
    """
    Prevent dt.random_move() from writing intermediate coreset solutions.
    (data.py의 random_move가 내부에서 WriteData()를 호출할 수 있어서, 기본은 막아둠)
    """
    if enable:
        return
    dt.WriteData = lambda *args, **kwargs: None


def process_one(args):
    coreset_path_str, full_path_str, benchmark_dir_str, also_write_coreset = args
    coreset_path = Path(coreset_path_str)
    benchmark_dir = Path(benchmark_dir_str)

    # 1) Load coreset instance
    dt_core = Data(str(coreset_path))
    _maybe_disable_writes(dt_core, enable=also_write_coreset)

    # 2) Run weighted random_move on coreset -> get center
    center = dt_core.random_move()
    if center is None:
        center = getattr(dt_core, "center", None)
    if center is None:
        print(f"[ERROR] Failed to obtain center from coreset: {coreset_path}")
        return

    # 3) Resolve full instance path
    if full_path_str:
        full_path = Path(full_path_str)
        if full_path.is_dir():
            inferred = _infer_full_instance_path(coreset_path, full_path)
            if inferred is None:
                inferred = _infer_full_instance_path(coreset_path, benchmark_dir)
            full_path = inferred if inferred is not None else Path(full_path_str)
    else:
        full_path = _infer_full_instance_path(coreset_path, benchmark_dir)

    if full_path is None or (not full_path.exists()):
        print(f"[WARN] Full instance not found for {coreset_path.name}. "
              f"Provide --full or check --benchmark-dir. (searched: {benchmark_dir})")
        return

    # 4) Evaluate center on FULL data and write solution
    dt_full = Data(str(full_path))
    dt_full.computeDistanceSum(center)

    # Ensure WriteData records the actual center
    try:
        dt_full.center = center.fast_copy()
    except Exception:
        dt_full.center = center

    dt_full.WriteData()
    print(f"[OK] coreset={coreset_path.name} -> full={full_path.name} : wrote solution for full instance")


def collect_coreset_files(path: Path):
    if path.is_file():
        return [path]

    files = []
    for p in sorted(path.glob("*.json")):
        if "_coreset" in p.name:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Run weighted random_move on a CORESET instance, then evaluate the resulting center on FULL data and WriteData()."
    )
    ap.add_argument("coreset", help="Coreset instance json file, or a directory containing *_coreset*.json files.")
    ap.add_argument("--full", default="", help="Full instance json file or directory. If omitted, infer from name and --benchmark-dir.")
    ap.add_argument("--benchmark-dir", default="./data/benchmark_instances",
                    help="Directory to search full instance json when --full is omitted (default: ./data/benchmark_instances).")
    ap.add_argument("--jobs", type=int, default=1, help="Number of parallel processes for directory mode (default: 1).")
    ap.add_argument("--also-write-coreset", action="store_true",
                    help="Allow dt_core.random_move() to WriteData() for the coreset too (default: disabled).")
    args = ap.parse_args()

    coreset_path = Path(args.coreset)
    benchmark_dir = Path(args.benchmark_dir)
    full_arg = args.full

    coreset_files = collect_coreset_files(coreset_path)
    if not coreset_files:
        print(f"[ERROR] No coreset json files found: {coreset_path}")
        sys.exit(1)

    payloads = [(str(c), full_arg, str(benchmark_dir), args.also_write_coreset) for c in coreset_files]

    if args.jobs <= 1 or len(payloads) == 1:
        for p in payloads:
            process_one(p)
    else:
        with Pool(processes=args.jobs) as pool:
            pool.map(process_one, payloads)


if __name__ == "__main__":
    main()
