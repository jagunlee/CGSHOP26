import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp

# data.py must be on PYTHONPATH (same folder as this script, or installed)
from data import Data


def _infer_full_instance_path(coreset_json: Path, benchmark_dir: Path):
    """
    Infer full instance json path from a coreset json name.

    Examples:
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
    Prevent dt.random_move() from writing intermediate coreset solutions if user doesn't want it.
    """
    if enable:
        return
    dt.WriteData = lambda *args, **kwargs: None


def _solution_cost(sol_json_path: Path):
    """
    Read a *.solution.json and return its cost (dist).
    Supports either meta.dist, or fallback to flips length.
    """
    try:
        with open(sol_json_path, "r", encoding="utf-8") as f:
            root = json.load(f)
        meta = root.get("meta", {})
        if isinstance(meta, dict) and "dist" in meta:
            return float(meta["dist"])
        flips = root.get("flips", None)
        if flips is not None:
            return float(sum(len(r) for r in flips))
    except Exception:
        return None
    return None


def _best_opt_cost(instance_uid: str, opt_dir: Path):
    """
    Mimic data.py WriteData() behavior: search opt_dir for any file containing
    '<instance_uid>.solution.json' and return the minimum cost among them.
    """
    if not opt_dir.exists():
        return None

    needle = f"{instance_uid}.solution.json"
    best = None
    for p in opt_dir.iterdir():
        if not p.is_file():
            continue
        if needle in p.name:
            c = _solution_cost(p)
            if c is None:
                continue
            if best is None or c < best:
                best = c
    return best


def _best_result_csv_cost(result_csv: Path, instance_uid: str):
    """
    Return historical best (minimum) for instance_uid column in result.csv.
    """
    if not result_csv.exists():
        return None

    try:
        with open(result_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or instance_uid not in reader.fieldnames:
                return None
            best = None
            for row in reader:
                v = row.get(instance_uid, "")
                if v is None:
                    continue
                v = str(v).strip()
                if not v:
                    continue
                try:
                    x = float(v)
                except Exception:
                    continue
                if not (x < float("inf")):
                    continue
                if best is None or x < best:
                    best = x
            return best
    except Exception:
        return None


def _format_delta(old, new):
    if old is None or new is None:
        return "N/A"
    try:
        d = float(old) - float(new)
        return f"{d:.6f}"
    except Exception:
        return "N/A"


def _format_ratio(new, old):
    if old is None or new is None:
        return "N/A"
    try:
        old = float(old)
        new = float(new)
        if old == 0:
            return "INF" if new != 0 else "1.0"
        return f"{(new / old):.6f}"
    except Exception:
        return "N/A"


def _append_log(log_path: Path, lock, text: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text)
        return
    with lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text)


def process_one(payload):
    (
        coreset_path_str,
        full_path_str,
        benchmark_dir_str,
        also_write_coreset,
        result_csv_str,
        opt_dir_str,
        log_path_str,
        lock,
    ) = payload

    coreset_path = Path(coreset_path_str)
    benchmark_dir = Path(benchmark_dir_str)
    result_csv = Path(result_csv_str)
    opt_dir = Path(opt_dir_str)
    log_path = Path(log_path_str)

    # 1) Load coreset instance
    dt_core = Data(str(coreset_path))
    _maybe_disable_writes(dt_core, enable=also_write_coreset)

    # 2) Run (weighted) random_move on coreset -> center
    center = dt_core.random_move()
    if center is None:
        center = getattr(dt_core, "center", None)
    if center is None:
        msg = f"[ERROR] Failed to obtain center from coreset: {coreset_path}\n"
        print(msg, end="")
        _append_log(log_path, lock, msg)
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
        msg = (f"[WARN] Full instance not found for {coreset_path.name}. "
               f"Provide --full or check --benchmark-dir. (searched: {benchmark_dir})\n")
        print(msg, end="")
        _append_log(log_path, lock, msg)
        return

    # 4) Load FULL instance (to get uid), read baselines BEFORE writing
    dt_full = Data(str(full_path))

    # ✅ instance_uid: 파일명만 (확장자 제거)
    uid = Path(full_path).stem

    old_opt_best = _best_opt_cost(uid, opt_dir)
    old_csv_best = _best_result_csv_cost(result_csv, uid)

    # 5) Evaluate center on FULL
    t0 = time.time()
    dt_full.computeDistanceSum(center)
    t1 = time.time()

    try:
        dt_full.center = center.fast_copy()
    except Exception:
        dt_full.center = center

    new_cost = getattr(dt_full, "dist", None)
    if new_cost is None:
        try:
            new_cost = float(sum(len(x) for x in dt_full.pFlips))
        except Exception:
            new_cost = None

    if old_opt_best is None:
        status = "NO_PREVIOUS_OPT"
    else:
        try:
            status = "IMPROVED_OPT" if (new_cost is not None and new_cost < float(old_opt_best)) else "WORSE_THAN_OPT"
        except Exception:
            status = "UNKNOWN"

    dt_full.WriteData()

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = []
    entry.append("-------------------------------------------------------\n")
    entry.append(f"time                    = {now}\n")
    entry.append(f"instance_uid            = {uid}\n")  # ✅ 경로 제거됨
    entry.append(f"coreset_file            = {coreset_path.name}\n")
    entry.append(f"full_file               = {full_path.name}\n")
    entry.append(f"old_opt_best            = {old_opt_best if old_opt_best is not None else 'None'}\n")
    entry.append(f"old_result_csv_best     = {old_csv_best if old_csv_best is not None else 'None'}\n")
    entry.append(f"new_cost_full           = {new_cost if new_cost is not None else 'None'}\n")
    entry.append(f"improve_vs_old_opt      = {_format_delta(old_opt_best, new_cost)}\n")
    entry.append(f"improve_vs_csv_best     = {_format_delta(old_csv_best, new_cost)}\n")
    entry.append(f"ratio_new/old_opt       = {_format_ratio(new_cost, old_opt_best)}\n")
    entry.append(f"ratio_new/csv_best      = {_format_ratio(new_cost, old_csv_best)}\n")
    entry.append(f"full_eval_time_sec      = {t1 - t0:.3f}\n")
    entry.append(f"status                  = {status}\n")
    entry.append("-------------------------------------------------------\n")

    entry_text = "".join(entry)
    _append_log(log_path, lock, entry_text)

    print(f"[OK] {uid}: new={new_cost} | old_opt={old_opt_best} | csv_best={old_csv_best} | {status}")



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
        description="Run weighted random_move on a CORESET instance, then evaluate the resulting center on FULL data and WriteData(). Also logs improvements vs opt and result.csv best."
    )
    ap.add_argument("coreset", help="Coreset instance json file, or a directory containing *_coreset*.json files.")
    ap.add_argument("--full", default="", help="Full instance json file or directory. If omitted, infer from name and --benchmark-dir.")
    ap.add_argument("--benchmark-dir", default="./data/benchmark_instances",
                    help="Directory to search full instance json when --full is omitted.")
    ap.add_argument("--jobs", type=int, default=1, help="Number of parallel processes for directory mode.")
    ap.add_argument("--also-write-coreset", action="store_true",
                    help="Allow dt_core.random_move() to WriteData() for the coreset too (default: off).")

    ap.add_argument("--result-csv", default="result.csv", help="Path to result.csv (default: result.csv).")
    ap.add_argument("--opt-dir", default="opt", help="Directory containing existing solutions to compare (default: opt).")
    ap.add_argument("--log", default="./logs/random_move_coreset.log", help="Log file path (default: ./logs/random_move_coreset.log).")

    args = ap.parse_args()

    coreset_path = Path(args.coreset)
    benchmark_dir = Path(args.benchmark_dir)

    coreset_files = collect_coreset_files(coreset_path)
    if not coreset_files:
        print(f"[ERROR] No coreset json files found: {coreset_path}")
        sys.exit(1)

    # header for log
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n========== random_move_coreset run @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
        f.write(f"coreset={args.coreset}\nfull={args.full}\nbenchmark_dir={args.benchmark_dir}\n")
        f.write(f"result_csv={args.result_csv}\nopt_dir={args.opt_dir}\n\n")

    manager = mp.Manager()
    lock = manager.Lock() if args.jobs and args.jobs > 1 else None

    payloads = []
    for c in coreset_files:
        payloads.append((
            str(c),
            args.full,
            str(benchmark_dir),
            args.also_write_coreset,
            args.result_csv,
            args.opt_dir,
            str(log_path),
            lock,
        ))

    if args.jobs <= 1 or len(payloads) == 1:
        for p in payloads:
            process_one(p)
    else:
        # On Windows, spawn is default; Pool works fine when guarded by __main__
        with mp.Pool(processes=args.jobs) as pool:
            pool.map(process_one, payloads)

    print(f"\n[LOG] Saved to: {log_path}")


if __name__ == "__main__":
    main()
