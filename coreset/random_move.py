import argparse
import time
import sys
import os
from pathlib import Path
from multiprocessing import Pool

# Add path to find data.py in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import Data
from utils import patch_noassert, load_best_costs, evaluate_distance_and_path

def eval_pair(full_path: Path, core_path: Path, write_sol: bool) -> dict:
    """Optimizes on coreset using random_move and evaluates on full instance."""
    dt_core, dt_full = Data(str(core_path)), Data(str(full_path))
    
    # Patch to bypass internal assertion errors
    patch_noassert(dt_core)
    patch_noassert(dt_full)

    # Optimize on Coreset
    t0 = time.time()
    center_tri = dt_core.random_move()
    core_time = time.time() - t0
    
    # Using robust evaluation instead of unstable computeDistanceSum
    evaluate_distance_and_path(dt_core, center_tri)
    core_obj = float(dt_core.dist)

    # Evaluate on Full Instance
    t1 = time.time()
    evaluate_distance_and_path(dt_full, center_tri)
    full_time = time.time() - t1
    full_dist = float(dt_full.dist)

    # Save Solution
    try: dt_full.center = center_tri.fast_copy()
    except: dt_full.center = center_tri
    if write_sol: dt_full.WriteData()

    return {
        "uid": getattr(dt_full, "instance_uid", full_path.stem),
        "full_file": str(full_path.name),
        "core_file": str(core_path.name),
        "core_obj": core_obj,
        "full_dist": full_dist,
        "core_time_s": core_time,
        "full_time_s": full_time
    }

def _worker(args: tuple) -> dict:
    """Worker function for parallel processing."""
    f_path, c_path, write_sol = args
    f_path, c_path = Path(f_path), Path(c_path)

    if not f_path.exists() or not c_path.exists():
        return {"status": "skip", "reason": "file_missing", "full_file": f_path.name}

    try:
        res = eval_pair(f_path, c_path, write_sol)
        res["status"] = "ok"
        return res
    except Exception as e:
        return {"status": "error", "reason": str(e), "full_file": f_path.name}

def solve_random(bench_dir: str, core_dir: str, workers: int, exclude_rirs: bool, write_sol: bool, log_path: str, res_csv: str):
    """Executes search and logs results in block format."""
    b_path, c_path = Path(bench_dir), Path(core_dir)
    jobs = []

    # Load reference costs for comparison
    inst_best = load_best_costs(res_csv)

    for f_path in sorted(b_path.glob("*.json")):
        if f_path.name.endswith("_coreset.json"): continue
        if exclude_rirs and f_path.stem.lower().startswith("rirs"): continue
        
        co_path = c_path / f"{f_path.stem}_coreset.json"
        jobs.append((str(f_path), str(co_path), write_sol))

    print(f"[INFO] Starting {len(jobs)} jobs with {workers} workers...")
    t_start = time.time()

    if workers <= 1:
        results = [_worker(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_worker, jobs)

    # Write log file in block format
    out_log = Path(log_path)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_log, "a", encoding="utf-8") as f:
        f.write(f"\n========== solve_random Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
        for r in results:
            if r.get("status") != "ok": continue
            
            uid = r["uid"]
            full_dist = r["full_dist"]
            ref_cost = inst_best.get(uid) or inst_best.get(Path(r["full_file"]).stem)
            
            # Calculate quality ratio
            ratio_str = "N/A"
            if ref_cost and ref_cost > 0:
                ratio_str = f"{full_dist / ref_cost:.6f}"

            lines = [
                "-------------------------------------------------------",
                f"[{r['core_file']}]",
                f"instance_uid                 = {uid}",
                f"full_dist                    = {full_dist:.6f}",
                f"core_obj                     = {r['core_obj']:.6f}",
                f"ratio(ours / best)           = {ratio_str}",
                f"core_time_s                  = {r['core_time_s']:.3f}",
                f"full_time_s                  = {r['full_time_s']:.3f}",
                "-------------------------------------------------------"
            ]
            f.write("\n".join(lines) + "\n")

    ok_cnt = sum(1 for r in results if r.get("status") == "ok")
    err_cnt = sum(1 for r in results if r.get("status") == "error")
    
    print(f"[DONE] Success: {ok_cnt}, Error: {err_cnt}, Time: {time.time() - t_start:.2f}s")
    print(f"[LOG] Saved to {out_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel random_move search on coresets.")
    parser.add_argument("-b", "--bench_dir", default="data/benchmark_instances")
    parser.add_argument("-c", "--core_dir", default="data/coreset_instances")
    parser.add_argument("-w", "--workers", type=int, default=16)
    parser.add_argument("-r", "--exclude_rirs", action="store_true", help="Exclude RIRS instances")
    parser.add_argument("-s", "--write_sol", action="store_true")
    parser.add_argument("-v", "--res_csv", default="result.csv", help="Reference CSV for ratio")
    parser.add_argument("-l", "--log_path", default="coreset/logs/solve_random.log", help="Output log path")
    args = parser.parse_args()

    solve_random(args.bench_dir, args.core_dir, args.workers, args.exclude_rirs, args.write_sol, args.log_path, args.res_csv)