import argparse
import time
import math
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data import Data
from utils import load_best_costs, get_core_weights, evaluate_distance_and_path

def solve_global(bench_dir: str, core_dir: str, res_csv: str, log_path: str, exclude_rirs: bool):
    b_path, c_path = Path(bench_dir), Path(core_dir)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    inst_best = load_best_costs(res_csv)
    print(f"[INFO] Loaded {len(inst_best)} instance best costs from {res_csv}")

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("========== solve_from_coreset (findCenterGlobal + coreset weighted sum) ==========\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"benchmark_dir = {bench_dir}\n")
        lf.write(f"coreset_dir   = {core_dir}\n")
        lf.write(f"result_csv    = {res_csv}\n\n")

    coreset_files = sorted(c_path.glob("*_coreset.json"))
    print(f"[INFO] Found {len(coreset_files)} coreset json files in {core_dir}")

    for co_file in coreset_files:
        print("\n=======================================================")
        print(f"[CORESET] {co_file.name}")

        base_stem = co_file.stem[:-len("_coreset")] if co_file.stem.endswith("_coreset") else co_file.stem

        if exclude_rirs and base_stem.lower().startswith("rirs"):
            print(f"  [SKIP] base_stem starts with 'rirs': {base_stem}")
            continue

        full_file = b_path / f"{base_stem}.json"
        if not full_file.exists():
            print(f"  [WARN] full instance not found: {full_file} -> skip")
            continue

        dt_core, dt_full = Data(str(co_file)), Data(str(full_file))
        uid = getattr(dt_full, "instance_uid", base_stem)

        if exclude_rirs and uid.lower().startswith("rirs"):
            print(f"  [SKIP] instance_uid starts with 'rirs': {uid}")
            continue

        weights = get_core_weights(str(co_file), len(dt_core.triangulations))
        w_sum = float(sum(weights))

        # Find Center on Coreset
        t0 = time.time()
        center_tri = dt_core.findCenterGlobal()

        if center_tri is None:
            print("      [WARN] findCenterGlobal returned None! Trying internal self.center...")
            center_tri = getattr(dt_core, "center", None)
            if center_tri is None:
                print("      [WARN] Falling back to random_move() to generate center...")
                center_tri = dt_core.random_move()
        if center_tri is None:
            print("      [ERROR] Could not generate any center! Skipping this instance.")
            continue

        find_center_time = time.time() - t0
        print(f"      findCenterGlobal done. time = {find_center_time:.3f} s")

        # Evaluate on Coreset
        t_core0 = time.time()
        evaluate_distance_and_path(dt_core, center_tri)
        core_eval_time = time.time() - t_core0

        coreset_unweighted = float(sum(len(p) for p in dt_core.pFlips))
        coreset_weighted = float(sum(weights[i] * len(dt_core.pFlips[i]) for i in range(len(dt_core.pFlips))))

        print(f"      coreset dist sum (unweighted) : {coreset_unweighted}")
        print(f"      coreset dist sum (weighted)   : {coreset_weighted}   (sum_w={w_sum})")
        print(f"      time for coreset computeDistanceSum: {core_eval_time:.3f} s")

        # Evaluate on FULL instance
        t1 = time.time()
        evaluate_distance_and_path(dt_full, center_tri)
        eval_time = time.time() - t1

        coreset_dist_on_full = float(sum(len(p) for p in dt_full.pFlips))

        print(f"      center_dist on FULL (dist sum): {coreset_dist_on_full}")
        print(f"      time for FULL computeDistanceSum: {eval_time:.3f} s")

        # Save solution
        try: dt_full.center = center_tri.fast_copy()
        except: dt_full.center = center_tri
        dt_full.WriteData()

        # Compare with result.csv
        ref_cost = inst_best.get(uid) or inst_best.get(base_stem)
        diff = ratio = improvement = None
        comparison = "N/A"

        if ref_cost is not None:
            diff = coreset_dist_on_full - ref_cost
            improvement = ref_cost - coreset_dist_on_full
            ratio = (coreset_dist_on_full / ref_cost) if ref_cost != 0 else math.inf

            if coreset_dist_on_full < ref_cost: comparison = "better_than_result_csv"
            elif coreset_dist_on_full > ref_cost: comparison = "worse_than_result_csv"
            else: comparison = "equal_to_result_csv"

            print(f"      best cost in result.csv for [{uid}]: {ref_cost}")
            print(f"      diff(ours - best)            = {diff}")
            print(f"      improvement(best - ours)     = {improvement}")
            print(f"      ratio(ours / best)           = {ratio:.6f}")
            print(f"      comparison                   = {comparison}")
        else:
            print(f"      [WARN] No header column matching '{uid}' or '{base_stem}' in result.csv")

        # Save log
        lines = [
            "-------------------------------------------------------",
            f"[{co_file.name}]",
            f"instance_uid                 = {uid}",
            f"center_dist_on_full          = {coreset_dist_on_full:.6f}",
            f"coreset_dist_sum_unweighted  = {coreset_unweighted:.6f}",
            f"coreset_dist_sum_weighted    = {coreset_weighted:.6f}",
            f"coreset_weights_sum          = {w_sum:.6f}",
            f"findCenterGlobal_time       = {find_center_time:.3f}s",
            f"coreset_computeDistSum_time  = {core_eval_time:.3f}s",
            f"full_computeDistanceSum_time = {eval_time:.3f}s"
        ]

        if ref_cost is not None:
            lines.extend([
                f"best_cost(result.csv)        = {ref_cost:.6f}",
                f"diff(ours - best)            = {diff:.6f}",
                f"improvement(best - ours)     = {improvement:.6f}",
                f"ratio(ours / best)           = {ratio:.6f}",
                f"comparison                   = {comparison}"
            ])
        else:
            lines.append("best_cost(result.csv)        = None (no matching header column)")

        lines.append("-------------------------------------------------------")

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + "\n")

        print("=======================================================")

    print(f"\n[LOG] Saved results to {log_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Global search on coresets and evaluate on full instances.")
    p.add_argument("-b", "--bench_dir", default="data/benchmark_instances", help="Full instances directory")
    p.add_argument("-c", "--core_dir", default="data/coreset_instance", help="Coreset instances directory")
    p.add_argument("-v", "--res_csv", default="result.csv", help="CSV containing best costs")
    p.add_argument("-l", "--log_path", default="coreset/logs/solve_global.log", help="Output log file")
    p.add_argument("-r", "--exclude_rirs", action="store_true", default=False, help="Exclude rirs instances")

    args = p.parse_args()

    solve_global(args.bench_dir, args.core_dir, args.res_csv, args.log_path, args.exclude_rirs)
