import argparse
import json
import time
import math
import sys
import random
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data import Data
from utils import get_center_from_sol, get_safe_distances

def build_coreset(data: Data, center_tri, eps=0.1, alpha=8.0, min_alpha=16.0, dist_mode="min", weighted=True):
    """
    Performs 1D grid bucketing based on the center triangulation
    to return coreset indices, weights, and distance info.
    """
    n = len(data.triangulations)
    if n == 0:
        return [], [], {}

    weights = data.get_weights(n) if hasattr(data, 'get_weights') else [1.0] * n
    total_weight = float(sum(weights))

    # Safe distance computation (prevents assert errors)
    nearest_dist, nu_A, dist_info = get_safe_distances(data, center_tri, dist_mode, weighted)
    nearest_dist = np.asarray(nearest_dist, dtype=float)

    R = max(float(nu_A) / max(1.0, total_weight), 1e-12)
    Dmax = float(nearest_dist.max())

    # If all data points are within a very small radius, pick a single representative
    if Dmax < R:
        rep = int(np.argmin(nearest_dist))
        return [rep], [total_weight], dist_info

    M = max(1, math.ceil(math.log2((Dmax + R) / R)))
    buckets = {}

    for i in range(n):
        dist, wi = float(nearest_dist[i]), float(weights[i])
        j = min(int(math.ceil(math.log2(max(dist, 1e-12) / R))), M) if dist >= R else 0
        rj = max((eps * R * (2 ** j)) / float(alpha), (eps * R) / float(min_alpha))
        key = (j, int(math.floor(dist / rj)))

        if key not in buckets:
            buckets[key] = [int(i), wi]
        else:
            buckets[key][1] += wi

    reps, wts = zip(*buckets.values())
    return list(reps), list(wts), dist_info


def process_all(bench_dir, core_dir, opt_dir, eps, alpha, min_alpha, dist_mode, weighted, exclude_rirs):
    """Iterates through all instances to generate and save coresets. Uses random center if opt is missing."""
    bench_path, core_path = Path(bench_dir), Path(core_dir)
    core_path.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(bench_path.glob("*.json")):
        # === [NEW] Skip rirs instances ONLY IF exclude_rirs is True ===
        if exclude_rirs and json_file.stem.lower().startswith("rirs"):
            continue

        print(f"\n[PROCESS] {json_file.name}")
        start_t = time.time()

        try:
            inst = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        base_uid = inst.get("instance_uid", json_file.stem)
        num_tris = len(inst.get("triangulations", []))
        if num_tris == 0:
            continue

        data_obj = Data(str(json_file))

        # Reconstruct the center triangulation from the optimal solution
        center_tri, center_meta = get_center_from_sol(data_obj, opt_dir, base_uid)

        # === Fallback to a random center if solution is not found ===
        if center_tri is None:
            rand_idx = random.randint(0, num_tris - 1)
            center_tri = data_obj.triangulations[rand_idx]

        # Generate coreset
        reps, wts, dist_info = build_coreset(
            data_obj, center_tri, eps, alpha, min_alpha, dist_mode, weighted
        )

        # Assemble and save as a new JSON instance
        coreset_inst = {
            "content_type": inst.get("content_type", "CGSHOP2026_Instance"),
            "instance_uid": f"{base_uid}_coreset",
            "points_x": inst["points_x"],
            "points_y": inst["points_y"],
            "triangulations": [inst["triangulations"][i] for i in reps],
            "coreset_weights": wts,
            "meta": {
                **center_meta,
                **dist_info,
                "eps": eps,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        out_file = core_path / f"{json_file.stem}_coreset.json"
        out_file.write_text(json.dumps(coreset_inst, indent=2))
        print(f"  [DONE] reps: {len(reps)}, time: {time.time()-start_t:.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build coresets for CG:SHOP 2026 instances.")
    p.add_argument("-b", "--bench_dir", default="data/benchmark_instances", help="Path to benchmark directory")
    p.add_argument("-c", "--core_dir", default="data/coreset_instance", help="Output directory for coresets")
    p.add_argument("-o", "--opt_dir", default="../opt", help="Directory containing optimal solutions")
    p.add_argument("-e", "--eps", type=float, default=0.1, help="Epsilon for coreset precision")
    p.add_argument("-a", "--alpha", type=float, default=8.0, help="Alpha parameter")
    p.add_argument("-m", "--min_alpha", type=float, default=16.0, help="Minimum alpha parameter")
    p.add_argument("-d", "--dist_mode", default="min", choices=["pfp", "pfp2", "min"], help="Distance calculation mode")
    p.add_argument("-w", "--weighted", action="store_true", default=True, help="Use weighted distance")
    p.add_argument("-r", "--exclude_rirs", action="store_true", default=False, help="Exclude rirs instances")

    args = p.parse_args()

    process_all(
        args.bench_dir, args.core_dir, args.opt_dir,
        args.eps, args.alpha, args.min_alpha,
        args.dist_mode, args.weighted, args.exclude_rirs
    )
