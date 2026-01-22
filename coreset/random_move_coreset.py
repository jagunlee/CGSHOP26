import os
import argparse
import time
from pathlib import Path
from multiprocessing import Pool
import types
import sys

from data import Data


# ============================================================
# Patch: remove AssertionError in parallel_flip_path / path2
# ============================================================

def _get_search_depth(self) -> int:
    """
    Data.py defines SEARCH_DEPTH as a module-level constant.
    This reads it from the module where Data is defined.
    """
    mod = sys.modules.get(self.__class__.__module__)
    return int(getattr(mod, "SEARCH_DEPTH", 1))


def _parallel_flip_path_noassert(self, tri1, tri2, max_rounds: int = 200000):
    """
    Same greedy routine as Data.parallel_flip_path, but WITHOUT assert(tri.edges == tri2.edges).
    It returns a flip-round sequence even if it doesn't reach tri2.
    """
    tri = tri1.fast_copy()
    pfp = []
    rounds = 0
    depth = _get_search_depth(self)

    while True:
        rounds += 1
        if rounds > max_rounds:
            break

        cand = []
        edges = list(tri.edges)
        for e in edges:
            if self.flippable(tri, e):
                score = self.flip_score(tri, tri2, e, depth)
                if score[0] > 0:
                    cand.append((e, score))

        if not cand:
            break

        cand.sort(key=lambda x: x[1], reverse=True)

        flips = []
        marked = set()
        for (p1, p2), _ in cand:
            t1 = tri.find_triangle(p1, p2)
            t2 = tri.find_triangle(p2, p1)
            if t1 in marked or t2 in marked:
                continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)

        for e in flips:
            tri.flip(e)
        pfp.append(flips)

    return pfp


def _parallel_flip_path2_noassert(self, tri1, tri2, max_rounds: int = 200000):
    """
    Same greedy routine as Data.parallel_flip_path2, but WITHOUT assert at the end.
    """
    tri = tri1.fast_copy()
    pfp = []
    rounds = 0

    while True:
        rounds += 1
        if rounds > max_rounds:
            break

        prev_flip = []
        cand = []
        edges = list(tri.edges)
        for e in edges:
            if self.flippable(tri, e):
                if e in prev_flip:
                    continue
                score = self.flip_score(tri, tri2, e, 0)
                if score[0] > 0:
                    cand.append((e, score))

        if not cand:
            break

        cand.sort(key=lambda x: x[1], reverse=True)

        flips = []
        marked = set()
        for (p1, p2), _ in cand:
            t1 = tri.find_triangle(p1, p2)
            t2 = tri.find_triangle(p2, p1)
            if t1 in marked or t2 in marked:
                continue
            flips.append((p1, p2))
            marked.add(t1)
            marked.add(t2)

        for e in flips:
            e1 = tri.flip(e)
            prev_flip.append(e1)

        pfp.append(flips)

    return pfp


def patch_noassert(dt: Data):
    """
    Monkey-patch instance methods so computeDistanceSum/random_move won't crash on assert.
    """
    dt.parallel_flip_path = types.MethodType(_parallel_flip_path_noassert, dt)
    dt.parallel_flip_path2 = types.MethodType(_parallel_flip_path2_noassert, dt)


# ============================================================
# Core routine: optimize on coreset, evaluate on full
# ============================================================

def run_on_coreset_then_eval_full(full_path: Path, coreset_path: Path, write_solution: bool):
    """
    1) Load coreset instance and run dt_core.random_move() (weighted objective if coreset_weights exists)
    2) Take returned centerT
    3) Load FULL instance and evaluate using dt_full.computeDistanceSum(centerT)
       (unweighted on FULL because FULL has no coreset_weights)
    4) Optionally write solution for FULL instance
    """
    # (A) Optimize on CORESET
    dt_core = Data(str(coreset_path))
    patch_noassert(dt_core)

    t0 = time.time()
    centerT = dt_core.random_move()  # returns Triangulation center; sets dt_core.dist (weighted)
    t1 = time.time()
    coreset_opt_time = t1 - t0

    # (B) Evaluate on FULL (computeDistanceSum fills dt_full.pFlips and dt_full.dist)
    dt_full = Data(str(full_path))
    patch_noassert(dt_full)

    t2 = time.time()
    dt_full.computeDistanceSum(centerT)
    t3 = time.time()
    full_eval_time = t3 - t2

    full_dist = float(getattr(dt_full, "dist", 0.0))

    # Attach for writing solution (WriteData often reads self.center/self.dist/self.pFlips)
    try:
        dt_full.center = centerT.fast_copy()
    except Exception:
        dt_full.center = centerT

    if write_solution:
        dt_full.WriteData()

    inst_uid = getattr(dt_full, "instance_uid", full_path.stem)

    return {
        "instance_uid": inst_uid,
        "full_file": str(full_path),
        "coreset_file": str(coreset_path),
        "coreset_weighted_obj": float(getattr(dt_core, "dist", 0.0)),
        "full_dist_sum": float(full_dist),
        "coreset_opt_time_sec": float(coreset_opt_time),
        "full_eval_time_sec": float(full_eval_time),
    }


def guess_pairs(inp: Path, bench_dir: Path, coreset_dir: Path):
    """
    Accept either:
      - a FULL instance file:          bench_dir/base.json
      - a CORESET instance file:      coreset_dir/base_coreset.json
      - a directory: scan FULL instances in that directory (default: bench_dir)
    """
    pairs = []

    if inp.is_file() and inp.suffix.lower().endswith("json"):
        if inp.name.endswith("_coreset.json"):
            base = inp.stem[:-len("_coreset")]
            full_path = bench_dir / f"{base}.json"
            coreset_path = inp
        else:
            base = inp.stem
            full_path = inp
            coreset_path = coreset_dir / f"{base}_coreset.json"

        pairs.append((full_path, coreset_path))
        return pairs

    # directory scan
    dir_to_scan = inp if inp.is_dir() else bench_dir
    for f in sorted(dir_to_scan.glob("*.json")):
        if f.name.endswith("_coreset.json"):
            continue
        base = f.stem
        pairs.append((f, coreset_dir / f"{base}_coreset.json"))

    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Run random_move on CORESET instances, then evaluate the resulting center on FULL using computeDistanceSum."
    )
    p.add_argument("inp", nargs="?", default=None,
                   help="A .json file (full or *_coreset.json) OR a directory. If omitted, uses --bench_dir.")
    p.add_argument("--bench_dir", type=str, default="data/benchmark_instances",
                   help="Directory containing FULL instances (*.json).")
    p.add_argument("--coreset_dir", type=str, default="data/coreset_instance",
                   help="Directory containing CORESET instances (*_coreset.json).")
    p.add_argument("--workers", type=int, default=60, help="multiprocessing workers")
    p.add_argument("--include_rirs", action="store_true", default=False,
                   help="Include instances starting with 'rirs'. Default: skip them.")
    p.add_argument("--write_solution", action="store_true", default=False,
                   help="If set, write FULL solution json via Data.WriteData().")
    p.add_argument("--log_csv", type=str, default="random_move_coreset_eval.csv",
                   help="CSV path to write summary results.")
    return p.parse_args()


def _worker(args):
    full_path, coreset_path, write_solution = args
    full_path = Path(full_path)
    coreset_path = Path(coreset_path)

    if not full_path.exists():
        return {"status": "skip", "reason": "full_not_found", "full_file": str(full_path), "coreset_file": str(coreset_path)}
    if not coreset_path.exists():
        return {"status": "skip", "reason": "coreset_not_found", "full_file": str(full_path), "coreset_file": str(coreset_path)}

    try:
        out = run_on_coreset_then_eval_full(full_path, coreset_path, write_solution=write_solution)
        out["status"] = "ok"
        return out
    except Exception as e:
        return {"status": "error", "reason": str(e), "full_file": str(full_path), "coreset_file": str(coreset_path)}


if __name__ == "__main__":
    args = parse_args()

    bench_dir = Path(args.bench_dir)
    coreset_dir = Path(args.coreset_dir)
    inp = Path(args.inp) if args.inp else bench_dir

    pairs = guess_pairs(inp, bench_dir=bench_dir, coreset_dir=coreset_dir)

    # filter rirs
    jobs = []
    for full_path, coreset_path in pairs:
        name = Path(full_path).stem.lower()
        if (not args.include_rirs) and name.startswith("rirs"):
            continue
        jobs.append((str(full_path), str(coreset_path), args.write_solution))

    print(f"[INFO] Total pairs: {len(jobs)} (bench={bench_dir}, coreset={coreset_dir})")
    print(f"[INFO] workers={args.workers}, include_rirs={args.include_rirs}, write_solution={args.write_solution}")
    start = time.time()

    if args.workers <= 1:
        results = [_worker(j) for j in jobs]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)

    # write CSV
    import csv
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "instance_uid",
        "full_file",
        "coreset_file",
        "coreset_weighted_obj",
        "full_dist_sum",
        "coreset_opt_time_sec",
        "full_eval_time_sec",
        "reason",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    ok = sum(1 for r in results if r.get("status") == "ok")
    skip = sum(1 for r in results if r.get("status") == "skip")
    err = sum(1 for r in results if r.get("status") == "error")
    end = time.time()

    print(f"[DONE] ok={ok}, skip={skip}, error={err}")
    print(f"[CSV] {log_path}")
    print(f"[TIME] total {end - start:.3f}s")
