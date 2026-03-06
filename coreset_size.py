# compare_coreset_sizes.py
import json
from pathlib import Path
from collections import defaultdict

ORIG_DIR = Path("./data/benchmark_instances")
CORE_DIR = Path("./data/coreset_instance_260118")
OUT_LOG  = Path("output2.log")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tri_count(inst: dict) -> int:
    tris = inst.get("triangulations", [])
    return len(tris) if isinstance(tris, list) else 0


def summarize_group(records):
    """
    records: list of (name, orig_n, core_n, reduction%)
    """
    if not records:
        return None
    avg_reduction = sum(r[3] for r in records) / len(records)
    avg_remaining = sum((r[2] / r[1]) for r in records) / len(records) * 100.0
    avg_orig_n = sum(r[1] for r in records) / len(records)
    avg_core_n = sum(r[2] for r in records) / len(records)
    return {
        "count": len(records),
        "avg_reduction": avg_reduction,
        "avg_remaining": avg_remaining,
        "avg_orig_n": avg_orig_n,
        "avg_core_n": avg_core_n,
        "min_reduction": min(r[3] for r in records),
        "max_reduction": max(r[3] for r in records),
    }


def main():
    orig_files = sorted(ORIG_DIR.glob("*.json"))
    if not orig_files:
        msg = f"[ERROR] No json files found in {ORIG_DIR.resolve()}\n"
        OUT_LOG.write_text(msg, encoding="utf-8")
        print(msg, end="")
        return

    records = []     # (file_name, orig_n, core_n, reduction_percent)
    missing = []
    parse_fail = []

    for orig_path in orig_files:
        try:
            orig = load_json(orig_path)
            orig_n = tri_count(orig)
        except Exception as e:
            parse_fail.append((orig_path.name, f"orig read failed: {e}"))
            continue

        core_path = CORE_DIR / f"{orig_path.stem}_coreset.json"
        if not core_path.exists():
            missing.append(orig_path.name)
            continue

        try:
            core = load_json(core_path)
            core_n = tri_count(core)
        except Exception as e:
            parse_fail.append((orig_path.name, f"coreset read failed: {e}"))
            continue

        if orig_n <= 0:
            continue

        reduction = (orig_n - core_n) / orig_n * 100.0
        records.append((orig_path.name, orig_n, core_n, reduction))

    # aggregate by original_n
    by_orig_n = defaultdict(list)      # orig_n -> list of reductions
    by_orig_n_sizes = defaultdict(list)  # orig_n -> list of core_n
    for _, orig_n, core_n, red in records:
        by_orig_n[orig_n].append(red)
        by_orig_n_sizes[orig_n].append(core_n)

    # group split: orig_n <= 10 vs > 10
    group_le10 = [r for r in records if r[1] <= 10]
    group_gt10 = [r for r in records if r[1] > 10]
    sum_le10 = summarize_group(group_le10)
    sum_gt10 = summarize_group(group_gt10)

    lines = []
    lines.append("=== Coreset size reduction report ===")
    lines.append(f"orig_dir   : {ORIG_DIR.resolve()}")
    lines.append(f"coreset_dir: {CORE_DIR.resolve()}")
    lines.append(f"total original instances: {len(orig_files)}")
    lines.append(f"matched instances       : {len(records)}")
    lines.append(f"missing coreset files   : {len(missing)}")
    lines.append(f"parse failures          : {len(parse_fail)}")
    lines.append("")

    if records:
        overall = summarize_group(records)
        lines.append("=== Overall ===")
        lines.append(f"Average reduction (%) : {overall['avg_reduction']:.2f}")
        lines.append(f"Average remaining (%) : {overall['avg_remaining']:.2f}  (coreset/original)")
        lines.append(f"Avg sizes             : orig={overall['avg_orig_n']:.2f}, coreset={overall['avg_core_n']:.2f}")
        lines.append(f"Reduction range (%)   : [{overall['min_reduction']:.2f}, {overall['max_reduction']:.2f}]")
        lines.append("")

        lines.append("=== Split by original triangulation count (<=10 vs >10) ===")
        if sum_le10:
            lines.append(f"orig_n <= 10 : #inst={sum_le10['count']}, "
                         f"avg_reduction={sum_le10['avg_reduction']:.2f}%, "
                         f"avg_remaining={sum_le10['avg_remaining']:.2f}%, "
                         f"avg_sizes(orig->core)={sum_le10['avg_orig_n']:.2f}->{sum_le10['avg_core_n']:.2f}")
        else:
            lines.append("orig_n <= 10 : #inst=0")

        if sum_gt10:
            lines.append(f"orig_n >  10 : #inst={sum_gt10['count']}, "
                         f"avg_reduction={sum_gt10['avg_reduction']:.2f}%, "
                         f"avg_remaining={sum_gt10['avg_remaining']:.2f}%, "
                         f"avg_sizes(orig->core)={sum_gt10['avg_orig_n']:.2f}->{sum_gt10['avg_core_n']:.2f}")
        else:
            lines.append("orig_n >  10 : #inst=0")
        lines.append("")

        lines.append("=== Reduction by original triangulation count ===")
        lines.append("orig_n | #inst | avg_core_n | avg_reduction(%) | min(%) | max(%)")
        for orig_n in sorted(by_orig_n.keys()):
            reds = by_orig_n[orig_n]
            cores = by_orig_n_sizes[orig_n]
            avg_red = sum(reds) / len(reds)
            avg_core = sum(cores) / len(cores)
            lines.append(
                f"{orig_n:6d} | {len(reds):5d} | {avg_core:10.2f} | {avg_red:15.2f} | {min(reds):6.2f} | {max(reds):6.2f}"
            )
        lines.append("")

        records_sorted = sorted(records, key=lambda x: x[3], reverse=True)
        lines.append("=== Examples (top 10 reductions) ===")
        for name, orig_n, core_n, red in records_sorted[:10]:
            lines.append(f"{name}: {orig_n} -> {core_n}  ({red:.2f}%)")
        lines.append("")
        lines.append("=== Examples (bottom 10 reductions) ===")
        for name, orig_n, core_n, red in records_sorted[-10:]:
            lines.append(f"{name}: {orig_n} -> {core_n}  ({red:.2f}%)")
        lines.append("")
    else:
        lines.append("[WARN] No matched instances to report.\n")

    if missing:
        lines.append("=== Missing coreset files (first 30) ===")
        for m in missing[:30]:
            lines.append(m)
        if len(missing) > 30:
            lines.append(f"... and {len(missing) - 30} more")
        lines.append("")

    if parse_fail:
        lines.append("=== Parse failures (first 30) ===")
        for name, err in parse_fail[:30]:
            lines.append(f"{name}: {err}")
        if len(parse_fail) > 30:
            lines.append(f"... and {len(parse_fail) - 30} more")
        lines.append("")

    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote report to {OUT_LOG.resolve()}")


if __name__ == "__main__":
    main()