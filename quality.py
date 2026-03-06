# analyze_ratio_log.py
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

LOG_PATH  = Path("./logs/solve_from_coreset-260119.log")
BENCH_DIR = Path("./data/benchmark_instances")
OUT_LOG   = Path("output.log")

# log parsing
SEP_RE   = re.compile(r"^-{5,}\s*$")
FILE_RE  = re.compile(r"^\[([^\]]+)\]\s*$")
RATIO_RE = re.compile(r"^ratio\(ours\s*/\s*best\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def tri_count(p: Path) -> int:
    inst = load_json(p)
    tris = inst.get("triangulations", [])
    return len(tris) if isinstance(tris, list) else 0


def coreset_to_original_name(fname: str) -> str:
    """
    Map 'xxx_coreset.json' -> 'xxx.json'
    (Only apply when suffix exists; otherwise keep as-is but with .json)
    """
    name = Path(fname).name
    if name.endswith("_coreset.json"):
        return name.replace("_coreset.json", ".json")
    # just in case other variants appear
    if name.endswith("-coreset.json"):
        return name.replace("-coreset.json", ".json")
    if name.endswith("_coreset"):
        return name[:-7] + ".json"
    return Path(name).with_suffix(".json").name


def parse_log_entries(text: str):
    """
    Returns list of (coreset_filename, ratio)
    """
    lines = text.splitlines()
    blocks, cur = [], []

    for ln in lines:
        if SEP_RE.match(ln.strip()):
            if cur:
                blocks.append(cur)
                cur = []
        else:
            if ln.strip() != "":
                cur.append(ln.rstrip("\n"))
    if cur:
        blocks.append(cur)

    entries = []
    for blk in blocks:
        fname, ratio = None, None
        for ln in blk:
            m = FILE_RE.match(ln.strip())
            if m:
                fname = m.group(1).strip()
                continue
            m = RATIO_RE.match(ln.strip())
            if m:
                ratio = float(m.group(1))
                continue
        if fname is not None and ratio is not None:
            entries.append((fname, ratio))
    return entries


def summarize(pairs):
    """
    pairs: list of (ratio, orig_n)
    """
    if not pairs:
        return None
    rs = [r for r, _ in pairs]
    return {
        "count": len(rs),
        "mean": mean(rs),
        "median": median(rs),
        "min": min(rs),
        "max": max(rs),
    }


def main():
    if not LOG_PATH.exists():
        OUT_LOG.write_text(f"[ERROR] log not found: {LOG_PATH.resolve()}\n", encoding="utf-8")
        print(f"[ERROR] log not found: {LOG_PATH.resolve()}")
        return
    if not BENCH_DIR.exists():
        OUT_LOG.write_text(f"[ERROR] benchmark dir not found: {BENCH_DIR.resolve()}\n", encoding="utf-8")
        print(f"[ERROR] benchmark dir not found: {BENCH_DIR.resolve()}")
        return

    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    entries = parse_log_entries(text)

    matched = []    # (ratio, orig_n, coreset_file, orig_file)
    missing = []    # (coreset_file, expected_orig_file)
    read_fail = []  # (orig_file, error)

    for coreset_file, ratio in entries:
        orig_name = coreset_to_original_name(coreset_file)
        orig_path = BENCH_DIR / orig_name

        if not orig_path.exists():
            missing.append((Path(coreset_file).name, orig_name))
            continue

        try:
            n = tri_count(orig_path)
        except Exception as e:
            read_fail.append((orig_name, str(e)))
            continue

        matched.append((ratio, n, Path(coreset_file).name, orig_name))

    overall_pairs = [(r, n) for (r, n, _, _) in matched]
    le10_pairs    = [(r, n) for (r, n, _, _) in matched if n <= 10]
    gt10_pairs    = [(r, n) for (r, n, _, _) in matched if n > 10]

    overall = summarize(overall_pairs)
    le10 = summarize(le10_pairs)
    gt10 = summarize(gt10_pairs)

    by_n = defaultdict(list)
    for r, n, _, _ in matched:
        by_n[n].append(r)

    lines = []
    lines.append("=== Ratio statistics (ours/best) grouped by ORIGINAL triangulation count ===")
    lines.append(f"log_file   : {LOG_PATH.resolve()}")
    lines.append(f"bench_dir  : {BENCH_DIR.resolve()}")
    lines.append(f"parsed entries (with ratio): {len(entries)}")
    lines.append(f"matched to benchmark .json : {len(matched)}")
    lines.append(f"missing benchmark .json    : {len(missing)}")
    lines.append(f"benchmark read failures    : {len(read_fail)}")
    lines.append("")

    def fmt(title, s):
        if s is None:
            lines.append(f"{title}: (no data)")
        else:
            lines.append(
                f"{title}: #={s['count']}, mean={s['mean']:.6f}, median={s['median']:.6f}, "
                f"min={s['min']:.6f}, max={s['max']:.6f}"
            )

    fmt("Overall", overall)
    fmt("orig_n <= 10", le10)
    fmt("orig_n >  10", gt10)
    if le10 and gt10:
        lines.append(f"Mean gap (orig_n>10 - orig_n<=10) = {(gt10['mean'] - le10['mean']):.6f}")
    lines.append("")

    lines.append("=== Mean ratio by original triangulation count (orig_n) ===")
    lines.append("orig_n | #inst | mean_ratio | min_ratio | max_ratio")
    for n in sorted(by_n.keys()):
        rs = by_n[n]
        lines.append(f"{n:6d} | {len(rs):5d} | {mean(rs):9.6f} | {min(rs):9.6f} | {max(rs):9.6f}")
    lines.append("")

    if missing:
        lines.append("=== Missing benchmark files (first 30) ===")
        for coreset_name, expected_orig in missing[:30]:
            lines.append(f"{coreset_name}  -> expected {expected_orig} (not found)")
        if len(missing) > 30:
            lines.append(f"... and {len(missing) - 30} more")
        lines.append("")

    if read_fail:
        lines.append("=== Benchmark read failures (first 30) ===")
        for orig_name, err in read_fail[:30]:
            lines.append(f"{orig_name}: {err}")
        if len(read_fail) > 30:
            lines.append(f"... and {len(read_fail) - 30} more")
        lines.append("")

    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote report to {OUT_LOG.resolve()}")


if __name__ == "__main__":
    main()