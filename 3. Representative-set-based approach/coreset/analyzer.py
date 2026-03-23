import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

# Log Parsing Regex (from your quality.py)
SEP_RE   = re.compile(r"^-{5,}\s*$")
FILE_RE  = re.compile(r"^\[([^\]]+)\]\s*$")
RATIO_RE = re.compile(r"^ratio\(ours\s*/\s*best\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")

# Helpers
def load_json(p: Path) -> dict:
    """Loads a JSON dictionary safely."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def tri_count(p: Path) -> int:
    """Returns the number of triangulations in a JSON instance."""
    try:
        inst = load_json(p)
        tris = inst.get("triangulations", [])
        return len(tris) if isinstance(tris, list) else 0
    except Exception:
        return 0

def coreset_to_original_name(fname: str) -> str:
    """Maps coreset filename to original benchmark filename (e.g., xxx_coreset.json -> xxx.json)."""
    name = Path(fname).name
    if name.endswith("_coreset.json"):
        return name.replace("_coreset.json", ".json")
    if name.endswith("-coreset.json"):
        return name.replace("-coreset.json", ".json")
    if name.endswith("_coreset"):
        return name[:-7] + ".json"
    return Path(name).with_suffix(".json").name

def summarize(pairs):
    """Calculates statistics for a list of (ratio, orig_n) pairs."""
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

# Size Analysis (Reduction Analysis)
def analyze_size(bench_dir: Path, core_dir: Path, out_log: Path, exclude_rirs: bool):
    """Analyzes the reduction in triangulation count between original and coreset files."""
    orig_files = sorted(bench_dir.glob("*.json"))
    records = []

    for orig_path in orig_files:
        if exclude_rirs and orig_path.stem.lower().startswith("rirs"):
            continue

        core_path = core_dir / f"{orig_path.stem}_coreset.json"
        if not core_path.exists():
            continue

        orig_n = tri_count(orig_path)
        core_n = tri_count(core_path)
        if orig_n <= 0: continue

        reduction = (orig_n - core_n) / orig_n * 100.0
        records.append((orig_path.name, orig_n, core_n, reduction))

    if not records:
        print("[WARN] No matched instances for size analysis.")
        return

    lines = ["=== Coreset Size Reduction Report ==="]
    avg_red = mean([r[3] for r in records])
    avg_orig = mean([r[1] for r in records])
    avg_core = mean([r[2] for r in records])

    lines.append(f"Overall: #inst={len(records)}, Avg Reduction={avg_red:.2f}%, Avg Size: {avg_orig:.1f} -> {avg_core:.1f}")
    lines.append("\norig_n | #inst | avg_core_n | avg_reduction(%)")

    by_n = defaultdict(list)
    for _, o_n, c_n, red in records:
        by_n[o_n].append((c_n, red))

    for o_n in sorted(by_n.keys()):
        c_ns, reds = zip(*by_n[o_n])
        lines.append(f"{o_n:6d} | {len(reds):5d} | {mean(c_ns):10.2f} | {mean(reds):15.2f}%")

    report_text = "\n".join(lines) + "\n"
    print(report_text)
    if out_log:
        with open(out_log, "a", encoding="utf-8") as f:
            f.write(report_text)


# Quality Analysis (Log Parsing Logic)
def parse_log_entries(text: str):
    """Parses solver log blocks to extract coreset filenames and their corresponding ratios."""
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

def analyze_quality(bench_dir: Path, log_path: Path, out_log: Path, exclude_rirs: bool):
    """Processes the solver log and benchmark JSONs to generate a quality report."""
    if not log_path.exists():
        print(f"[ERROR] Solver log not found: {log_path}")
        return

    text = log_path.read_text(encoding="utf-8", errors="replace")
    entries = parse_log_entries(text)

    matched = []
    missing = []
    read_fail = []

    for coreset_file, ratio in entries:
        orig_name = coreset_to_original_name(coreset_file)
        if exclude_rirs and orig_name.lower().startswith("rirs"):
            continue

        orig_path = bench_dir / orig_name
        if not orig_path.exists():
            missing.append((coreset_file, orig_name))
            continue

        try:
            n = tri_count(orig_path)
            matched.append((ratio, n, coreset_file, orig_name))
        except Exception as e:
            read_fail.append((orig_name, str(e)))

    if not matched:
        print("[WARN] No valid matched entries found for quality analysis.")
        return

    overall = summarize([(r, n) for r, n, _, _ in matched])
    le10 = summarize([(r, n) for r, n, _, _ in matched if n <= 10])
    gt10 = summarize([(r, n) for r, n, _, _ in matched if n > 10])

    lines = ["\n=== Quality (Ratio = Ours / Best) Report ==="]
    lines.append(f"Source Log: {log_path.name}")

    def fmt_line(title, s):
        if s:
            lines.append(f"{title:<12}: #={s['count']}, mean={s['mean']:.6f}, median={s['median']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")

    fmt_line("Overall", overall)
    fmt_line("orig_n <= 10", le10)
    fmt_line("orig_n > 10", gt10)

    if le10 and gt10:
        lines.append(f"Mean gap (orig_n>10 - orig_n<=10) = {(gt10['mean'] - le10['mean']):.6f}")

    lines.append("\n=== Mean ratio by original triangulation count (orig_n) ===")
    lines.append("orig_n | #inst | mean_ratio | min_ratio | max_ratio")

    by_n = defaultdict(list)
    for r, n, _, _ in matched:
        by_n[n].append(r)

    for n in sorted(by_n.keys()):
        rs = by_n[n]
        lines.append(f"{n:6d} | {len(rs):5d} | {mean(rs):9.6f} | {min(rs):9.6f} | {max(rs):9.6f}")

    report_text = "\n".join(lines) + "\n"
    print(report_text)
    if out_log:
        with open(out_log, "a", encoding="utf-8") as f:
            f.write(report_text)


# Main Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze coreset size and quality based on solver logs.")
    parser.add_argument("-b", "--bench_dir", default="data/benchmark_instances", help="Original benchmark directory")
    parser.add_argument("-c", "--core_dir", default="data/coreset_instances", help="Coreset directory")
    parser.add_argument("-l", "--log_path", default="coreset/logs/solve_global.log", help="Path to the solver log file")
    parser.add_argument("-o", "--out_log", default="coreset/logs/coreset_analysis.log", help="Output report file")
    parser.add_argument("-r", "--exclude_rirs", action="store_true", help="Exclude RIRS instances")
    parser.add_argument("-m", "--mode", choices=["size", "quality", "both"], default="both", help="Analysis mode")

    args = parser.parse_args()
    bench = Path(args.bench_dir)
    core = Path(args.core_dir)
    log_p = Path(args.log_path)
    out_p = Path(args.out_log)

    # Ensure output directory exists
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ["size", "both"]:
        print("\n" + "="*60)
        analyze_size(bench, core, out_p, args.exclude_rirs)

    if args.mode in ["quality", "both"]:
        print("\n" + "="*60)
        analyze_quality(bench, log_p, out_p, args.exclude_rirs)

    print(f"\n[DONE] Report written to {out_p.resolve()}")
