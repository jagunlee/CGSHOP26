import data as parallel
import os
import argparse

def read_dt(d, fcgs, fcgp1, fcgp2, rp, workers, chunk_size):
    dt = parallel.FastData(d)
    prev = dt.dist
    print("Initial dist = ", prev)
    while True:
        dt.random_new_center(fcgs, fcgp1, fcgp2, rp, workers, chunk_size)
        if dt.dist < prev:
            print(f"[{dt.instance_uid}] Improved: {prev} -> {dt.dist}")
            dt.WriteData()
            prev = dt.dist
            dt.pfd_distribution()
        if dt.dist >= prev*1.01:
            dt = parallel.FastData(d)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='solutions/[instance].solution.json')
    parser.add_argument('--fcg_serial', type=str, default=False, help='serial findCenterGlobal()')
    parser.add_argument('--fcg_pr1', type=str, default=False, help='first (parallel/serial) part in findCenterGlobal()')
    parser.add_argument('--fcg_pr2', type=str, default=False, help='second (parallel/serial) part in findCenterGlobal()')
    parser.add_argument('--replace_pr', type=str, default=False, help='parallel random_compute_pfd_replace()')
    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--ch_size', type=int, default=1)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    inp = args.data
    fcgs=args.fcg_serial
    fcgp1=args.fcg_pr1
    fcgp2=args.fcg_pr2
    rp=args.replace_pr
    workers=int(args.cpus)
    chunk_size=int(args.ch_size)

    fcgs = True if fcgs=='t' else False
    fcgp1 = True if fcgp1=='t' else False
    fcgp2 = True if fcgp2=='t' else False
    rp = True if rp=='t' else False

    if "json" in inp:
        read_dt(inp, fcgs, fcgp1, fcgp2, rp, workers, chunk_size)
