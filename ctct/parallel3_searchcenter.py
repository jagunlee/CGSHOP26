import working3_opt_p_data as parallel
#import working2_opt_p_data as parallel
import os
import argparse

def read_dt(d):
    dt = parallel.FastData(d)
    prev = dt.dist
    print("initial dist = ", prev)
    count = 0
    while dt.dist < prev * 1.1:
        dt.random_new_center()
        count += 1
        if dt.dist < prev:
            print(f"[{dt.instance_uid}] Improved: {prev} -> {dt.dist}")
            dt.WriteData()
            prev = dt.dist
        print("-------------------------")
    print("Done")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    inp = args.data
    if "json" in inp:
        read_dt(inp)
    #else:
    #    inp_list = os.listdir(inp)
    #    json_list = []
    #    for inp1 in inp_list:
    #        if "json" not in inp1:
    #            continue
    #        json_list.append(os.path.join(inp,inp1))
    #    print(json_list)
    #    pool = Pool(60)
    #    pool.map(read_dt, json_list)
