import faster_data as faster
from multiprocessing import Process, Pool
import os
import argparse
import logging

def read_dt(d):
    dt = faster.FastData(d)
    logger = logging.getLogger(name=dt.instance_uid)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler('log/rirs2.log', mode='a') ## 파일 핸들러 생성
    file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
    logger.addHandler(file_handler)
    prev = dt.dist
    dt.log = False
    count = 0
    logger.info('File loaded')
    while True:
        dt.random_new_center()
        if dt.dist < prev:
            logger.info(f"Improved: {prev} -> {dt.dist}")
            dt.WriteData()
            prev = dt.dist
        count += 1
        if dt.dist >= prev * 1.1:
            dt = faster.FastData(d)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    inp = args.data
    print(inp)
    if "json" in inp:
        read_dt(inp)
    else:
        inp_list = os.listdir(inp)
        # json_list.reverse()
        json_list = []
        for inp1 in inp_list:
            if "json" not in inp1:
                continue
            if "rirs" not in inp1:
                continue
            json_list.append(os.path.join(inp,inp1))

            # if "-20-" in inp1:
            #     continue
            # rirs_list.append(os.path.join(inp,inp1))
            # dt = Data(os.path.join(inp,inp1))
        assert(len(json_list) == 49)
        pool = Pool(49)
        pool.map(read_dt, json_list)
