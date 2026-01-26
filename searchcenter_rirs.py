from faster_data import *
from multiprocessing import Process, Pool
import os
import logging
import sys

def read_dt(d):
    dt = FastData(d)
    logger = logging.getLogger(name=dt.instance_uid)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler('log/rirs_faster.log', mode='a') ## 파일 핸들러 생성
    file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
    logger.addHandler(file_handler)
    prev = dt.dist
    dt.log = False
    count = 0
    logger.info('File loaded')
    try:
        while True:
            dt.random_new_center()
            if dt.dist < prev:
                logger.info(f"Improved: {prev} -> {dt.dist}")
                dt.WriteData()
                prev = dt.dist
            count += 1
            if dt.dist >= prev * 1.01:
                dt = FastData(d)
    except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
        logger.info(f"Exception occured: {e}")
        

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "opt/rirs-500-20-5e21448d.solution.json"
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
        # print(json_list)
        assert(len(json_list) == 49)
        pool = Pool(49)
        pool.map(read_dt, json_list)
