import sys
import os
from data import *
import time
import csv
import argparse

def mySort(s : str): 
    if 'woc' in s: 
        return int(s.split('-')[1]) * 5
    elif 'random' in s:
        return int(s.split('_')[4]) * int(s.split('_')[5].split('.')[0])
    elif 'rirs' in s:
        return int(s.split('-')[1]) * int(s.split('-')[2])
    else:
        raise Exception('instance name invalid')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=0)
    parser.add_argument('-l', type=str)
    return parser
        
if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    start = time.time()
    instances = []
    userInput = False

    for file in os.listdir(os.path.dirname(__file__) + '/data/benchmark_instances'):
        if 'pdf' == file[len(file)-3:]:
            pass
        else:
            instances.append('data/benchmark_instances/'+file)

    instances.sort(key = mySort)
    instances = [instances[int(args.n)]]
    log = open('log/'+args.l, 'w')
    sys.stdout = log

    result = []

    for instance in instances:
        D = Data(instance)
        print('instance', instance, 'read')
        end = time.time()
        print('total time:', f"{end - start:.5f} sec")
        centerT = D.findCenterGlobal()
        D.WriteData()
        end = time.time()
        print('total time:', f"{end - start:.5f} sec")
