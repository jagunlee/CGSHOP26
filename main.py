import sys
import os
from data import *
import time

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "data/benchmark_instances/random_instance_4_40_2.json"
    
    D = Data('random_instance_36_80_3.json')
    # D = Data('random_instance_90_320_5.json')

    # T1 -> T2와 T2 -> T1도 차이 많이 나나?
    # 지금 parallel이 아예 들어가있지 않음.

    # pairwise distance 비교
    for i in range(len(D.triangulations)):
        for j in range(i+1, len(D.triangulations)):
            pfp = D.parallel_flip_path(D.triangulations[i], D.triangulations[j])
            print('parallel flip distance from T', i, 'to T', j, ':', len(pfp)) # , 'pfp:', pfp)
            
            # print('distance from T', i, 'to T', j, ':', len(D.flip_sequence(D.triangulations[i], D.triangulations[j])[0]))
            
    print()

    centerT = D.parallel_flip_path_all()
    for i in range(len(D.triangulations)):
        pfp = D.parallel_flip_path(centerT, D.triangulations[i])
        print('parallel flip distance from the center to T', i, ':', len(pfp))
        
        # print('distance from the center to T', i, ':', len(D.flip_sequence(centerT, D.triangulations[i])[0]))
        # opencv