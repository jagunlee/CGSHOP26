import sys
import os
from data import *
import time
import csv

if __name__=="__main__":
    argument = sys.argv

    start = time.time()

    instances = []

    if len(argument)>=2:
        instances = ['data\\benchmark_instances\\' + argument[1]]

    else:
        for file in os.listdir(os.path.dirname(__file__) + '\\data\\benchmark_instances'):
            if 'pdf' == file[len(file)-3:]:
                pass
            else:
                instances.append('data\\benchmark_instances\\'+file)
    
    # print('time:', f"{end - start:.5f} sec")

    # debug
    # print(instances)

    # list of (instance_name, radius)
    result = []

    for instance in instances:

        D = Data(instance)

        print('instance', instance, 'read')

        end = time.time()
        print('total time:', f"{end - start:.5f} sec")

        '''
        # pairwise distance 비교
        for i in range(len(D.triangulations)):
            for j in range(i+1, len(D.triangulations)):

                pfp = D.parallel_flip_path(D.triangulations[i], D.triangulations[j])
                print('parallel flip distance from T', i, 'to T', j, ':', len(pfp)) # , 'pfp:', pfp)
                
                end = time.time()
                print('time:', f"{end - start:.5f} sec")

                # print('distance from T', i, 'to T', j, ':', len(D.flip_sequence(D.triangulations[i], D.triangulations[j])[0]))
        
        end = time.time()

        print()
        '''

        centerT = D.parallel_flip_path_all()
        for i in range(len(D.triangulations)):
            pfp = D.parallel_flip_path(centerT, D.triangulations[i])
            print('parallel flip distance from the center to T', i, ':', len(pfp))
            
            end = time.time()
            print('time:', f"{end - start:.5f} sec")

            # print('distance from the center to T', i, ':', len(D.flip_sequence(centerT, D.triangulations[i])[0]))
            # opencv

        end = time.time()
        print('total time:', f"{end - start:.5f} sec")

    # 결과 작성
    f = open("result.csv", "a")
    for instance in instances:
        wr = csv.writer(f)
        wr.writerow(result)
    f.close()