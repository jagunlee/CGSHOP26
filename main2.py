import sys
import os
from data import *
import time
import csv

def mySort(s : str): 
    if 'woc' in s: 
        return int(s.split('-')[1]) * 5
    elif 'random' in s:
        return int(s.split('_')[4]) * int(s.split('_')[5].split('.')[0])
    elif 'rirs' in s:
        return int(s.split('-')[1]) * int(s.split('-')[2])
    else:
        raise Exception('instance name invalid')
        
if __name__=="__main__":
    
    argument = sys.argv 

    start = time.time()

    instances = []

    userInput = False

    if len(argument)==2:
        instances = ['data/benchmark_instances/' + argument[1]]

    else:
        for file in os.listdir(os.path.dirname(__file__) + '/data/benchmark_instances'):
            if 'pdf' == file[len(file)-3:]:
                pass
            else:
                instances.append('data/benchmark_instances/'+file)

        instances.sort(key = mySort)
        if len(argument)==3:
            instances=instances[int(argument[1]):int(argument[2])+1]
        elif len(argument)==1:
            pass
        else:
            raise Exception('input arguments invalid')
    # print('time:', f"{end - start:.5f} sec")

    # debug

    # print(instances)
    # (key=lambda x : int(x.split('_')[-2]) * int(x.split('_')[-1].split('.')[0]))

    # sys.exit()

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
        
        centerT = D.findCenterGlobal2()
    
        D.WriteData()

        # D.verify()

        end = time.time()
        print('total time:', f"{end - start:.5f} sec")

    '''
    # 결과 작성
    f = open("result.csv", "a")
    # for instance in instances:
    wr = csv.writer(f)
    for instanceResult in result:
        wr.writerow(instanceResult)
    f.close()
    '''