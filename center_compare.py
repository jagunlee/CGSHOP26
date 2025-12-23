import numpy as np
import th2_data as th2
import copy

if __name__ == '__main__':
    Tris_path = './data/benchmark_instances/'
    inst_file = 'random_instance_444_15_10'
    dt = th2.Data(Tris_path + inst_file + '.json')

    path='./checkpoint/debug/zdiedbo7mk'
    print(inst_file)
    for i in range(1,6):
        lines=[]
        sum_E = dict()
        dSum= dict()
        with open(path + f'/search_output_{i}.txt',"r") as f:
            for line in f:
                lines.append(line.strip())
        f.close()
        print(f"\nin search_output_{i}.txt -------------how many objs? = {len(lines)}")


        for obj in lines:
            # string flip -> int list flip
            str_nodes = obj.split('.')
            del str_nodes[-1]
            flips=[]
            edge=[]
            for n in str_nodes:
                if n=='': continue
                if len(edge)<2:
                    edge.append(int(n))
                else:
                    flips.append(edge)
                    edge=[]
                    edge.append(int(n))
            if len(edge)==2:
                flips.append(edge)

            tmp = copy.deepcopy(dt.triangulations[0])
            for edge in flips:
                if dt.flippable(tmp, tuple(edge)):
                    tmp.flip(edge)
            centerT = tmp

            all_pFlips = dt.computeDistanceSum(centerT)
            Dsum = sum([len(f) for f in all_pFlips])

            if Dsum not in sum_E:
                sum_E[Dsum]=[centerT.edges]
                dSum[Dsum]=1
            elif Dsum in sum_E:
                dSum[Dsum]+=1
                new_edges = False
                for prev_edges in sum_E[Dsum]:
                    if prev_edges == centerT.edges:
                        new_edges = False
                        break
                    else:
                        new_edges = True
                if new_edges: sum_E[Dsum].append(centerT.edges)
        print("Dsum: unique centers / total centers")
        for Dsum in sum_E.keys():
            print(Dsum, ": ",  len(sum_E[Dsum]), "/", dSum[Dsum])




#   if prev_center==None:
#       prev_center = copy.deepcopy(centerT)
#       prev_Dsum = Dsum
#       prev_E.append(prev_center.edges)
#
#
#   if prev_Dsum == Dsum:
#       # compare prev_center & centerT
#       curr_edges = centerT.edges
#       print("Dsum=", Dsum, " and this center is same with prev centers? ")
#       same=[]
#       diff=[]
#       for i, prev_edges in enumerate(prev_E):
#           if curr_edges != prev_edges:
#               prev_E.append(curr_edges)
#               diff.append(i)
#           else:
#               same.append(i)
#       print("same with prev: ", same, " , different from prev: ", diff)
#   else:
#       print("Dsum = ", Dsum)
#       prev_center = copy.deepcopy(centerT)
#       prev_E=[]
#       prev_E.append(prev_center.edges)

