
from concurrent.futures import ProcessPoolExecutor, as_completed

def init_worker(my_list):
    global gb_list
    gb_list = my_list


def test(idx):
    global gb_list
    gb_list[idx]=0
    return idx, gb_list

if __name__=="__main__":
    my_list =[1,2,3,4,5,6,7,8,9,10]
    with ProcessPoolExecutor(initializer=init_worker, initargs=(my_list,),max_workers=3) as exe:
        future=[]
        for i in range(10):
            f = exe.submit(test, i)
            future.append(f)
        for f in as_completed(future):
            print(f.result())
    print("done")


