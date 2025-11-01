import os
import pathlib

if __name__=="__main__":
    '''
    name =  os.path.dirname(os.getcwd())
    dir = pathlib.Path(name)
    pdir = dir.parent
    print(dir)
    print(pdir)

    print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    print(os.listdir(os.path.dirname(__file__) + '\\data\\benchmark_instances'))
    '''

    
    for file in os.listdir(os.path.dirname(__file__) + '\\data\\benchmark_instances'):
        if 'pdf' == file[len(file)-3:]:
            print('pdf')
        else:
            print('json')