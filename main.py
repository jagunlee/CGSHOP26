import sys
from data import *

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "data/examples/example_ps_200_nt16_pfd40_tsplib.json"
    dt = Data(inp)
    dt.find_center()