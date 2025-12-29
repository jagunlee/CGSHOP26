from data import *

if __name__=="__main__":
    argument = sys.argv
    if len(argument)>=2:
        inp = argument[1]
    else:
        # inp = "data/examples/example_ps_20_nt2_pfd5_random.json"
        inp = "opt/rirs-1500-20-abcb179b.solution.json"
    if "json" in inp:
        start = time.time()
        dt = Data(inp)
    else:
        json_list = os.listdir(inp)
        json_list.reverse()
        sol_list = os.listdir("opt")
        rirs_list = []
        for inp1 in json_list:
            if "json" not in inp1:
                continue
            # if "-20-" in inp1:
            #     continue
            # rirs_list.append(os.path.join(inp,inp1))
            dt = Data(os.path.join(inp,inp1))