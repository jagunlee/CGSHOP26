import os
import sys
import json
import datetime
import pandas as pd

if __name__=="__main__":
    
    # arguments = sys.argv 
    # if arguments[1][0] == "/": target = arguments[1]
    # else: target = os.path.dirname(__file__) + "/" + arguments[1]
    target = "/home/huntrix/CGSHOP26/opt"
    for file in os.listdir(target):
        nf = open(target+"/"+file, "r", encoding="utf-8")
        nroot = json.load(nf)
        instance = nroot["instance_uid"]
        try:
            of = open("opt/"+instance+".solution.json", "r", encoding="utf-8")
            oroot = json.load(of)
            if oroot["meta"]["dist"] > nroot["meta"]["dist"]:
                os.remove("opt/"+instance+".solution.json")
                with open("opt/"+instance+".solution"+".json", "w", encoding="utf-8") as f:
                    json.dump(nroot, f, indent='\t')
                print("updated", instance, oroot["meta"]["dist"], "->", nroot["meta"]["dist"])
        except:
            with open("opt/"+instance+".solution"+".json","w", encoding="utf-8") as f: 
                json.dump(nroot, f, indent='\t')
            print("updated",instance,"inf ->",nroot["meta"]["dist"])
        fname = "result.csv"

        if not os.path.exists(fname):
            df_dict = dict()
            df_dict["date"] = datetime.date.today()
            df_dict[instance] = [nroot["meta"]["dist"]]
            df = pd.DataFrame(df_dict)
            df.to_csv("result.csv")

        else:
            df = pd.read_csv(fname, index_col = 0)
            col = df.columns
            if instance not in col:
                df[instance] = float("INF")
            today = datetime.date.today().isoformat()
            # pdb.set_trace()
            if df["date"].iloc[-1]!=today:
                df.loc[len(df)] = list(df.iloc[-1])
                df.loc[len(df.index)-1, "date"] = today

            df.loc[len(df.index)-1, instance] = min(df.loc[len(df.index)-1, instance], nroot["meta"]["dist"])
            df.to_csv("result.csv")
            pass
