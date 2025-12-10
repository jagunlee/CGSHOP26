obj ='6.7.2.9.6.8.2.8.,0.6.7.13.2.4.,.,'
flips=[]
print(obj)
pll_flips = obj.split(',')
for pll in obj.split(','):
    int_nodes=[]
    if pll =='': continue
    str_nodes = pll.split('.')
    del str_nodes[-1]
    edge=[]
    print("str_nodes = ", str_nodes)
    if len(str_nodes)%2==1 or str_nodes==[] or '' in str_nodes:
        continue
    for n in str_nodes:
        if len(edge)<2:
            edge.append(int(n))
        elif len(edge)==2:
            int_nodes.append(edge)
            edge=[]
            edge.append(int(n))
    int_nodes.append(edge)
    flips.append(int_nodes)

# compute CenterT from T_0
print(flips)
