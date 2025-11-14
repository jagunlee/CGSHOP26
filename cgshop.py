
from hy_data import *

import numpy as np

class Database:
    def __init__(self, objects:dict, rewards:dict):
        self.objects={}
        self.rewards={}


def read_center(input_file):
    Tris_path = './data/benchmark_instances/'
    # Read a Center C
    with open(Tris_path + input_file, 'r') as f:
        root = json.load(f)
        pts_x = root["points_x"]
        pts_y = root["points_y"]
        pts=[]
        for i in range(len(pts_y)):
            pts.append(Point(pts_x[i], pts_y[i]))

    Center_path = './centers/'
    input_file = root["instance_uid"]+'.json'
    with open(Center_path + input_file, 'r') as f:
        C_edges = json.load(f)
    center = Triangulation(pts, C_edges)

    # Read Triangulations Ts
    dt = Data(Tris_path + input_file)
    return center, dt

def reward(db, obj):
    if obj in db.objects:
        return False
    else: return True

def add_db(db, list_obj, list_rew):
    for i in range(len(list_obj)):
        obj = list_obj[i][0]
        if obj not in db.objects:
            rew = list_rew[i][0]
            db.objects[obj] = rew
            if rew not in db.rewards:
                db.rewards[rew] = [obj]
            else:
                db.rewards[rew].append(obj)

def my_print_db(db):
    sorted_score = sorted(db.rewards)
    print(sorted_score)
    for sc in sorted_score:
        v = db.rewards[sc]
        print(f"score {sc}: {len(v)}")
    #for k, v in db.rewards.items():
    #    print(f"score {k}: {len(v)}")


def convert_to_string(center):
    # center -> adjmat
    edges = list(center.edges.keys())
    N = len(center.pts)
    adjmat = np.zeros((N,N))
    for i,j in edges:
        adjmat[i,j]=int(1)
        adjmat[j,i]=int(1)

    # adjmat -> string
    entries = []
    for i in range(N-1):
        for j in range(i+1, N):
            entries.append(str(int(adjmat[i,j])))
        entries.append(",")
    return "".join(entries)

def local_search_on_object(db, dt, center):
    objects=[]
    rewards=[]
    origin_center = center
    # Perturbe C
    center.random_flip(10)

    # Compute pfd(T,C), save it as a rewards for sorting
    dist, _ = dt.compute_center_dist(center)
    _, dist2 = dt.random_move() # too long
    if dist2 < dist:
        dist = dist2

    rew = dist
    obj = convert_to_string(center)
    new = reward(db, obj)
    if new:
       objects.append(obj)
       rewards.append(rew)
    return objects, rewards

    #center = origin_center #hy: maybe not helpful?
    # Save perturbed Cs in 'search_output_{i}.txt'

def local_search(db, input_file):
    if 'solution' not in input_file:
        center, dt = read_center(input_file) # './centers/~'
    elif 'solution' in input_file:
        with open('./solutions/'+ input_file, 'r') as f:
            root = json.load(f)
        dt = Data(root["meta"]["input"])
        center = dt.center
    #elif 'decoded' in input_file:
        # Read 'transformer-output-decoded.txt' and save it in C
    single_thread_obj=[]
    single_thread_rew =[]
    for i in range(2):
       # print(f"{i}th local_search_on_object")
        obj, rew = local_search_on_object(db, dt, center)
        single_thread_obj.append(obj)
        single_thread_rew.append(rew)

    # now add_db! part
    add_db(db, single_thread_obj, single_thread_rew)
    my_print_db(db)
    # print_db part


if __name__ == '__main__':
    Tris_path = './data/benchmark_instances/'
    input_file = 'random_instance_110_15_3.json'

    db = Database({},{})
    print("from taehoon_hwi center file...")
    local_search(db, input_file)

    print()
    print("from pohang center file...")
    input_file = 'random_instance_110_15_3.solution.json'
    local_search(db, input_file)

