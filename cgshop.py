
from hy_data import *

import numpy as np
import argparse
from utils import initialize_exp

class Database:
    def __init__(self, objects:dict, rewards:dict):
        self.objects={}
        self.rewards={}


def find_next_available_filename(write_path, base, extension):
    i=1
    while True:
        filename = f"./{write_path}/{base}_{i}.{extension}"
        if os.path.isfile(filename) == False:
            return filename
        i+=1

def write_output_to_file(db, write_path):
    final_database_size = 50
    sorted_score = sorted(db.rewards)
    base_name = "search_output"
    extension = "txt"
    filename = find_next_available_filename(write_path, base_name, extension)
    curr_rew_index = 0
    lines_written = 0
    with open(filename, "w") as file:
        print(filename)
        while lines_written < final_database_size and curr_rew_index < len(sorted_score):
            curr_rew = sorted_score[curr_rew_index]
            for obj in db.rewards[curr_rew][0:min(final_database_size - lines_written, len(db.rewards[curr_rew]))]:
                file.write(obj + "\n")
            lines_written += len(db.rewards[curr_rew])
            curr_rew_index +=1
    file.close()
    print(f"Data written to {filename}")
    print(f"An example of an object with maximum reward ({str(sorted_score[0])}):")
    print(db.rewards[sorted_score[0]][0])

    #print("Read file..")
    #with open(filename, "r") as file:
    #    print(file.read())
    #file.close()





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
    #for sc in sorted_score:
    #    v = db.rewards[sc]
    #    print(f"score {sc}: {len(v)}")
    db_size =0
    for r in sorted_score:
        db_size += len(db.rewards[r]) # number of objects with same reward
    # shrink database if necessary
    #if db_size > 2*target_db_size: #target_db_size: size of caache during local search loop, should be larger than training set size



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
    center.random_flip(10) # Need to modify!!! It produces not flippable edge.

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

def local_search(db, path, input_file):
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
        obj, rew = local_search_on_object(db, dt, center)
        single_thread_obj.append(obj)
        single_thread_rew.append(rew)

    # add_db! part
    add_db(db, single_thread_obj, single_thread_rew)

    # print_db part
    #my_print_db(db)

    # Write search_output.txt file
    write_output_to_file(db, path)
    # Write plot file

def get_parser():
    # For mkdirs in checkpoint/debug/
    parser.add_argument('--max_epochs', type=int, default= 10, help='number of epochs')
    parser = argparse.ArgumentParser('Generate training sample of low braids via reservoir sampling')
    parser.add_argument("--dump_path", type=str, default="checkpoint",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    #dump_path
    #command
    #exp_name
    #pkl
    #exp_id
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    Tris_path = './data/benchmark_instances/'
    input_file = 'random_instance_110_15_3.json'



    db = Database({},{})
    for i in range(1, args.max_epochs):
        if not os.path.isfile(f"{args.dump_path}/search_output_{i}-tokenized.txt"):
            break
    initial_gen = i-1
    if initial_gen ==0:
        print("from taehoon_hwi center file...")
        local_search(db, args.dump_path, input_file)
        tokenize(f"{args.dump_path}/search_output_1.txt", args.n_tokens)
        initial_gen = 1

    print()
    print("from pohang center file...")
    input_file = 'random_instance_110_15_3.solution.json'
    local_search(db, args.dump_path, input_file)

