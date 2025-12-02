from jg_data import *

import numpy as np
import argparse
from utils import initialize_exp
from cgshop2026_pyutils.verify import check_for_errors
from cgshop2026_pyutils.schemas import CGSHOP2026Instance, CGSHOP2026Solution

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
from makemoretokens import CharDataset, ModelConfig, Transformer, InfiniteDataLoader, generate
from utils import bool_flag#, initialize_exp
#ModelConfig, CharDataset, Transformer, Bigram, MLP, RNN, BoW, InfiniteDataLoader, evaluate, generate


class Database:
    def __init__(self, objects:dict, rewards:dict, num_pts:int):
        self.objects={}
        self.rewards={}
        self.num_pts=num_pts


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

def local_search_on_object(db, dt, center, input_file):
    objects=[]
    rewards=[]
    origin_center = center
    # Perturbe C
    center.random_flip(20)

    # Compute pfd(T,C), save it as a rewards for sorting
    dist, flip = dt.compute_center_dist(center)
    #_, dist2, flip2 = dt.random_move() # too long time
    #if dist2 < dist:
    #    dist = dist2
    #    center.flip = flip2
    #else:
    #    center = origin_center
    #    center.flip = flip
    # error test
    Tris_path = './data/benchmark_instances/'
    with open(Tris_path+input_file, "r") as f:
        file = json.load(f)
        instance = CGSHOP2026Instance(
                instance_uid = file["instance_uid"],
                points_x = file["points_x"],
                points_y = file["points_y"],
                triangulations=file["triangulations"]
                )
        solution = CGSHOP2026Solution(
                instance_uid = file["instance_uid"],
                flips=flip
                )

    errors = check_for_errors(instance, solution)
    assert not errors, f"Errors found in solution: {errors}"

    #print("dist = ", dist)

    rew = dist
    obj = convert_to_string(center)
    new = reward(db, obj)
    if new:
       objects.append(obj)
       rewards.append(rew)
    return objects, rewards

    #center = origin_center #hy: maybe not helpful?
    # Save perturbed Cs in 'search_output_{i}.txt'


def local_search_from_decoded(db, inst_file, path, input_file):
    Tris_path = './data/benchmark_instances/'
    #if 'decoded' in input_file:
    lines=[]
    with open(path+input_file,"r") as file:
        for line in file:
            lines.append(line.strip())
    # Read 'transformer-output-decoded.txt' and save it in C
    # Read Triangulations Ts
    dt = Data(Tris_path + inst_file + '.json')
    N = db.num_pts
    for obj in lines:
        adjmat = np.zeros((N,N))
        index=0
        C_edges=[]
        for i in range(N-1):
            for j in range(i+1,N):
                while obj[index] ==',':
                    index+=1
                adjmat[i,j] = int(obj[index])
                adjmat[j,i] = adjmat[i,j]
                if adjmat[i,j] ==1:
                    C_edges.append([i,j])
        print("hy:  center = ")
        print(C_edges)

        # Remove intersecting edges
        # C_edges의 엣지를 하나하나 추가하면서 intersect 하는 부분 지우기
        valid_edge=[]
        random.shuffle(C_edges)
        for e in C_edges:
            if len(valid_edge)>0:
                for le in valid_edge:
                    if dt.intersect(e[0],e[1], le[0], le[1])==False:
                        valid_edge.append(le)
            else: valid_edge.append(e)
        C_edges = valid_edge
        print(list(C_edges))
        exit(0)
        # Complete Triangulation

        # Insert Hull edges <- maybe? done by make_triangulation()

        # convert_to_string(adjmat) <- This going to be written in search_output_{i+1}.txt for next tokenize



        center = Triangulation(dt.pts, C_edges)
        dt.center = center #???? it works? and.. it is needed?

        single_thread_obj=[]
        single_thread_rew =[]
        for i in range(20):
            obj, rew = local_search_on_object(db, dt, center, input_file)
            single_thread_obj.append(obj)
            single_thread_rew.append(rew)
        # add_db! part
        add_db(db, single_thread_obj, single_thread_rew)
        # Write search_output.txt file
        write_output_to_file(db, path)




def local_search(db, path, input_file):
    if 'solution' not in input_file and 'decoded' not in input_file:
        center, dt = read_center(input_file) # './centers/~'
    elif 'solution' in input_file:
        with open('./solutions/'+ input_file, 'r') as f:
            root = json.load(f)
        dt = Data(root["meta"]["input"])
        center = dt.center

    single_thread_obj=[]
    single_thread_rew =[]
    for i in range(20):
        obj, rew = local_search_on_object(db, dt, center, input_file)
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
    parser = argparse.ArgumentParser('Generate training sample of low braids via reservoir sampling')

    parser.add_argument('--sample-only', type=int, default=500, help="sample the specified number from the model in each loop")
    parser.add_argument('--max-steps', type=int, default=200, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument("--dump_path", type=str, default="checkpoint",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    parser.add_argument('--n-layer', type=int, default=2, help="number of layers")
    parser.add_argument('--n-head', type=int, default=8, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=32, help="number of feature channels elsewhere in the model")
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument('--max-output-length', type=int, default=160, help="maximum output length")
    parser.add_argument('--gen_batch_size', type=int, default=10, help="generation batch size")
    parser.add_argument('--n_tokens', type=int, default=100, help="nr tokens in tokenizer")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")


    parser.add_argument("--cpu", type=bool_flag, default="true",#hy
                        help="run on cpu only")

    #dump_path
    #command
    #exp_name
    #pkl
    #exp_id
    return parser



def tokenize(input_file_path, n_tokens):

    directory_name = args.dump_path + '/' + "tokenizer_data"
    tokenizer_file = directory_name + "/tokenizer.json"

    if os.path.exists(tokenizer_file):
        logger.info(f"Loading tokenizer from {tokenizer_file}...")
        tokenizer = Tokenizer.from_file(tokenizer_file)
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(vocab_size=n_tokens)

        source_file_path = args.dump_path+'/search_output_1.txt'
        destination_file_path = args.dump_path+"/temp.txt"

        logger.info(f'Created {destination_file_path} and training tokenizer...')
        # Reading the first 100,000 lines from the source file and training the tokenizer on them
        with open(source_file_path, 'r') as source_file, open(destination_file_path, 'w') as destination_file:
            for i in range(5_000):
                line = source_file.readline()
                if not line:
                    break
                destination_file.write(line)

        if not os.path.isdir(directory_name):
            # Create the directory
            os.mkdir(directory_name)
            logger.info(f"Directory '{directory_name}' created.")

        tokenizer.train([destination_file_path], trainer)
        tokenizer.save(tokenizer_file)

        if os.path.exists(destination_file_path):
            os.remove(destination_file_path)
            logger.info(f"File '{destination_file_path}' has been deleted.")

    # input_file_path = input_path
    with open(input_file_path, "r") as file:
        text_data = [line.strip() for line in file]

    # Now create tokenized output file
    token_file_out = input_file_path.rsplit('.', 1)[0] + '-tokenized.txt'
    with open(token_file_out, "w") as file:
        print("Tokenizing training set...")
        for i, sequence in enumerate(text_data):
            if i % 10000 == 0:
                logger.info(f"{i} / {len(text_data)}")
            myids = tokenizer.encode(sequence).ids
            file.write(','.join(["V" + str(id) for id in myids]))
            file.write("\n")


def decode():
    # Load the tokenizer from the saved file
    tokenizer_path = os.path.join(args.dump_path+'/tokenizer_data', "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        logger.error(f"No tokenizer found at {tokenizer_path}. Please check the path and try again.")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    def decode_tokens(token_line):
        # Remove the 'V' prefix and convert to integers
        #print(token_line)
        token_ids = [int(token[1:]) for token in token_line.split(',')]
        # Decode the token ids to text

        return tokenizer.decode(token_ids).replace(" ","")


    # Process the input file
    input_file = args.dump_path+"/out.txt"
    if os.path.exists(input_file):
        with open(input_file, 'r') as file:
            tokenized_lines = file.readlines()

        # Decode each line and collect the results
        decoded_text = [decode_tokens(line.strip()) for line in tokenized_lines if len(line) > 1]

        # Write the decoded text to the output file
        output_file = args.dump_path+"/transformer-output-decoded.txt"
        with open(output_file, 'w') as file:
            for line in decoded_text:
                file.write(line + '\n')

        logger.info(f"Decoding complet. Check the output in {output_file}")
    else:
        logger.info(f"Error: The file {input_file} does not exist.")

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    words = [w.split(",") for w in words]

    # maybe a tad hacky: we sort our dataset so that it is ordered V1, V2, .... V10, V11 ....
    chars = sorted(list(set([i for word in words for i in word])), key=lambda x: int(x[1:]))

    max_word_length = max(len(w) for w in words)
    logger.info(f"number of examples in the dataset: {len(words)}")
    logger.info(f"max word length: {max_word_length}")
    logger.info(f"number of unique characters in the vocabulary: {len(chars)}")
    logger.info("vocabulary:")
    logger.info(chars)
    assert max_word_length <= args.max_output_length, f'block size too large {max_word_length} vs {args.max_output_length}'

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples

    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    logger.info(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, args.max_output_length)
    test_dataset = CharDataset(test_words, chars, args.max_output_length)
    return train_dataset, test_dataset

def write_samples(num=10, new_file=False, use_logger=False):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, temperature = args.temperature, top_k=top_k, do_sample=True).to('cpu')
    #logger.info(f"generated")
    n_samp =0
    max_samp=0
    sum_samp=0
    samples = []
#    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row) #hy: row element should be less than number of unique characters in vocabulary
        samples.append(word_samp)
    for s in samples:
        n_samp +=1
        sum_samp += len(s)
        max_samp = max(max_samp, len(s))
    out_file = args.dump_path + "/out.txt"
    #if use_logger:
        #logger.info("decoded")
        # logger.info(f"Printing {len(samples)} samples to {out_file}.")
    #else:
        # print(f"Printing {len(samples)} samples to {out_file}.")
    if not new_file:
        with open(out_file, "a") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    else:
        with open(out_file, "w") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    #logger.info("printed")
    return n_samp, sum_samp, max_samp









if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    Tris_path = './data/benchmark_instances/'
    #inst_file = 'random_instance_110_15_3'
    #inst_file = 'random_instance_93_40_10'
    inst_file = 'random_instance_444_15_10'


    num_pts = int(inst_file.split('_')[-1])
    db = Database({},{},num_pts)
    for i in range(1, args.max_epochs):
        if not os.path.isfile(f"{args.dump_path}/search_output_{i}-tokenized.txt"):
            break
    initial_gen = i-1
    if initial_gen ==0:
        #print("from pohang center file...")
        #local_search(db, args.dump_path, inst_file+'.json')
        print("from taehoon_hwi center file...")
        #### Instead of search.jl ######
        local_search(db, args.dump_path, inst_file+'.json')
        #### -------------------- ######
        tokenize(f"{args.dump_path}/search_output_1.txt", args.n_tokens)
        initial_gen = 1


    logger.info(f"initializing at generation: {initial_gen}")
    input_file = args.dump_path + f"/search_output_{initial_gen}-tokenized.txt"
    train_dataset, test_dataset = create_datasets(input_file)
    #vocab_size = args.n_tokens + 1
    vocab_size = train_dataset.get_vocab_size() #hy
    block_size = args.max_output_length + 1
    logger.info(f"dataset determined that: {vocab_size=}, {block_size=}")


    args.device = "cpu" if args.cpu else "cuda"
    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        logger.error(f'model type {args.type} is not recognized')
    model.to(args.device)
    logger.info(f"model #params: {sum(p.numel() for p in model.parameters())}")
    model_path = os.path.join(args.dump_path, "model.pt")

    for generation in range(initial_gen,args.max_epochs + 1):
        logger.info(f"============ Start of generation {generation} ============")
        logger.info(f"cuda Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, cuda reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")

        logger.info("training")
        # python makemoretokens.py --i search_output_1-tokenized.txt --device cuda
        #train_makemore()
        # init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

        # init dataloader
        #batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)#hy
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, num_workers=0)

        # training loop
        best_loss = None
        step = 0
        while True:

            t0 = time.time()

            # get the next batch, ship to device, and unpack it to input and target
            batch = batch_loader.next()
            batch = [t.to(args.device) for t in batch]
            X, Y = batch

            # feed into the model
            try:
                logits, loss = model(X, Y)
                # calculate the gradient, update the weights
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                logger.info("Caught RuntimeError during forward pass.")
                logger.info(f"Shape of x before error: {X.shape}")
                logger.info(f"Shape of y before error: {Y.shape}")
                logger.info(f"Shape of logits (if calculated): {logits.shape if 'logits' in locals() else 'Not calculated'}")

                #raise e



            # wait for all CUDA work on the GPU to finish then calculate iteration time taken
            if args.device =="cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            # logging
            if step % 100 == 0:
                logger.info(f"step {step}/{args.max_steps} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

            # evaluate the model
            if step > 0 and step % 500 == 0:
                train_loss = evaluate(model, train_dataset, args.device, batch_size=100, max_batches=10)
                test_loss  = evaluate(model, test_dataset,  args.device, batch_size=100, max_batches=10)
                logger.info(f"step {step} train loss: {train_loss} test loss: {test_loss}")
                # save the model to disk if it has improved
                if best_loss is None or test_loss < best_loss:
                    out_path = os.path.join(args.dump_path, "model.pt")
                    logger.info(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                    torch.save(model.state_dict(), out_path)
                    best_loss = test_loss
    #            print_samples(num=10)

            step += 1
            # termination conditions
            if args.max_steps >= 0 and step >= args.max_steps:
                break
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")

        logger.info('generating')
        sample_batch_size =args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
        todo = args.sample_only
        tot_n = 0
        tot_sum = 0
        tot_max = 0
        out_file = args.dump_path + "/out.txt"
        in_file = args.dump_path + f"/search_output_{generation}-tokenized.txt"
        #infilz = f"{args.dump_path}/search_output_{generation}.txt"
        with open(in_file, 'r') as f:
            data = f.read()
        words = data.splitlines()
        with open(out_file, "w") as file:
            for word in words:
                file.write(word)
                file.write("\n")
        while sample_batch_size < todo:
            if todo % 50000 ==0 :
                logger.info(f'{todo} samples remaining')
            n, sm, mx = write_samples(num=sample_batch_size)
            tot_n+=n
            tot_sum+=sm
            tot_max = max(tot_max,mx)
            todo = todo - sample_batch_size
        n, sm, mx = write_samples(num=todo)
        tot_n+=n
        tot_sum+=sm
        tot_max = max(tot_max,mx)
        logger.info(f"distribution of sample lengths: average: {tot_sum/tot_n if tot_n != 0 else 0} max: {tot_max}")
        logger.info('decoding')
        decode()
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        logger.info(f"============ End of generation {generation} ============")
        #logger.info(f"launching search.jl")
        #os.environ["JULIA_NUM_THREADS"] = str(args.nb_threads)  # Set the environment variable
        #logger.info(f"JULIA_NUM_THREADS is set to {os.environ['JULIA_NUM_THREADS']}")

        #subprocess.run(["julia", "search_pfd.jl", args.dump_path, str(args.nb_local_searches), str(args.num_initial_empty_objects), str(args.final_database_size), str(args.target_db_size), '-i', args.dump_path + '/transformer-output-decoded.txt'])

        #### Instead of search.jl ######
        local_search_from_decoded(db, inst_file, args.dump_path,'/transformer-output-decoded.txt')
        #### -------------------- ######



        if os.path.exists(args.dump_path+"/distribution.txt"):
            with open(args.dump_path+"/distribution.txt", 'r') as file:
                d_lines = file.readlines()
        logger.info("distribution of scores")
        for l in d_lines:
            logger.info(l[:-1])


        logger.info("tokenizing")
        tokenize(f"{args.dump_path}/search_output_{generation+1}.txt", args.n_tokens)
        input_file = args.dump_path + f"/search_output_{generation+1}-tokenized.txt"
        train_dataset, test_dataset = create_datasets(input_file)
