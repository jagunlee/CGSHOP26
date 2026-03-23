# Algorithm for optimization

Algorithms for Solution Optimization for the CG:SHOP 2026 competition. (related to section 3)

## How to run

1. First, put the instance file(s) into [`data/benchmark_instances`](../data/benchmark_instances), and solution file(s) into `solutions` (if necessary).

2. Then, use the following command :
* run all in serial
```bash
python main.py --data solutions/[solution file] --fcg_serial t --replace_pr f
```
* run all in parallel
```bash
python main.py --data solutions/[solution file] --fcg_pr1 t --fcg_pr2 t --replace_pr t --cpus [number of cpus] --ch_size [chunk size]
```
* `fcg_pr1`, `fcg_pr2`, `replace_pr` : Whether to parallelize the operation (t/f)
* `chunk_size`: You need it when --fcg_pr2 t. We recommand 25%~50% of the number of edges
* For exmaple:
```bash
python main.py --data solutions/random_instance_73_160_10.solution.json --fcg_pr1 t --fcg_pr2 t --replace_pr t --cpus 5 --ch_size 200
or
python main.py --data solutions/random_instance_73_160_10.solution.json --fcg_pr1 f --fcg_pr2 t --replace_pr t --cpus 5 --ch_size 200
or
python main.py --data solutions/random_instance_73_160_10.solution.json --fcg_pr1 f --fcg_pr2 f --replace_pr t --cpus 5
or
python main.py --data solutions/random_instance_73_160_10.solution.json --fcg_serial t --replace_pr t --cpus 5
```

3. The results will be stored in the `/opt` folder.
