# Algorithm for optimization

Algorithms for Solution Optimization for the CG:SHOP 2026 competition. (related to section 3)

## How to run

1. First, put the instance file(s) into `data/benchmark_instances`, and solution file(s) into `solutions` (if necessary).

2. Then, use the following command :

```bash
python main.py --fcg_old [t/f] --fcg_pr1 [t/f] --fcg_pr2 [t/f] --replace_pr [t/f] --cpus [number of cpus] --ch_size [chunk size] --data [solution file for optimization]
```
* `fcg_old`, `fcg_pr1`, `fcg_pr2`, `replace_pr` : Whether to parallelize the operation (T/F)

3. The results will be stored in the `/optimal` folder.