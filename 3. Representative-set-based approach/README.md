# Representative-set-based approach

Representative-set-based solvers for the CG:SHOP 2026 competition. (related to section 4)

* `builder` : Generates representative set from benchmark instances
* `initial_solver` : Finds centers using generated representative set
* `analyzer` : Produces size and quality analysis reports

## Quick Start

To run the pipeline, use the following command examples:

```bash
# 1. Build coresets
python coreset/builder.py -b data/benchmark_instances -c data/coreset_instances

# 2. Initial solver
python coreset/initial_solver.py -v result.csv -l coreset/logs/solve_global.log

# 3. Analysis report
python coreset/analyzer.py -l coreset/logs/solve_global.log