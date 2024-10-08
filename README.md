# Artifact for SC24

---
To build iGUARD: from the `iGUARD-SOSP21` directory, run `make setup`.


The following make commands will generate each figure for comparison. Within the `results/paper_figures` directory are copies of the baseline figures for comparison.

`make table1` (~4-6 hours) \
This will run `scripts/hirace_experiments.py` to run all of the correctness results shown in table 1. The data will be stored in `results/hirace_correctness_results.sqlite3`, and when complete the database will be queried to generate the table with `scripts/gen_table1.py`.

If the run needs to be restarted, use `make -B table` to resume the tests where they were stopped.


`make fig5` (~8-12 hours) \
This will perform the same tests as above, but will first download a set of larger graphs and then test them on a smaller code set, stored in `results/hirace_perf_results_speedup.sqlite3`.


`make fig6` (~2-6 hours) \
Tests a few domain specific codes for performance comparisons.

`make clean` \
Cleans some build files and large graph files

`make clean-all` \
Additionally removes generated results
