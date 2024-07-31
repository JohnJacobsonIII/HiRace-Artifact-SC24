#!/bin/bash
nvcc -I../indigo/indigo_include ../indigo/indigo_sources/conditional_edge_neighbors_block/conditional_edge_neighbors_block_atomicBug.cu -o _ig_tst
nvcc -I../src/hirace -DRACECHECK ../indigo/indigo_hirace_sources/conditional_edge_neighbors_block/conditional_edge_neighbors_block_hirace_atomicBug.cu -o _hr_tst

LD_PRELOAD=../iGUARD-SOSP21/nvbit_release/tools/detector/detector.so   ./_ig_tst ../indigo/input/DAG_100n_200e.egr 256 1024
./_hr_tst ../indigo/input/DAG_100n_200e.egr 256 1024

rm _ig_tst
rm _hr_tst
