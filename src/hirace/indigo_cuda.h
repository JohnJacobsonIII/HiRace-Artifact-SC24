/* This file is part of the Indigo benchmark suite version 0.9.

Copyright 2021, Texas State University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Contributors: Yiqian Liu, Noushin Azami, Corbin Walters, and Martin Burtscher

URL: The latest version of the Indigo benchmark suite is available at
https://cs.txstate.edu/~burtscher/research/IndigoSuite/.
 */


#include <cuda.h>
#include <cstdlib>
#include <cstdio>


/************ HIRACE INCLUDE BEGIN ************/
#include "HiRace.h"

using data_ptr = HiRaceDataWrap<data_t>;
/************ HIRACE INCLUDE BEGIN ************/

/****************************************************************************************/

struct ECLgraph {
  int nodes;
  int edges;
  int* nindex;
  int* nlist;
};

ECLgraph readECLgraph(const char* const fname)
{
  ECLgraph g;
  int cnt;

  FILE* f = fopen(fname, "rb");  if (f == NULL) {fprintf(stderr, "ERROR: could not open file %s\n\n", fname);  exit(-1);}
  cnt = fread(&g.nodes, sizeof(g.nodes), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to read nodes\n\n");  exit(-1);}
  cnt = fread(&g.edges, sizeof(g.edges), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to read edges\n\n");  exit(-1);}
  printf("input graph: %d nodes and %d edges\n", g.nodes, g.edges);
  if ((g.nodes < 1) || (g.edges < 0)) {fprintf(stderr, "ERROR: node or edge count too low\n\n");  exit(-1);}

  g.nindex = (int*)malloc((g.nodes + 1) * sizeof(g.nindex[0]));
  g.nlist = (int*)malloc(g.edges * sizeof(g.nlist[0]));

  if ((g.nindex == NULL) || (g.nlist == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

  cnt = fread(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);  if (cnt != g.nodes + 1) {fprintf(stderr, "ERROR: failed to read neighbor index list\n\n");  exit(-1);}
  cnt = fread(g.nlist, sizeof(g.nlist[0]), g.edges, f);  if (cnt != g.edges) {fprintf(stderr, "ERROR: failed to read neighbor list\n\n");  exit(-1);}
  fclose(f);
  return g;
}

void freeECLgraph(struct ECLgraph* g)
{
  if (g->nindex != NULL) free(g->nindex);
  if (g->nlist != NULL) free(g->nlist);
  g->nindex = NULL;
  g->nlist = NULL;
}

/****************************************************************************************/

__global__ void test_kernel(int* nindex, int* nlist, data_t* data1, data_t* data2, int n, hr_shadowt* metadata1, hr_shadowt* metadata2);
void serial_code(int* nindex, int* nlist, data_t* data1, data_t* data2, int n);
int verify_result(int* nindex, int* nlist, data_t* h_data1, data_t* h_data2, data_t* d_data1, data_t* d_data2, int n, int e, int blocks, int threadsperblock);

/****************************************************************************************/

int main (int argc, char* argv[])
{
  // process command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s input_file_name threads_per_block num_blocks\n", argv[0]); exit(-1);}
  ECLgraph g = readECLgraph(argv[1]);
  int threadsPerBlock = atoi(argv[2]);
  if ((threadsPerBlock < 1) || (threadsPerBlock > 1024)) {fprintf(stderr, "ERROR: threads_per_block must be at least 1 and at most 1024\n"); exit(-1);}
  int numBlocks = atoi(argv[3]);
  if (numBlocks < 1) {fprintf(stderr, "ERROR: num_blocks must be at least 1\n"); exit(-1);}
  
  // allocate and init two data arrays
  int n = g.nodes;
  int e = g.edges;
  if (e == 0) {
    e = 1;
  }
  int s = ((n > e) ? n : e);
  data_t* data1 = (data_t*)malloc(s * sizeof(data_t));
  data_t* data2 = (data_t*)malloc(s * sizeof(data_t));
  data_t* h_data1 = (data_t*)malloc(s * sizeof(data_t));
  data_t* h_data2 = (data_t*)malloc(s * sizeof(data_t));
  data1[0] = 0;
  h_data1[0] = 0;
  data2[0] = 0;
  h_data2[0] = 0;
  for (int i = 1; i < s; i++) {
    data1[i] = rand() % n;
    h_data1[i] = data1[i];
  }
  for (int i = 1; i < s; i++) {
    data2[i] = rand() % e;
    h_data2[i] = data2[i];
  }

  // allocate and copy GPU data
  int* d_nindex;
  int* d_nlist;
  data_t* d_data1;
  data_t* d_data2;
  cudaMalloc((void**) &d_nindex, (n + 1) * sizeof(int));
  cudaMalloc((void**) &d_nlist, e * sizeof(int));
  cudaMalloc((void**) &d_data1, s * sizeof(data_t));
  cudaMalloc((void**) &d_data2, s * sizeof(data_t));
  cudaMemcpy(d_nindex, g.nindex, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nlist, g.nlist, e * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data1, data1, s * sizeof(data_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data2, data2, s * sizeof(data_t), cudaMemcpyHostToDevice);
  
  /************ HIRACE INIT BEGIN ************/
  HIRACE_SHADOW_DECL(d_data1)
  HIRACE_SHADOW_DECL(d_data2)

  HIRACE_MALLOC(d_data1, s)
  HIRACE_MALLOC(d_data2, s)
  
  HIRACE_MEMSET(d_data1, s)
  HIRACE_MEMSET(d_data2, s)
  /************ HIRACE INIT END ************/

  // run kernel
  test_kernel<<<numBlocks, threadsPerBlock>>>(d_nindex, 
                                              d_nlist, 
                                              d_data1, 
                                              d_data2, 
                                              n, 
                                              __hr_metadata_d_data1, // HIRACE PARAM
                                              __hr_metadata_d_data2); // HIRACE PARAM

  cudaDeviceSynchronize();
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda err: %s\n", cudaGetErrorString(err));
  }
  
  // copy result back to CPU
  cudaMemcpy(data1, d_data1, s * sizeof(data_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(data2, d_data2, s * sizeof(data_t), cudaMemcpyDeviceToHost);
  
  // check result
  serial_code(g.nindex, g.nlist, h_data1, h_data2, n);

  int ret = verify_result(g.nindex, g.nlist, h_data1, h_data2, data1, data2, n, e, numBlocks, threadsPerBlock);
  if (ret == 1) {
    printf("result matches serial code\n");
  } else if (ret == 0) {
    printf("result differs from serial code\n");
  }
  

  // cleanup
  cudaFree(d_nindex);
  cudaFree(d_nlist);
  cudaFree(d_data1);
  cudaFree(d_data2);
  free(data1);
  free(data2);
  free(h_data1);
  free(h_data2);
  freeECLgraph(&g);
  
  /************ HIRACE TEARDOWN BEGIN ************/
  HIRACE_CUDA_FREE(d_data1)
  HIRACE_CUDA_FREE(d_data2)
  /************ HIRACE TEARDOWN END ************/

  cudaDeviceReset();
  
  return 0;
}
